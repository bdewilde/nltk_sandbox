import bs4
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import re
import readability
import requests
import scipy as sp
import sys

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.95 Safari/537.11'

# search Google for a topic
# return parsed HTML
def google(topic) :
    response = requests.get('http://google.com/search',
                            params={'q':topic},
                            headers={'User-agent':USER_AGENT})
    soup = bs4.BeautifulSoup(response.text)
    print "\nGOOGLE :", response.url
    return soup


def get_top_search_results(soup) :
    search = soup.find('div', id='search')
    lis = search.find_all('li', class_='g', id=False)
    search_results = []
    for li in lis :
        h3 = li.find('h3', class_='r')
        if h3 is not None :
            a = h3.find('a')
            search_results.append(a.get('href'))
    return search_results


def sub_leaves(tree, node):
    return [t.leaves() for t in tree.subtrees (lambda s: s.node == node)]


def parse_page_text(url) :
    response = requests.get(url, headers={'User-agent':USER_AGENT})
    html = response.text
    readable_html = readability.readability.Document(html)
    try :
        article_only = readable_html.summary()
    except :
        return []
    raw = nltk.clean_html(article_only)
    #soup = bs4.BeautifulSoup(html)
    #raw = nltk.clean_html(str(soup))
    sents = nltk.sent_tokenize(raw)
    sents = [nltk.wordpunct_tokenize(sent) for sent in sents]
    #sents = [nltk.tokenize.WhitespaceTokenizer().tokenize(sent) for sent in sents]
    tagged_sents = [nltk.pos_tag(sent) for sent in sents]
    
    # get interesting collocations
    #words = nltk.wordpunct_tokenize(raw)
    words = nltk.tokenize.WhitespaceTokenizer().tokenize(raw)
    words = [word.lower() for word in words]
    punctuation = re.compile(r'[-.?!,":;()]')
    good_words = [punctuation.sub("", word) for word in words]
    bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(good_words)
    trigram_finder = nltk.collocations.TrigramCollocationFinder.from_words(good_words)
    bigram_finder.apply_freq_filter(2)
    trigram_finder.apply_freq_filter(1)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    collocations = []
    collocations.extend(bigram_finder.nbest(bigram_measures.pmi, 10))
    collocations.extend(trigram_finder.nbest(trigram_measures.pmi, 10))
    print "\nCOLLOCATIONS :", collocations
    
    # get named entities
    ne_chunks = [nltk.ne_chunk(sent, binary=True) for sent in tagged_sents]
    nes = [sub_leaves(ne_chunk, 'NE') for ne_chunk in ne_chunks]
    entities = []
    for ne in nes :
        if len(ne) == 0 : continue
        ne_string = ''
        for pairs in ne :
            for pair in pairs :
                ne_string = ' '.join((ne_string, pair[0]))
        entities.append(ne_string[1:])
    print "\nNES :", entities
    
    # get noun phrases
    nps = []
    grammar = r"""
        NP: {<PP\$>? <JJ>* <NN.*>+} # NP
        P: {<IN>}           # Preposition
        V: {<V.*>}          # Verb
        PP: {<P> <NP>}      # PP -> P NP
        VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*
    """
    cp = nltk.RegexpParser(grammar)
    for sent in tagged_sents :
        tree = cp.parse(sent)
        for subtree in tree.subtrees() :
            if subtree.node == 'NP' :
                try : subtree = str(subtree).split()[1:]
                except UnicodeEncodeError : continue # HACK HACK HACK
                subtree = ' '.join([item.split('/')[0] for item in subtree])
                nps.append(subtree)
    print "\nNPS :", nps
    return nps


def plot_freq_dist(fd, title=False) :
    fig = plt.figure("fd", figsize=(12,6), dpi=150, facecolor='white', edgecolor='white')
    ax = fig.add_subplot(1,1,1)
    if title is not False :
        ax.set_title(title)
    plot = ax.plot(range(0, len(fd.keys())), [fd[key] for key in fd.keys()],
                   linewidth=3, color='red')
    x_ticks = []
    x_ticklabels = []
    for i, label in enumerate(fd.keys()) :
        x_ticks.append(i)
        x_ticklabels.append(label)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, rotation='vertical')
    ax.set_ylim(0, fd[fd.keys()[0]]+1)
    ax.set_xlim(0,20)
    ax.set_ylabel('Number of Mentions')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.35)
    plt.show()


if __name__ == '__main__':
    
    try :
        search = sys.argv[1]
    except IndexError :
        print '\nERROR: Please provide search term on command line'
        sys.exit()
    
    google_soup = google(search)
    search_results = get_top_search_results(google_soup)
    meta_fd = nltk.FreqDist()
    for search_result in search_results :
        print "\n", "#"*75
        print "URL("+str(search_results.index(search_result))+")", search_result
        nps = parse_page_text(search_result)
        fd = nltk.FreqDist(np.lower() for np in nps)
        #plot_freq_dist(fd)
        for sample in fd.samples() :
            meta_fd.inc(sample, count=fd[sample])
    plot_freq_dist(meta_fd, "Key Phrases in Google Search Results for \""+search+"\"")
        
    