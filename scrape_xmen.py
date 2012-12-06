import bs4
import nltk
import requests
import sys

# search Wikipedia for a subject or provide its URL, return the parsed HTML
# spoof the user-agent, let's pretend we're Firefox :)
def wikipedia_search(subject, url=False):
    if url is False :
        response = requests.get('http://en.wikipedia.org/w/index.php',
                                params={'search':subject},
                                headers={'User-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.11 (KHTML, like Gecko)'})
    else :
        response = requests.get(url,
                                headers={'User-agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2) AppleWebKit/537.11 (KHTML, like Gecko)'})
    soup = bs4.BeautifulSoup(response.text)
    return soup

# search Wikipedia for the X-Men
# find list of members in side panel, return dictionary of names and URLs
def get_xmen_members():
    soup = wikipedia_search('X-men')
    infobox = soup.find('table', class_='infobox')
    members = infobox.find('th', text='Member(s)')
    members = members.next_sibling.next_sibling
    xmen = {}
    for member in members.find_all('a') :
        xmen[member.get_text()] = 'http://en.wikipedia.org'+member.get('href')
    return xmen

# take parsed HTML for X-man's Wikipedia page
# return list of abilities, lightly cleaned
def get_xmen_abilities(soup):
    infobox = soup.find('table', class_='infobox')
    if infobox is not None :
        abilities = infobox.find('th', text='Abilities')
        if abilities is not None :
            abilities_list = abilities.next_sibling.next_sibling.find_all(text=True)
            abilities_list = [item.strip("\n") for item in abilities_list if item!='' and item!='\n']
            return abilities_list
    else : return []

# take parsed HTML for X-man's Wikipedia page
# return all text, excluding References and downward
def get_xmen_text(soup):
    
    #en_stopwords = set(nltk.corpus.stopwords.words('english'))
    raw = nltk.clean_html(str(soup))
    raw_trunc = raw[:raw.rfind('References')]
    sents = nltk.sent_tokenize(raw_trunc)
    words = [nltk.word_tokenize(sent) for sent in sents]
    poss = [nltk.pos_tag(word) for word in words]
    #nes = [nltk.ne_chunk(pos, binary=True) for pos in poss]
    #for pos in poss: print pos
    poss_filter = [filter_insignificant(pos, tag_suffixes=['DT']) for pos in poss]
    print poss_filter
    nes = [nltk.ne_chunk(poss_filter, binary=True) for pos in poss_filter]
    
    def sub_leaves(tree, node):
        return [t.leaves() for t in tree.subtrees (lambda s: s.node == node)]
    
    people = [sub_leaves(ne, 'NE') for ne in nes]
    people = [item for sublist in people
              for subsublist in sublist
              for subsubsublist in subsublist
              for item in subsubsublist
              if item not in ('NNP', 'NN', 'NNPS', 'JJ')]
    people = merge_people(people)
    fd = nltk.FreqDist(person for person in people if person!='Magneto')
    fd.plot(50)


def filter_insignificant(chunk, tag_suffixes=['DT', 'CC']):
    good = []
    for word, tag in chunk:
        ok = True
        for suffix in tag_suffixes:
            if tag.endswith(suffix):
                ok = False
                break
        if ok:
            good.append((word, tag))
    return good

def merge_people(people):
    pairs = [('Charles', 'Xavier'), ('Emma', 'Frost'), ('Scarlet', 'Witch'),
             ('Captain', 'America'), ('Jean', 'Grey'), ('Doctor', 'Strange'),
             ('Iron', 'Man'), ('Doctor', 'Doom'), ('New', 'Avengers'),
             ('Asteroid', 'M'), ('Savage', 'Land'), ('Hellfire', 'Club'),
             ('Utopia', 'Island'), ('Hope', 'Summers'),
             ('White', 'King'), ('Grey', 'King'), ('Michael', 'Xavier'),
             ('Stan', 'Lee'), ('Marvel', 'Comics')]
    for pair in pairs :
        bigrams = nltk.bigrams(people)
        inds = [ind for ind, bigram in enumerate(bigrams) if bigram==pair]
        for ind in inds :
            people[ind] = ' '.join(pair)
        people = [person for ind, person in enumerate(people)
                  if ind not in [ind+1 for ind in inds]]
    return people
    

#def process_xmen_text(raw_text):
#    tokens = nltk.word_tokenize(raw_text)

if __name__ == '__main__':
    
    magneto = wikipedia_search("magneto_(comics)")
    get_xmen_text(magneto)
    sys.exit()
    
    xmen = get_xmen_members()
    xmen_abilities = {}
    for xman in xmen :
        html = wikipedia_search(xman, xmen[xman])
        abilities = get_xmen_abilities(html)
        xmen_abilities[xman] = abilities
        print "\n", xman, "\n", xmen_abilities[xman]
    