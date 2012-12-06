from __future__ import division
import nltk
import sys
import re, pprint

which = sys.argv[1]
print "Exercise", which

# Exercise 2.4
if which == '2.4' :
    from nltk.corpus import state_union
    targets = ['men','women','people']
    cdf = nltk.ConditionalFreqDist(
        (target, fileid[:4])
        for fileid in state_union.fileids()
        for word in state_union.words(fileid)
        for target in targets
        if word.lower() == target)
    
    cdf.plot()

# Exercise 2.5
if which == '2.5' :
    from nltk.corpus import wordnet as wn
    nouns = ['school', 'tree', 'car']
    for noun in nouns :
        synsets = wn.synsets(noun)
        print "\n", noun
        for synset in synsets :
            print "...", synset, synset.definition
            print "...... part meronyms:", synset.part_meronyms()
            print "...... substance meronyms:", synset.substance_meronyms()
            print "...... member meronyms:", synset.member_meronyms()
            print "...... part holonyms:", synset.part_holonyms()
            print "...... substance holonyms:", synset.substance_holonyms()
            print "...... member holonyms:", synset.member_holonyms()


# Exercise 2.7
if which == '2.7' :
    from nltk.book import *
    word = 'however'
    print "\nWORD =", word.upper()
    texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9]
    for text in texts :
        print text, "\n"
        print text.concordance(word, lines=25)


# Exercise 2.8
if which == '2.8' :
    names = nltk.corpus.names
    cdf = nltk.ConditionalFreqDist(
        (fileid, name[0])
        for fileid in names.fileids()
        for name in names.words(fileid))
    
    cdf.plot()

# Exercise 2.10
if which == '2.10' :
    from nltk.book import *
    texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9]
    for text in texts :
        print "\n", text
        fd = nltk.FreqDist([word for word in text if word.isalpha()])
        top_word_count = 0
        for item in fd.items()[:20] :
            top_word_count += item[1]
        n_tokens = len([word for word in text if word.isalpha()])
        print "total word count:", n_tokens
        print "top word count:", top_word_count
        print "top word percent:", round(100.0*top_word_count/float(n_tokens), 4)
        

# Exercise 2.13
if which == '2.13' :
    from nltk.corpus import wordnet as wn
    all_synsets = [synset for synset in wn.all_synsets('n')]
    no_hypos = [synset for synset in wn.all_synsets('n') if len(synset.hyponyms())==0]
    print "percentage of synsets without hyponyms:",
    print round(100.0*len(no_hypos)/float(len(all_synsets)), 4), "%"


# Exercise 2.15
if which == '2.15' :
    from nltk.corpus import brown
    fdist = nltk.FreqDist([w.lower() for w in brown.words()])
    # all words with counts >= 3
    plus3s = [w.lower() for w in brown.words() if fdist[w]>=3]
    print "number of words with at least three mentions in the brown corpus:", len(plus3s)


# Exercise 3.8
if which == '3.8' :
    import requests
    def GetUrl(url):
        r = requests.get(url)
        raw = r.text
        return raw
    
    url = "http://www.google.com"
    text = GetUrl(url)
    print text


# Exercise 3.10
if which == '3.10' :
    sent = ['The', 'dog', 'gave', 'John', 'the', 'newspaper']
    result = [(word, len(word)) for word in sent]
    print result


# Exercise 3.18
if which == '3.18' :
    from nltk.corpus import gutenberg
    raw = gutenberg.raw('austen-sense.txt')
    words = nltk.word_tokenize(raw)
    whs = [word for word in words if word.startswith('wh')]
    print whs


# Exercise 3.21
if which == '3.21' :
    import requests
    url = 'http://www.decayfilm.com'
    r = requests.get(url)
    raw = r.text
    clean = nltk.clean_html(raw)
    allwords = re.findall(r'\w+', clean)
    words = set([word.lower() for word in allwords])
    english_vocab = set([w.lower for w in nltk.corpus.words.words()])
    unusual = words.difference(english_vocab)
    print sorted(unusual)


# Exercise 3.24
if which == '3.24' :
    try :
        text = sys.argv[2]
    except IndexError :
        text = 'Hello, World'
    def Text2Hacker(text):
        hack = re.sub(r'e', '3', text)
        hack = re.sub(r'i', '1', hack)
        hack = re.sub(r'o', '0', hack)
        hack = re.sub(r's', '5', hack)
        hack = re.sub(r'l', '|', hack)
        return hack
    
    print Text2Hacker(text)


# Exercise 3.25
if which == '3.25' :
    try :
        text = sys.argv[2]
    except IndexError :
        text = 'Hello, World'
    def English2PigLatin(text):
        words = nltk.word_tokenize(text)
        new_words = []
        for word in words :
            if re.match(r'^\W', word) is not None :
                new_words.append(word)
                continue
            pattern = r'^(qu|[^aeiouAEIOU]+)'
            re_indices = [m.span() for m in re.finditer(pattern, word)]
            new = word[re_indices[0][1]:] + word[:re_indices[0][1]] + "ay"
            new_words.append(new)
        new_words = ' '.join(new_words)
        return new_words
    piglatin = English2PigLatin(text)
    print piglatin
    

# Exercise 3.29
if which == '3.29' :
    import numpy as np
    from nltk.corpus import brown
    for category in brown.categories() :
        sents = brown.sents(categories=category)
        words = brown.words(categories=category)
        sent_mean = np.mean([len(sent) for sent in sents])
        word_mean = np.mean([len(word) for word in words])
        ari = 4.71*word_mean + 0.5*sent_mean - 21.43
        print category, ari


# Exercise 6.2
if which == '6.2' :
    from nltk.corpus import names
    #from nltk.classify import NaiveBayesClassifier
    import random
    names = ([(name, 'male') for name in names.words('male.txt')] +
             [(name, 'female') for name in names.words('female.txt')])
    random.shuffle(names)
    def get_features(name):
        features = {}
        features['first_letter'] = name[0]
        #features['last_letter'] = name[-1]
        features['last_2_letters'] = name[-2:]
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features['count(%s)' % letter] = name.lower().count(letter)
            features['has(%s)' % letter] = (letter in name.lower())
        return features
    
    features = [(get_features(name), gender) for (name, gender) in names]
    train_set = features[:int(len(features)*0.8)]
    cv_set = features[int(len(features)*0.8):int(len(features)*0.9)]
    cv_set_names = names[int(len(features)*0.8):int(len(features)*0.9)]
    test_set = features[int(len(features)*0.9):]
    nbc = nltk.NaiveBayesClassifier.train(train_set)
    errors = []
    for (name,tag) in cv_set_names :
        guess = nbc.classify(get_features(name))
        if guess != tag :
            errors.append( (tag, guess, name) )
    #print errors
    print "NBC accuracy:", nltk.classify.accuracy(nbc, test_set)
    print nbc.show_most_informative_features(10)


# Exercise 6.4
if which == '6.4' :
    try :
        which_classifier = sys.argv[2]
    except IndexError :
        which_classifier = "NaiveBayes"
    from nltk.corpus import movie_reviews
    import random
    docs = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(categories=category)]
    random.shuffle(docs)
    all_words = nltk.FreqDist(word.lower() for word in movie_reviews.words())
    top_words = all_words.keys()[:2000]
    def get_features(doc):
        doc_words = set(doc)
        features = {}
        for word in top_words :
            features['contains(%s)' % word] = (word in doc_words)
        return features
    
    features = [(get_features(doc), cat) for (doc, cat) in docs]
    train_set = features[:int(len(docs)*0.8)]
    test_set = features[int(len(docs)*0.8):]
    if which_classifier == "NaiveBayes" :
        print "\nNaive Bayes Classifier"
        nbc = nltk.NaiveBayesClassifier.train(train_set) 
        print "accuracy:", nltk.classify.accuracy(nbc, test_set)
        print nbc.show_most_informative_features(30)
    elif which_classifier == "DecisionTree" :
        print "\nDecision Tree Classifier"
        dtc = nltk.DecisionTreeClassifier.train(train_set, binary=False,
                                                entropy_cutoff=0.1, depth_cutoff=75)
        print "accuracy:", nltk.classify.accuracy(dtc, test_set)
    elif which_classifier == "MaximumEntropy" :
        print "\nMaximum Entropy Classifier"
        mec = nltk.MaxentClassifier.train(train_set, algorithm='gis', max_iter=20)
        mec.show_most_informative_features(n=10)
        print "accuracy:", nltk.classify.accuracy(mec, test_set)

















