# -*- coding:utf-8 -*
f = open('1.txt',"r") #open the file located at "path" as a file object (f) that is readonly
docall = f.read().decode('utf8') # read raw text into a variable (raw) after decoding it from utf8
f.close()
#print docall
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
stop = set(stopwords.words('french'))
exclude = set(string.punctuation)

def get_tokens(raw,encoding='utf8'):
    no_commas = re.sub(r'[.|,|\']', ' ', raw)  # filter out all the commas, periods, and appostrophes using regex
    tokens = nltk.word_tokenize(no_commas)  # generate a list of tokens from the raw text
    #text = nltk.Text(tokens, encoding)  # create a nltk text from those tokens
    #return text
    return tokens


def get_stopswords(type="veronis"):
    '''returns the veronis stopwords in unicode, or if any other value is passed, it returns the default nltk french stopwords'''
    if type == "veronis":
        # VERONIS STOPWORDS
        raw_stopword_list = ["Ap.", "Apr.", "GHz", "MHz", "USD", "a", "afin", "ah", "ai", "aie", "aient", "aies", "ait",
                             "alors", "après", "as", "attendu", "au", "au-delà", "au-devant", "aucun", "aucune",
                             "audit", "auprès", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras",
                             "aurez", "auriez", "aurions", "aurons", "auront", "aussi", "autour", "autre", "autres",
                             "autrui", "aux", "auxdites", "auxdits", "auxquelles", "auxquels", "avaient", "avais",
                             "avait", "avant", "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons",
                             "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était", "car", "ce",
                             "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là",
                             "celui", "celui-ci", "celui-là", "celà", "cent", "cents", "cependant", "certain",
                             "certaine", "certaines", "certains", "ces", "cet", "cette", "ceux", "ceux-ci", "ceux-là",
                             "cf.", "cg", "cgr", "chacun", "chacune", "chaque", "chez", "ci", "cinq", "cinquante",
                             "cinquante-cinq", "cinquante-deux", "cinquante-et-un", "cinquante-huit", "cinquante-neuf",
                             "cinquante-quatre", "cinquante-sept", "cinquante-six", "cinquante-trois", "cl", "cm",
                             "cm²", "comme", "contre", "d", "d'", "d'après", "d'un", "d'une", "dans", "de", "depuis",
                             "derrière", "des", "desdites", "desdits", "desquelles", "desquels", "deux", "devant",
                             "devers", "dg", "différentes", "différents", "divers", "diverses", "dix", "dix-huit",
                             "dix-neuf", "dix-sept", "dl", "dm", "donc", "dont", "douze", "du", "dudit", "duquel",
                             "durant", "dès", "déjà", "e", "eh", "elle", "elles", "en", "en-dehors", "encore", "enfin",
                             "entre", "envers", "es", "est", "et", "eu", "eue", "eues", "euh", "eurent", "eus", "eusse",
                             "eussent", "eusses", "eussiez", "eussions", "eut", "eux", "eûmes", "eût", "eûtes", "f",
                             "fait", "fi", "flac", "fors", "furent", "fus", "fusse", "fussent", "fusses", "fussiez",
                             "fussions", "fut", "fûmes", "fût", "fûtes", "g", "gr", "h", "ha", "han", "hein", "hem",
                             "heu", "hg", "hl", "hm", "hm³", "holà", "hop", "hormis", "hors", "huit", "hum", "hé", "i",
                             "ici", "il", "ils", "j", "j'", "j'ai", "j'avais", "j'étais", "jamais", "je", "jusqu'",
                             "jusqu'au", "jusqu'aux", "jusqu'à", "jusque", "k", "kg", "km", "km²", "l", "l'", "l'autre",
                             "l'on", "l'un", "l'une", "la", "laquelle", "le", "lequel", "les", "lesquelles", "lesquels",
                             "leur", "leurs", "lez", "lors", "lorsqu'", "lorsque", "lui", "lès", "m", "m'", "ma",
                             "maint", "mainte", "maintes", "maints", "mais", "malgré", "me", "mes", "mg", "mgr", "mil",
                             "mille", "milliards", "millions", "ml", "mm", "mm²", "moi", "moins", "mon", "moyennant",
                             "mt", "m²", "m³", "même", "mêmes", "n", "n'avait", "n'y", "ne", "neuf", "ni", "non",
                             "nonante", "nonobstant", "nos", "notre", "nous", "nul", "nulle", "nº", "néanmoins", "o",
                             "octante", "oh", "on", "ont", "onze", "or", "ou", "outre", "où", "p", "par", "par-delà",
                             "parbleu", "parce", "parmi", "pas", "passé", "pendant", "personne", "peu", "plus",
                             "plus_d'un", "plus_d'une", "plusieurs", "pour", "pourquoi", "pourtant", "pourvu", "près",
                             "puisqu'", "puisque", "q", "qu", "qu'", "qu'elle", "qu'elles", "qu'il", "qu'ils", "qu'on",
                             "quand", "quant", "quarante", "quarante-cinq", "quarante-deux", "quarante-et-un",
                             "quarante-huit", "quarante-neuf", "quarante-quatre", "quarante-sept", "quarante-six",
                             "quarante-trois", "quatorze", "quatre", "quatre-vingt", "quatre-vingt-cinq",
                             "quatre-vingt-deux", "quatre-vingt-dix", "quatre-vingt-dix-huit", "quatre-vingt-dix-neuf",
                             "quatre-vingt-dix-sept", "quatre-vingt-douze", "quatre-vingt-huit", "quatre-vingt-neuf",
                             "quatre-vingt-onze", "quatre-vingt-quatorze", "quatre-vingt-quatre", "quatre-vingt-quinze",
                             "quatre-vingt-seize", "quatre-vingt-sept", "quatre-vingt-six", "quatre-vingt-treize",
                             "quatre-vingt-trois", "quatre-vingt-un", "quatre-vingt-une", "quatre-vingts", "que",
                             "quel", "quelle", "quelles", "quelqu'", "quelqu'un", "quelqu'une", "quelque", "quelques",
                             "quelques-unes", "quelques-uns", "quels", "qui", "quiconque", "quinze", "quoi", "quoiqu'",
                             "quoique", "r", "revoici", "revoilà", "rien", "s", "s'", "sa", "sans", "sauf", "se",
                             "seize", "selon", "sept", "septante", "sera", "serai", "seraient", "serais", "serait",
                             "seras", "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sinon", "six",
                             "soi", "soient", "sois", "soit", "soixante", "soixante-cinq", "soixante-deux",
                             "soixante-dix", "soixante-dix-huit", "soixante-dix-neuf", "soixante-dix-sept",
                             "soixante-douze", "soixante-et-onze", "soixante-et-un", "soixante-et-une", "soixante-huit",
                             "soixante-neuf", "soixante-quatorze", "soixante-quatre", "soixante-quinze",
                             "soixante-seize", "soixante-sept", "soixante-six", "soixante-treize", "soixante-trois",
                             "sommes", "son", "sont", "sous", "soyez", "soyons", "suis", "suite", "sur", "sus", "t",
                             "t'", "ta", "tacatac", "tandis", "te", "tel", "telle", "telles", "tels", "tes", "toi",
                             "ton", "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente",
                             "trente-cinq", "trente-deux", "trente-et-un", "trente-huit", "trente-neuf",
                             "trente-quatre", "trente-sept", "trente-six", "trente-trois", "trois", "très", "tu", "u",
                             "un", "une", "unes", "uns", "v", "vers", "via", "vingt", "vingt-cinq", "vingt-deux",
                             "vingt-huit", "vingt-neuf", "vingt-quatre", "vingt-sept", "vingt-six", "vingt-trois",
                             "vis-à-vis", "voici", "voilà", "vos", "votre", "vous", "w", "x", "y", "z", "zéro", "à",
                             "ç'", "ça", "ès", "étaient", "étais", "était", "étant", "étiez", "étions", "été", "étée",
                             "étées", "étés", "êtes", "être", "ô"]
    else:
        # get French stopwords from the nltk kit
        raw_stopword_list = stopwords.words('french')  # create a list of all French stopwords
    stopword_list = [word.decode('utf8') for word in
                     raw_stopword_list]  # make to decode the French stopwords as unicode objects rather than ascii
    return stopword_list


def filter_stopwords(text, stopword_list):
    '''normalizes the words by turning them all lowercase and then filters out the stopwords'''
    words = [w.lower() for w in text]  # normalize the words in the text, making them all lowercase
    # filtering stopwords
    filtered_words = []  # declare an empty list to hold our filtered words
    for word in words:  # iterate over all words from the text
        if word not in stopword_list and word.isalpha() and len(
                word) > 1:  # only add words that are not in the French stopwords list, are alphabetic, and are more than 1 character
            filtered_words.append(word)  # add word to filter_words list if it meets the above conditions
    filtered_words.sort()  # sort filtered_words list
    return filtered_words

def stem_words(words):
    #stemming words
    stemmed_words = [] #declare an empty list to hold our stemmed words
    stemmer = FrenchStemmer() #create a stemmer object in the FrenchStemmer class
    for word in words:
        stemmed_word=stemmer.stem(word) #stem the word
        stemmed_words.append(stemmed_word) #add it to our stemmed word list
    stemmed_words.sort() #sort the stemmed_words
    return stemmed_words


def clean(doc):
    stop_free = [i for i in doc if i not in stop]
    punc_free = [ch for ch in stop_free if ch not in exclude]
    normalized = stem_words(filter_stopwords(punc_free,get_stopswords()))
    return normalized
    #return punc_free
#print get_tokens(docall)
doc_clean = [clean(get_tokens(docall))]
#print  doc
#print doc_clean
doc_clean2 = " ".join(doc_clean[0])
print doc_clean2
doc_clean3=[]
doc_clean3.append(doc_clean2)
vectorizer = CountVectorizer()
word_frequence = vectorizer.fit_transform(doc_clean3)
words = vectorizer.get_feature_names()
p = 0

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(word_frequence)
weight = tfidf.toarray()

n = 10


import gensim
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)
f2=open('a.txt','w')
for i in ldamodel.print_topics(num_topics=5, num_words=3):
    f2.writelines(str(i).encode('utf-8'))
for w in weight:
    loc = np.argsort(-w)
    for i in range(n):
        f2.write('\n')
        f2.write(u'-{}: {} {}'.format(str(i + 1), words[loc[i]], w[loc[i]]).encode('utf-8'))
f2.close()
#print(ldamodel.print_topics(num_topics=3, num_words=3))
'''
'''
