from gensim import corpora, models, similarities
from collections import defaultdict
from pprinter import pprint
from six import iteritems

# PART 1: Create a corpus of documents represented as a stream of vectors
class MyCorpus(object):
     def __iter__(self):
         for line in open('mycorpus.txt'):
             yield dictionary.doc2bow(line.lower().split())

#list of string
#documents = []
documents = ["human machine interface for lab abc computer",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph human minors A human"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
# texts is a list of lists. Each list contains single words for that document.
# translation: each slack message will be a document. Thus, texts will be a list of lists=n messages, where each list has individual words from each message

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]




# create dictionary with texts
dictionary = corpora.Dictionary(texts)
# create corpus, which is an object containing sparse vectors.
# Each sparse vector represents a word (identified with a number=id, and its frequency in the message)
#corpus = [dictionary.doc2bow(text) for text in texts]

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))

stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed

#print dictionary
#for vector in corpus_memory_friendly:
#    print vector

# PART 2: Apply a transformation to the corpus to produce topics
# step 1 -- initialize a model
tfidf = models.TfidfModel(corpus_memory_friendly)

# step 2 -- use the model to transform vectors
corpus_tfidf = tfidf[corpus_memory_friendly]
#for doc in corpus_tfidf:
#    print doc

model = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=100)
corpus_lda = model[corpus_tfidf]
for doc in corpus_lda:
    print doc
