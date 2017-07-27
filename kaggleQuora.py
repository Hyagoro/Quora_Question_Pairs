import gensim
import csv
from pyemd import emd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from itertools import islice
from sklearn.decomposition import PCA
import time

# merge with toto.py to continue the training
# model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)


# model = gensim.models.Word2Vec \
#     .load_word2vec_format('/home/steve/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)
#
# toto = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# print(toto)
#
#
# def myfunc1(w2v_model, input_word="apple"):
#     print(w2v_model.similar_by_word(input_word))
#
#
# for word in ["apple", "man"]:
#     myfunc1(model, word)

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap

def wmd(v_1, v_2, D_):
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    D_ = D_.astype(np.double)
    D_ /= D_.max()  # just for comparison purposes
    print("d(doc_1, doc_2) = {:.2f}".format(emd(v_1, v_2, D_)))

X_text = []
y = []
documents_couple = []

print("Read CSV file")

with open('/home/steve/Documents/KaggleQuora/train.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

    for row in islice(spamreader, 1, None):
        documents_couple.append([row[3], row[4]])
        X_text.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[3]), row[5]))
        X_text.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[4]), row[5]))
        y.append(row[5])

model = gensim.models.doc2vec.Doc2Vec(size=100, window=5, min_count=5, workers=4)
try:
    print("Load W2V")
    model = gensim.models.doc2vec.Doc2Vec.load("/home/steve/Documents/KaggleQuora/w2v")
except Exception as e:
    print(e)
    print("Train W2V")
    model.build_vocab(X_text)
    model.train(X_text)
    model.save("/home/steve/Documents/KaggleQuora/w2v")

print("Infer W2V vectors")

X = []
for document in documents_couple:
    l1 = model.infer_vector(document[0])
    l2 = model.infer_vector(document[1])
    tmp = []
    tmp.extend(l1)
    tmp.extend(l2)
    X.append(tmp)

print("Normalize data")

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

print("PCA")

pca = PCA(n_components=200)
X = pca.fit_transform(X)

print("Split dataset")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print("Train MLPClassifier")

classifier = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(100,), random_state=1)


@timing
def train_classifier(classifier_, X_train_, y_train_):
    classifier_.fit(X_train_, y_train_)

train_classifier(classifier, X_train, y_train)
# classifier.fit(X_train, y_train)

print("Predict")

y_pred = classifier.predict(X_test)

print("Scores")

print(classification_report(y_test, y_pred))

