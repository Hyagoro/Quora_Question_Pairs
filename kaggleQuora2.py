import gensim
import csv
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from itertools import islice
import time
import math
from pyemd import emd


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


# merge with toto.py to continue the training
# model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

model = gensim.models.Word2Vec \
    .load_word2vec_format('/home/steve/Downloads/GoogleNews-vectors-negative300.bin.gz', binary=True)

X_text = []
y = []
documents_couple = []

print("Read CSV file")

with open('/home/steve/Documents/KaggleQuora/train.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')

    for row in islice(spamreader, 1, None):
        documents_couple.append([row[3], row[4]])
        y.append(row[5])


print("Infer W2V vectors")

X = []
for document in documents_couple:
    question1 = gensim.utils.simple_preprocess(document[0])
    question2 = gensim.utils.simple_preprocess(document[1])

    lenQ1 = len(question1)
    lenQ2 = len(question2)

    list_max = []
    wmd = model.wmdistance(question1, question2)
    if math.isnan(wmd) | math.isinf(wmd):
        wmd = 0

    matrix = [[0 for x in range(lenQ2)] for y in range(lenQ1)]

    for i in range(lenQ1):
        for j in range(lenQ2):
            try:
                result = model.wv.similarity(question1[i], question2[j])
                matrix[i][j] = result
            except Exception as e:
                matrix[i][j] = 0

    for i in range(lenQ1):
        # if len(matrix[i]) > 0:
        try:
            t = max(matrix[i])
            list_max.append(t)
        except:
            print("empty line on the matrix")

    sum_list_max = sum(list_max)
    if len(list_max) != 0:
        average = sum_list_max / len(list_max)
    else:
        average = 0

    X.append([sum_list_max, average, wmd])


# print("Normalize data")
#
# min_max_scaler = preprocessing.MinMaxScaler()
# X = min_max_scaler.fit_transform(X)

# print("PCA")
#
# pca = PCA(n_components=200)
# X = pca.fit_transform(X)
#
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

