from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import time
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('clean_combined_data.csv')
data = data[data['star_rating']!=3]
# shuffle
data = data.sample(frac=1).reset_index(drop=True)
########################################################################################################################
# Text to vectors
data_TR = data['headline'].fillna('')
#print(data_TR)

snow = nltk.stem.SnowballStemmer('english')
stop = set(stopwords.words('english'))

data_TR = data_TR.str.split()
data_TR = data_TR.apply(lambda x: [snow.stem(item) for item in x if item not in stop])

#print(data_TR)
def final_text(data, df, name):
    row_lst = []
    for lst in data:
        text = ''
        for word in lst:
            text = text + ' ' + word
        row_lst.append(text)
    df[name] = row_lst

final_text(data_TR, data, 'TR_final_text')


data_TR = data['TR_final_text']
#print(data_TR)

########################################################################################################################
labels = data['star_rating']
a = []
for i in labels:
    if i==4 or i==5:
        a.append(1)
    if i==1 or i==2:
        a.append(0)

labels = pd.Series(a)

from keras.utils import np_utils

labels = np_utils.to_categorical(labels, num_classes=2)
labels = np.argmax(labels, axis=1)
########################################################################################################################
# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(data_TR,labels, test_size=0.20, stratify=labels)#, random_state=18)

# Words to vector by Tfidf words bag method
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(X_train_vec.shape)
print(X_test_vec.shape)

########################################################################################################################
'''parameters = {
    'kernel': ('linear', 'rbf'),
    'C': [1, 2, 4],
    'gamma': [0.125, 0.25, 0.5, 1, 2, 4]}
svr = svm.SVC()
classifier = GridSearchCV(svr, parameters, n_jobs=-1)
'''
classifier = LinearSVC()
time_2 = time.time()
classifier.fit(X_train_vec, y_train)
print('SVM time:', time.time() - time_2)

predicts = classifier.predict(X_test_vec)
print('Predict labels:', predicts)
print('True labels:', np.array(y_test))
print('Accuracy:', accuracy_score(y_test, predicts))
print('f1 score:', f1_score(y_test, predicts, average='macro'))

########################################################################################################################
classifier = MultinomialNB()

time_3 = time.time()
classifier.fit(X_train_vec, y_train)
print('NB time:', time.time() - time_3)

predicts = classifier.predict(X_test_vec)
print('Predict labels:', predicts)
print('True labels:', np.array(y_test))
print('Accuracy:', accuracy_score(y_test, predicts))
print('f1 score:', f1_score(y_test, predicts, average='macro'))
########################################################################################################################

classifier = RandomForestClassifier()

time_4 = time.time()
classifier.fit(X_train_vec, y_train)
print('RF time:', time.time() - time_4)

predicts = classifier.predict(X_test_vec)
print('Predict labels:', predicts)
print('True labels:', np.array(y_test))
print('Accuracy:', accuracy_score(y_test, predicts))
print('f1 score:', f1_score(y_test, predicts, average='macro'))




########################################################################################################################
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3)
X_vec = vectorizer.fit_transform(data_TR)
classifier = LinearSVC()
scores = cross_val_score(classifier, X_vec, labels, cv=10, scoring='f1_macro')
print('CVM')
print(scores)
print("F1: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

########################################################################################################################
classifier = MultinomialNB()
scores = cross_val_score(classifier, X_vec, labels, cv=10, scoring='f1_macro')
print('NB')
print(scores)
print("F1: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))