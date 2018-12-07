import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from keras.utils import np_utils

data = pd.read_csv('clean_combined_data.csv')
data = data[data['star_rating']!=3]
# shuffle
data = data.sample(frac=1).reset_index(drop=True)
########################################################################################################################
# Text to vectors
data_TR = data['text_reviews'].fillna('')
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

labels = np_utils.to_categorical(labels, num_classes=2)

num=[]
for i in data_TR:
    num.append(len(i))
max_len = max(num)
#print(max_len_HL)

tokenizer = Tokenizer(num_words=20000)

tokenizer.fit_on_texts(data_TR)
TR_sequences = tokenizer.texts_to_sequences(data_TR)
TR_data = pad_sequences(TR_sequences, maxlen=max_len)

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(TR_data, labels, test_size=0.20, stratify=labels)#, random_state=18)


########################################################################################################################
from keras.layers import Embedding, GlobalMaxPooling1D, Dense, Activation, Dropout, Conv1D, Flatten
from keras.models import Sequential
from time import time

########################################################################################################################
model = Sequential()
model.add(Embedding(20000, 128, input_length=max_len))
model.add(Flatten())
model.add(Dense(256, activation='relu'))#, input_shape=(max_len,)))
model.add(Dropout(0.2))
#model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start1 = time()

model.fit(X_train, y_train, epochs=15, batch_size=1000)

et1 = (time() - start1)/60.

print("Model trained!")
print("Elapse time: ", et1, "minutes")

model.summary()

loss, accuracy = model.evaluate(X_test, y_test, batch_size=1000)
print(loss, accuracy)


predicted_label = model.predict_classes(X_test)
print(predicted_label[:10])
y_test = [np.argmax(y, axis=None, out=None) for y in y_test]
print(y_test)


from sklearn.metrics import (precision_score, recall_score, f1_score)
from sklearn.metrics import classification_report

print('Classification report:\n', classification_report(y_test, predicted_label))

print("Precision: %1.3f" % precision_score(y_test, predicted_label, average='macro'))
print("Recall: %1.3f" % recall_score(y_test, predicted_label, average='macro'))
print("F1: %1.3f\n" % f1_score(y_test, predicted_label, average='macro'))