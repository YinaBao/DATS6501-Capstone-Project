from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
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

conf_matrix = confusion_matrix(y_test, predicts)

df_cm = pd.DataFrame(conf_matrix)
plt.figure(figsize=(5,5))
print(conf_matrix)
hm = sns.heatmap(df_cm, cbar=False,annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns+1, xticklabels=df_cm.columns+1)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()



y_pred_proba = classifier.decision_function(X_test_vec)
fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


########################################################################################################################
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3)
X_vec = vectorizer.fit_transform(data_TR)
scores = cross_val_score(classifier, X_vec, labels, cv=10, scoring='f1_macro')
print('CVM')
print(scores)
print("F1: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
