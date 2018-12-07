from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('clean_combined_data.csv')
# shuffle
data = data.sample(frac=1).reset_index(drop=True)
########################################################################################################################
# Text to vectors
data_TR = data['text_reviews'].fillna('')
print(data_TR)

snow = nltk.stem.SnowballStemmer('english')
stop = set(stopwords.words('english'))

data_TR = data_TR.str.split()
data_TR = data_TR.apply(lambda x: [snow.stem(item) for item in x if item not in stop])

print(data_TR)
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
print(data_TR)

########################################################################################################################
labels = data['star_rating']

########################################################################################################################
# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(data_TR,labels, test_size=0.30)#, random_state=18)

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
print('Time:', time.time() - time_2)

predicts = classifier.predict(X_test_vec)
print('Predict labels:', predicts)
print('True labels:', np.array(y_test))
print('Accuracy:', accuracy_score(y_test, predicts))
print('f1 score:', f1_score(y_test, predicts, average='macro'))

########################################################################################################################
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3)
X_vec = vectorizer.fit_transform(data_TR)

scores = cross_val_score(classifier, X_vec, labels, cv=10, scoring='f1_macro')
print(scores)
print("F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

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