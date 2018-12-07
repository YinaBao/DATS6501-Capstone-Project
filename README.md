# DATS6501-Capstone-Project
##  Sentiment Classification of Amazon Reviews
### Author: Yina Bao
In recent years, electronic commerce more and more dominates the market, and there has been a huge increase of interest from brands, companies and data scientists in sentiment analysis and the application to find the business intelligence insight. Nowadays, widely available text data from social media and product reviews can help your business in different sides, such as brand monitoring, customer service, product quality, market research and strategy. Recent study from Zendesk mentioned that 45% of people share negative customer service experience and 30% share positive customer service experience via social media, which shows a high demand for mining the information and extracting the opinion and meaning for further analysis. Sentiment analysis has become one of the trendiest topics of scientific and market research in the field of Natural Language Processing and Machine Learning. During this project, I will explore both deep learning algorithms and traditional machine learning algorithms to classify and make the sentiment analysis of the customer reviews from Amazon based on the text descriptive reviews. Those machine learning methods of classification and sentiment analysis not only helped to match the rating and reviews, but also to detect opinions in different fields, such social media platform, news reports and etc. 

In this project, I will classify the reviews’ sentiment based on the text input. I used multi-layer perceptron, convolutional neural network, support vector machine, random forest and naive Bayes methods to find out how these algorithms compare to each other and how can we improve the project.


### Dataset
Amazon product reviews dataset consists of star ratings (1-5), the headline and the descriptive customer reviews.
The original dataset is pre-split into a testing set and training set. Training set contains 3,000,000 reviews and testing set contains 650,000 reviews.
Link to Kaggle data page: https://www.kaggle.com/bittlingmayer/amazonreviews
Link to dataset: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M


### Running the code
Environment Sepecification: 
* Python 2.7 or above
* Set DISPLAY=localhost:10.0

Before running the code, please make sure you have the following packages installed: keras, nltk, sklearn, pandas, numpy, matplotlib, seaborn, wget, Wordcloud.

* Run either get_data.py or get_data.ipynb to download the data for runing the following code.
* Plot.ipynb -- Draw the word cloud plots and count data plot
* Headline.py -- Classification by headline (SVM, Naive Bayes, Random Forest)
* Text_reviews.py -- Classification by reviews (SVM, Naive Bayes, Random Forest)
* MLP.py -- Sentiment classification by MLP model
* CNN.py -- Sentiment classification by CNN model
* SVM_2.py -- Sentiment classification by SVM model
* NB.py -- Sentiment classification by Naive Bayes model
* RF.py -- Sentiment classification by Random Forest model
* SVM_5.py -- 5 star classification by SVM model
* SVM_3.py -- Sentiment classification by SVM model include 3 star


If you want to regenerate the sample data, run following code by order (Runing those codes takes really long time!!!)
Please download the original dataset from the Google Drive link above.
* training_data_clean.ipynb
* testing_data_clean.ipynb
* combine_data.ipynb




Please read the report for detailed analysis.
