from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# SVM with kernel method
svm_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='rbf', C=1.0)),  # You can choose different kernels and parameters here
])

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

svm_clf.fit(X_train, y_train)
predicted_svm = svm_clf.predict(X_test)

print('\n')
print('\tSVM Text Classifier')
print('\n')
print(classification_report(y_test, predicted_svm))