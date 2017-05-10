import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(newsgroups.data)

cls = SVC(kernel='linear', random_state=241)
grid = {'C': np.power(10.0, range(-5, 6))}
kfold = KFold(n_splits=5, shuffle=True, random_state=241)
gridCV = GridSearchCV(cls, grid, scoring='accuracy', cv=kfold, n_jobs=-1)
gridCV.fit(X, newsgroups.target)
best_cls = gridCV.best_estimator_
results = []
