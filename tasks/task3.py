import pandas as pd
import sklearn
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale

data = pd.read_csv('../data/wine.csv')
print(data)
klass = data['Class']
data = data.drop(['Class'], axis=1)


# print(data)
def knn_unnorm():
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    acc = np.empty((1, 50))
    for i in range(1, 51):
        estimator = KNeighborsClassifier(n_neighbors=i)
        acc[0][i - 1] = np.mean(cross_val_score(estimator, data,
                                                y=klass,
                                                cv=kfold,
                                                n_jobs=-1,
                                                scoring='accuracy'))
    ans1 = np.argmax(acc) + 1
    ans2 = np.max(acc)
    with open('../output/week2/task1.txt', 'w+') as output_file:
        output_file.write(str(ans1))
    with open('../output/week2/task2.txt', 'w+') as output_file:
        output_file.write(str(np.round(ans2, 2)))


def knn_norm():
    norm_data = scale(data)
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    acc = np.empty((1, 50))
    for i in range(1, 51):
        estimator = KNeighborsClassifier(n_neighbors=i)
        acc[0][i - 1] = np.mean(cross_val_score(estimator, norm_data,
                                                y=klass,
                                                cv=kfold,
                                                n_jobs=-1,
                                                scoring='accuracy'))
    ans1 = np.argmax(acc) + 1
    ans2 = np.max(acc)
    with open('../output/week2/task3.txt', 'w+') as output_file:
        output_file.write(str(ans1))
    with open('../output/week2/task4.txt', 'w+') as output_file:
        output_file.write(str(np.round(ans2, 2)))


def choose_metric():
    from sklearn.datasets import load_boston
    data = load_boston()
    klass = data['target']
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    norm_features = scale(data['data'])
    res = np.array([])
    prang = np.linspace(1, 10, 200)
    for p in prang:
        cls = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
        res = np.append(res, np.mean(cross_val_score(cls, norm_features,
                                               y=klass,
                                               cv=kfold,
                                               n_jobs=-1,
                                               scoring='neg_mean_squared_error')))

    ans = prang[np.argmax(res)]
    print(res)
    with open('../output/week2/task4.txt', 'w+') as output_file:
        output_file.write(str(ans))


if __name__ == '__main__':
    choose_metric()
