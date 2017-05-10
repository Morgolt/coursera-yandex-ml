import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data = pandas.read_csv('../data/titanic.csv', index_col='PassengerId')


def get_important_features():
    trainset = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']].dropna(axis=0, how='any')
    trainset['Sex'] = trainset['Sex'].map(lambda x: convert_sex(x))
    target = trainset['Survived']
    trainset = trainset[['Pclass', 'Fare', 'Age', 'Sex']]
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(trainset, target)
    importances = clf.feature_importances_
    resind = np.argsort(-importances)[:2]
    result = list(trainset.columns[resind])

    with open('../output/week1/task7.txt', 'w+') as output_file:
        output_file.write(' '.join(result))


def convert_sex(sex):
    if sex == 'male':
        return 1
    else:
        return 0


if __name__ == '__main__':
    get_important_features()
