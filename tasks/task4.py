import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

trainset = pd.read_csv('../data/perceptron-train.csv', header=None, names=['Class', 'f1', 'f2'])
testset = pd.read_csv('../data/perceptron-test.csv', header=None, names=['Class', 'f1', 'f2'])

def perceptron():
    X = trainset[['f1', 'f2']]
    Y = trainset['Class']
    clf = Perceptron(random_state=241)
    clf.fit(X, Y)
    answers = clf.predict(testset[['f1', 'f2']])
    ans = accuracy_score(testset['Class'], answers)
    with open('../output/week2/task4_1.txt', 'w+') as output_file:
        output_file.write(str(ans))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(testset[['f1', 'f2']])
    clf.fit(X_scaled, Y)
    answers = clf.predict(X_test_scaled)
    ans2 = accuracy_score(testset['Class'], answers)
    with open('../output/week2/task4_2.txt', 'w+') as output_file:
        output_file.write(str(ans2 - ans))


if __name__ == '__main__':
    perceptron()


