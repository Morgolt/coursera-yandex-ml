import pandas as pd
from sklearn.svm import SVC

data = pd.read_csv('../../data/svm-data.csv')
klass = data['Class']
data = data.drop(['Class'], axis=1)

cls = SVC(C=100000, random_state=241, kernel='linear')
cls.fit(data, klass)
print(cls.support_)

with open('../../output/week3/task1.txt', 'w+') as output_file:
    output_file.write(' '.join(map(lambda x: str(x+1), cls.support_)))

