import re

import pandas
import numpy as np

data = pandas.read_csv('../data/titanic.csv', index_col='PassengerId')

"""
Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров. В качестве ответа приведите
два числа через пробел.

Коррелируют ли число братьев/сестер/супругов с числом родителей/детей? Посчитайте корреляцию
Пирсона между признаками SibSp и Parch.

Какое самое популярное женское имя на корабле? Извлеките из полного имени
пассажира (колонка Name) его личное имя (First Name). Это задание — типичный пример того, с чем сталкивается
специалист по анализу данных. Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен,
а также разделения их на женские и мужские. """


def count_males_females():
    female_count, male_count = data.groupby('Sex').size()
    with open('../output/week1/task1.txt', 'w+') as output_file:
        output_file.write(str(male_count) + ' ' + str(female_count))


def count_surv_percentes():
    nonsurv, survived = data.groupby('Survived').size()
    survived /= (survived + nonsurv)
    with open('../output/week1/task2.txt', 'w+') as output_file:
        output_file.write(str(np.round(survived * 100, 2)))


def count_fclass():
    fclass = data.groupby('Pclass').get_group(1).Pclass.size
    pass_count, _ = data.shape
    fclass = (fclass / pass_count) * 100
    with open('../output/week1/task3.txt', 'w+') as output_file:
        output_file.write(str(np.round(fclass, 2)))


def age_info():
    mean = np.mean(data.Age)
    median = np.nanmedian(data.Age)
    with open('../output/week1/task4.txt', 'w+') as output_file:
        output_file.write(str(np.round(mean, 2)) + ' ' + str(np.round(median, 2)))


def get_corell():
    correl = data[['SibSp', 'Parch']].corr().get_value('Parch', 'SibSp')
    with open('../output/week1/task5.txt', 'w+') as output_file:
        output_file.write(str(np.round(correl, 2)))


def most_popular_female_name():
    female_names = data.groupby('Sex').get_group('female').Name

    result = female_names.map(lambda x: get_fname(x)).mode()[0]

    with open('../output/week1/task6.txt', 'w+') as output_file:
        output_file.write(result)


def get_fname(full_name):
    pattern = re.compile(r'(?<=\()(\b[A-Za-z]+\b)')
    match = pattern.search(full_name)
    if match:
        return match.group(0)
    else:
        return match


count_males_females()
count_surv_percentes()
count_fclass()
age_info()
get_corell()
most_popular_female_name()
