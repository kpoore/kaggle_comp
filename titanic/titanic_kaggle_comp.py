import pandas as pd
import numpy as np
import pylab as plb

df = pd.read_csv('train.csv', header=0)

df['Age'].hist()
plb.show()

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df['Registered'] = df['Embarked'].map({'C': 0, 'S': 1, 'Q': 2} )

median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) &\
                                (df['Pclass'] == j+1)]['Age'].dropna().median()
print (median_ages)
df['AgeFill'] = df['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
