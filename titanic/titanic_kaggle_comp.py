import pandas as pd
import numpy as np
import pylab as plb
from sklearn.ensemble import RandomForestClassifier as rfc

df = pd.read_csv('train.csv', header=0)
tdf = pd.read_csv('test.csv', header=0)
# df['Age'].hist()
# plb.show()

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df['Registered'] = df['Embarked'].map({'C': 0, 'S': 1, 'Q': 2} )
tdf['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
tdf['Registered'] = df['Embarked'].map({'C': 0, 'S': 1, 'Q': 2} )

median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) &\
                                (df['Pclass'] == j+1)]['Age'].dropna().median()
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = tdf[(tdf['Gender'] == i) &\
                                (tdf['Pclass'] == j+1)]['Age'].dropna().median()
print (median_ages)
df['AgeFill'] = df['Age']
tdf['AgeFill'] = tdf['Age']

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

for i in range(0, 2):
    for j in range(0, 3):
        tdf.loc[ (tdf.Age.isnull()) & (tdf.Gender == i) & (tdf.Pclass == j+1),\
                'AgeFill'] = median_ages[i,j]

tdf['AgeIsNull'] = pd.isnull(tdf.Age).astype(int)


# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch']
df['age_class_mult'] = df.AgeFill * df.Pclass
tdf['FamilySize'] = df['SibSp'] + df['Parch']
tdf['age_class_mult'] = df.AgeFill * df.Pclass

# df.FamilySize.hist()
# plb.show()
#
# df.age_class_mult.hist()
# plb.show()
#
# df.AgeFill.hist()
# plb.show()

# Prepare data fro machine learning, no 'object' types
df = df.drop(['Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked',
                'PassengerId'], axis=1)
tdf = tdf.drop(['Name', 'Age', 'Sex', 'Ticket', 'Cabin', 'Embarked',
                'PassengerId'], axis=1)
df = df.dropna()
tdf = tdf.dropna()

train_data = df.values
test_data = tdf.values

# Create random forest object
forest = rfc(n_estimators = 100)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])
output = forest.predict(test_data)
