import pandas as pd
import numpy as np

df = pd.read_csv("./Iris.csv")
#print(df.head(10))
df = df[df["Species"] != "Iris-setosa"]
#print(df.isnull().sum())
df["Species"] = df["Species"].map({'Iris-versicolor': 0, 'Iris-virginica': 1})
from sklearn.model_selection import train_test_split
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score

classifier = LogisticRegression(solver='saga', random_state=42)
from sklearn.model_selection import GridSearchCV
parameters = {
    'penalty': ['l1', 'l2'],
    'C': [1, 2, 3, 4, 5, 6, 20, 30, 40, 50],
    'max_iter': [1000, 2000, 3000, 4000, 5000]
}
classifier_regressor = GridSearchCV(classifier,param_grid=parameters,scoring='accuracy',cv=5)

classifier_regressor.fit(X_train,y_train)

print(f"best param: {classifier_regressor.best_params_}")
print(f"best score: {classifier_regressor.best_score_}")
y_pred = classifier_regressor.predict(X_test)
score = accuracy_score(y_pred,y_test)
print(score)
print(classification_report(y_pred,y_test))

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df,hue='Species')
plt.show()
sns.heatmap(df.corr(),annot=True)
plt.show()
