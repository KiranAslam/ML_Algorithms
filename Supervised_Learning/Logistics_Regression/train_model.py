import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression


df = pd.read_pickle("./Data/cs-training-transformed.pkl")

features =  [
    'age_WoE', 'MonthlyIncome_WoE', 'DebtRatio_WoE', 'RevolvingUtilizationOfUnsecuredLines_WoE',
    'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse'
]
X = df[features]
Y= df["SeriousDlqin2yrs"]
x_train, x_test, y_train, y_test= train_test_split(
    X,Y,test_size=0.2,random_state=42,stratify=Y)

smote = SMOTE(random_state=42)
x_train_res,y_train_res = smote.fit_resample(x_train,y_train)

model = LogisticRegression(max_iter=1000, penalty='l2', random_state=42)
model.fit(x_train_res,y_train_res)

prediction = model.predict(x_test)
prob = model.predict_proba(x_test)[:, 1]

print("Classification report")
print(classification_report(y_test,prediction))
print("-"*50)
print(f"ROC-AUC Performance Score: {roc_auc_score(y_test, prob):.4f}")

print("\n[Confusion Matrix]:")
cm = confusion_matrix(y_test, prediction)
print(f"True Negatives (Safe predicted Safe): {cm[0][0]}")
print(f"False Positives (Safe predicted Default): {cm[0][1]}")
print(f"False Negatives (Default predicted Safe): {cm[1][0]} ")
print(f"True Positives (Default predicted Default): {cm[1][1]}")
print("="*68)
