import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("./Data/cs-training.csv")
#print(df.head(10))
print(df.describe())
#print(df.info())
#print(df.isnull().sum())
df.drop(columns=['Unnamed: 0'], inplace=True)
print(df.info())
inc_median= df["MonthlyIncome"].median()
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(inc_median)
num_dep_median = df["NumberOfDependents"].median()
df["NumberOfDependents"]=df["NumberOfDependents"].fillna(num_dep_median)
print(df.isnull().sum())
late_columns = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse"
]
for col in late_columns:
    col_mode = df[col].mode()[0] 
    df[col] = np.where(df[col] > 90, col_mode, df[col]) 
for col in late_columns:
    col_mode = df[col].mode()[0] 
    df[col] = np.where(df[col] > 90, col_mode, df[col]) 

df['RevolvingUtilizationOfUnsecuredLines'] = np.where(
    df['RevolvingUtilizationOfUnsecuredLines'] > 1.0, 
    1.0, 
    df['RevolvingUtilizationOfUnsecuredLines']
)

print(df[late_columns].max())

def calculate_woe_iv(data, feature, target, method='cut'):
    df_temp = data[[feature, target]].copy()
    
    if method == 'qcut':
        df_temp['bin'] = pd.qcut(df_temp[feature], q=5, duplicates='drop', labels=False)
    else:
        df_temp['bin'] = pd.cut(df_temp[feature], bins=5, labels=False)
    
    grouped = df_temp.groupby('bin')[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped['total'] - grouped['bad']
    grouped["good"] = grouped['good'].replace(0, 0.5)
    grouped["bad"] = grouped["bad"].replace(0, 0.5)
    
    total_goods = df_temp[target].value_counts()[0]
    total_bads = df_temp[target].value_counts()[1]
    
    grouped['Dist_Good'] = grouped['good'] / total_goods
    grouped['Dist_bad'] = grouped['bad'] / total_bads
    grouped['WoE'] = np.log(grouped['Dist_Good'] / grouped['Dist_bad'])
    grouped['IV'] = (grouped['Dist_Good'] - grouped['Dist_bad']) * grouped['WoE']
    
    return grouped['WoE'].to_dict(), grouped['IV'].sum()

target_col = 'SeriousDlqin2yrs'

features_config = {
    'age': 'cut',
    'RevolvingUtilizationOfUnsecuredLines': 'qcut',
    'MonthlyIncome': 'qcut', 
    'DebtRatio': 'qcut'
}
print("--- FINAL OPTIMIZED WoE & IV REPORT ---")
for col, method in features_config.items():
    woe_mapping, feature_iv = calculate_woe_iv(df, col, target_col, method=method)
    
    if method == 'qcut':
        temp_bins = pd.qcut(df[col], q=5, duplicates='drop', labels=False)
    else:
        temp_bins = pd.cut(df[col], bins=5, labels=False)
        
    df[f'{col}_WoE'] = temp_bins.map(woe_mapping)
    print(f"Feature: {col:35} | Method: {method:5} | IV Score: {feature_iv:.4f}")

df.to_pickle("./Data/cs-training-transformed.pkl")