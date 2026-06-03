import seaborn as sns


data = sns.load_dataset("anscombe")
df_1 = data[data["dataset"] == "I"]
df_2 = data[data["dataset"] =="II"]
df_3 = data[data["dataset"] =="III"]
df_4 = data[data["dataset"]== "IV"]
df_1.to_csv("Linear_success.csv", index=False)
df_2.to_csv("Linear_failure.csv", index=False)
df_3.to_csv("The_Outlier_Trick.csv", index=False)
df_4.to_csv("The-Vertical_Trap.csv", index=False)
print("Data saved successfully!")
print("Dataset I:")
print(df_1.head())
print(df_1.describe())
print(df_1.info())
print("\nDataset II:")
print(df_2.head())
print(df_2.describe())
print(df_2.info())
print("\nDataset III:")
print(df_3.head())
print(df_3.describe())
print(df_3.info())
print("\nDataset IV:")
print(df_4.head())
print(df_4.describe())
print(df_4.info())

