import seaborn as sns


data = sns.load_dataset("anscombe")
df_1 = data[data["dataset"] == "I"]
df_2 = data[data["dataset"] =="II"]
df_1.to_csv("Linear_success.csv", index=False)
df_2.to_csv("Linear_failure.csv", index=False)
print("Data saved successfully!")
print("Dataset I:")
print(df_1.head())
print(df_1.describe())
print(df_1.info())
print("\nDataset II:")
print(df_2.head())
print(df_2.describe())
print(df_2.info())
