from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd 

df_1 = pd.read_csv('Data/Linear_success.csv')
df_2 = pd.read_csv('Data/Linear_failure.csv')

def train_model(df, name):
    X = df[['x']]
    y = df['y']
    model = LinearRegression()
    model.fit(X,y)
    prediction = model.predict(X)
    mse = mean_squared_error(y,prediction)
    r2 = r2_score(y,prediction)
    print(f"Results for {name}")
    print(f"Slope (m): {model.coef_[0]:.3f}")
    print(f"Intercept (c): {model.intercept_:.3f}")
    print(f"R2 Score: {r2:.3f}")
    print(f"MSE: {mse:.3f}\n")
    return model, mse, r2

train_model(df_1,"Dataset I (success)")
train_model(df_2,"Dataset II (failure)")