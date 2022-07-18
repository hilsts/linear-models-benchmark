import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler 
import timeit


DATA_PATH = "/Users/hilsts/Documents/linear-models-benchmark/data/"

df = pd.read_csv(DATA_PATH+"song_data.csv")

scaler = StandardScaler()


X = df.drop(["song_popularity", "song_name"], axis=1)
scaled_X = scaler.fit_transform(X)
y = df["song_popularity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
X_train_s, X_test_s, y_train, y_test = train_test_split(scaled_X, y, test_size=0.25)

def linear_regression(X_train, y_train, X_test):
    
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

x = timeit.timeit('linear_regression(X_train, y_train, X_test)', number=10, globals = globals())
print(x)