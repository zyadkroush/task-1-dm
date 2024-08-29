
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = pd.read_csv('D:\summer course\Data Mining\dataset.csv')

print(data.head())

X = data.drop(columns=['data'])
y = data['data']

imputer = SimpleImputer(strategy='mean') 
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_imputed)

normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X_standardized)

processed_data = pd.DataFrame(X_normalized, columns=X.columns)
processed_data['data'] = y.values
processed_data.to_csv('D:\summer course\Data Mining\_data.csv', index=False)

