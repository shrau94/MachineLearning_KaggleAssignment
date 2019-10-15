# importing all the required librarires
import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
import matplotlib.pyplot as plt
import category_encoders as ce

# Reading data
dataset = pd.read_csv('/Users/shravani/Documents/Machine Learning/data.csv')
dataset1 = pd.read_csv('/Users/shravani/Documents/Machine Learning/data.csv')

# Removing null data
dataset = dataset.fillna(method='bfill')

# Removing outliers
ds = pd.DataFrame(dataset)
dataset = dataset[ds['Income in EUR'] < 5000000 ]

# Encoding all the categorical columns - Target encoding
df = pd.DataFrame(dataset)
genencoder = ce.TargetEncoder(cols=['Gender'])
dfgender = genencoder.fit_transform(df['Gender'], df['Income in EUR'])
profencoder = ce.TargetEncoder(cols=['Profession'])
dfprof = profencoder.fit_transform(df['Profession'], df['Income in EUR'])
countryencoder = ce.TargetEncoder(cols=['Country'])
dfcountry = countryencoder.fit_transform(df['Country'], df['Income in EUR'])
degreeencoder = ce.TargetEncoder(cols=['University Degree'])
dfdegree = degreeencoder.fit_transform(df['University Degree'], df['Income in EUR'])
df = df.drop(columns=['Gender','University Degree', 'Country','Profession'])
df = pd.concat([df, dfprof, dfgender, dfcountry, dfdegree], axis=1)
dataset = df

# Training the model
X = dataset[['Gender','University Degree', 'Age', 'Size of City', 'Country', 'Profession', 'Year of Record']]
y = dataset['Income in EUR']

# Scaling the data
df = pd.DataFrame(X)
sc = StandardScaler()
scaler = sc.fit_transform(df[['Size of City','Year of Record','Age']])
scaler = np.transpose(scaler)
df['Size of City'] = scaler[0]
df['Year of Record'] = scaler[1]
df['Age'] = scaler[2]
X = df

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print('Training data size: (%i,%i)' % X_train.shape)
print('Testing data size: (%i,%i)' % X_test.shape)


# Training the model
import xgboost as xgb
regression_model=xgb.XGBRegressor(objective='reg:squarederror', random_state=1, learning_rate=0.1, max_depth=5, n_estimators= 1000)

regression_model.fit(X_train,y_train)


# Testing the model
y_pred = regression_model.predict(X_test)
print("predicted data size:", y_pred.size)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Test RMSE: %f' % test_rmse)

# Plot for the test
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Predicting based on model
dataset = pd.read_csv('/Users/shravani/Documents/Machine Learning/tcd ml 2019-20 income prediction test (without labels).csv')
dataset1 = pd.read_csv('/Users/shravani/Documents/Machine Learning/tcd ml 2019-20 income prediction submission file.csv')

# Filling null values
dataset = dataset.fillna(method='bfill')

# Encoding all the categorical columns - Target encoding
df = pd.DataFrame(dataset)
dfprof = profencoder.transform(df['Profession'])
dfgender = genencoder.transform(df['Gender'])
dfcountry = countryencoder.transform(df['Country'])
dfdegree = degreeencoder.transform(df['University Degree'])
df = df.drop(columns=['Gender','University Degree', 'Country','Profession'])
df = pd.concat([df, dfprof, dfgender, dfcountry, dfdegree], axis=1)
dataset = df

X = dataset[['Gender','University Degree', 'Age', 'Size of City', 'Country', 'Profession', 'Year of Record']]

# Scaling the numerical data
df = pd.DataFrame(X)
scaler = sc.transform(df[['Size of City','Year of Record','Age']])
scaler=np.transpose(scaler)
df['Size of City'] = scaler[0]
df['Year of Record'] = scaler[1]
df['Age'] = scaler[2]
X = df

# Predicting the income
y_pred_for_given_data = regression_model.predict(X)
print(y_pred_for_given_data.size)

# Saving the values in a submission file
dataset1['Income'] = y_pred_for_given_data
dataset1.to_csv('/Users/shravani/Documents/Machine Learning/submission.csv', index=False)
