# Importing necessary packages
import pandas as pd  # python's data handling package
import numpy as np  # python's scientific computing package
import matplotlib.pyplot as plt  # python's plotting package
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_error, r2_score

'''
loading
'''

# Both features and target have already been scaled: mean = 0; SD = 1
reader = pd.read_excel('Iowa_Housing_Data_Mod.xlsx')

# creating the following variables and drop Id and others

columns = ['Id', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'BsmtUnfSF',
           'TotalBsmtSF', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
           'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
           'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',
           'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
           'ScreenPorch', 'PoolArea', 'YrSold', 'SalePrice']
data = pd.DataFrame()

for i in columns:
    data[i] = reader[i]
data['Age of House'] = data['YrSold'] - data['YearBuilt']
data['CentralAC Dummy'] = data['CentralAir'].map({'Y': 1, 'N': 0})

data = data.dropna()
data = data.drop(['YrSold', 'YearBuilt', 'CentralAir'], axis=1)

features = data.drop('SalePrice', axis=1)

'''
Linear regression
'''

# Importing models
from sklearn.linear_model import LinearRegression
from scipy import stats

y = data[['SalePrice']]
x = features[['GrLivArea']]
lr = LinearRegression()
lr.fit(x, y)
pred = lr.predict(x)
r2 = r2_score(y, pred)
print(f'Part A r2: {r2}')

beta = lr.coef_[0][0]
print(f'Part A beta: {beta}')
intercept = lr.intercept_[0]

params = np.append(lr.intercept_, lr.coef_)

# Compute the residual sum of squares and the mean squared error
residuals = y - pred
RSS = np.sum(residuals ** 2)
MSE = RSS / len(x)

# Compute the standard errors for the regression coefficients
X_train_ext = np.column_stack((np.ones(len(x)), x))
cov_matrix = MSE * np.linalg.inv(X_train_ext.T @ X_train_ext)

std_err = np.sqrt(np.diag(cov_matrix))

# Compute the t-statistics
t_stats = params / std_err

# Compute the p-values using the t-distribution
p_values = [2 * (1 - stats.t.cdf(np.abs(t_stat), len(x) - 1 - 1)) for t_stat in t_stats]

'''
Multiple Linear Regression
'''

features = ['LotArea', 'OverallQual', 'OverallCond', 'Age of House', 'CentralAC Dummy', 'GrLivArea', 'GarageCars']
y = data[['SalePrice']]
x = data[features]
lr2 = LinearRegression()
lr2.fit(x, y)
params = np.append(lr2.intercept_, lr2.coef_)
pred = lr2.predict(x)
r2 = r2_score(y, pred)

newX = pd.DataFrame({"Constant": np.ones(len(x))}).join(pd.DataFrame(x))

# Compute the residual sum of squares and the mean squared error
residuals = y - pred
RSS = np.sum(residuals ** 2)
MSE = RSS / len(x)

var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params / sd_b

p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]

sd_b = np.round(sd_b, 3)
ts_b = np.round(ts_b, 3)
p_values = np.round(p_values, 3)
params = np.round(params, 4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["t values"], myDF3["Probabilities"] = [params, sd_b, ts_b,
                                                                                              p_values]
print(myDF3)

# Lasso
from sklearn.linear_model import Lasso

# First 1800 data items are training set; the next 600 are the validation set: the final 508 are the etst set
train = data.iloc[:1800]
val = data.iloc[1800:2400]
test = data.iloc[2400:2908]

X_train, X_val, X_test = train.drop('SalePrice', axis=1), val.drop('SalePrice', axis=1), test.drop('SalePrice', axis=1)
y_train, y_val, y_test = train[['SalePrice']], val[['SalePrice']], test[['SalePrice']]

# Here we produce results for alpha=0.05 which corresponds to lambda=0.1 in Hull's book
lasso = Lasso(alpha=0.1, max_iter=10000000).fit(X_train, y_train)
lasso

# DataFrame with corresponding feature and its respective coefficients
coeffs = pd.DataFrame(
    [
        ['intercept'] + list(X_train.columns),
        list(lasso.intercept_) + list(lasso.coef_)
    ]
).transpose().set_index(0)
coeffs

# We now consider different lambda values. The alphas are half the lambdas
alphas = [0.01 / 2, 0.02 / 2, 0.03 / 2, 0.04 / 2, 0.05 / 2, 0.075 / 2, 0.1 / 2]
mses = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000000)
    lasso.fit(X_train, y_train)
    pred = lasso.predict(X_val)
    mses.append(mse(y_val, pred))
    print(mse(y_val, pred))

plt.plot(alphas, mses)
plt.show()

from sklearn.linear_model import Ridge

alphas = [0.01 * 1800, 0.02 * 1800, 0.03 * 1800, 0.04 * 1800, 0.05 * 1800, 0.075 * 1800, 0.1 * 1800, 0.2 * 1800,
          0.6 * 1800, 1.0 * 1800]
mses = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    pred = ridge.predict(X_val)
    mses.append(mse(y_val, pred))
    print(mse(y_val, pred))
