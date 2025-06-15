import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('打者年俸.csv')

reg_model = LinearRegression()

x = df[['打数','安打','二塁打','三塁打','本塁打','打点','四球','打率']].values
target = df['年俸(推定)'].values

reg_model.fit(x, target)

print('<重み>--------------')
print(reg_model.coef_)
print('<決定係数>----------')
print(reg_model.score(x, target))
print('<予測>--------------')
print('モデル値: ')
y = np.array([500,122,13,1,33,86,105,0.244])
print(y)
print('推定年俸(万円)')
print(reg_model.predict(y.reshape(1, 8)))