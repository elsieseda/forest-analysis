
import pandas
import sklearn
import matplotlib

df = pandas.read_csv('forestfires.csv')
print(df)
print(df.shape)
print(df.isnull().sum())
subset = df[['month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']]
subset['month'].replace({'jan':1, 'feb':2, 'mar':3, "apr":4, 'may':5, 'jun':6, "jul":7, "aug":8,
                     "sep":9, "oct":10, "nov":11, "dec":12}, inplace=True)
subset['day'].replace({'mon':1, 'tue':2, 'wed':3, "thu":4, 'fri':5, 'sat':6, "sun":7}, inplace=True)


array = subset.values
X = array[:, 0:10] # : means all rows from column 1..3
y = array[:, 10]  # 10th counted here

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=42)

# linear regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

model = LinearRegression()
model.fit(X_train, Y_train)
print('learning completed')

# ask the model to predict X_test
prediction = model.predict(X_test)
print(prediction)

# check accuracy/performance
from sklearn.metrics import r2_score
# r squared shows the percentage
print('R squared:', r2_score(Y_test, prediction))

from sklearn.metrics import mean_squared_error
print('mean square error:', mean_squared_error(Y_test, prediction))
# above its squared, so we find square root

new = [[4,7,86.2,26.2,94.3,5.1,8.2,51,6.7,0]]
observation = model.predict(new)
print('the  forest area affected by fire:', observation)

# plot linear regression
import matplotlib.pyplot as plt
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.scatter(Y_test, prediction)
ax.plot(Y_test, Y_test)
ax.set_title("predction vs Y_test")
ax.set_xlabel('Y test')
ax.set_ylabel('prediction')
plt.show()