import pandas
import sklearn

df = pandas.read_csv('forestfires.csv')
print(df)
print(df.shape)
print(df.isnull().sum())

# fill missing values using fillna, replace, dropna
subset = df[['month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'area']]

subset['month'].replace({'jan':1, 'feb':2, 'mar':3, "apr":4, 'may':5, 'jun':6, "jul":7, "aug":8,
                     "sep":9, "oct":10, "nov":11, "dec":12}, inplace=True)
subset['day'].replace({'mon':1, 'tue':2, 'wed':3, "thu":4, 'fri':5, 'sat':6, "sun":7}, inplace=True)

# split the data into training and test data at ratio of 70:30

array = subset.values
X = array[:, 0:10] # : means all rows from column 0..13
y = array[:, 10]  # 13 counted here

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
object = KNeighborsClassifier()
print('model training on data.... please wait')
object.fit(X_train,Y_train) # model is now fitted with data for training
print('learning completed')

# ask model to predict the y for  X_test
predictions = object.predict(X_test)
print(predictions)

# compare predictions with the Y_test
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))

from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test,predictions))

# model failed on yes.
# needs improvement in yes outcome

newperson = [[12,5,92.5,88,698.6,7.1,22.8,40,4,0]]
observation = object.predict(newperson)
print('predicted:', observation)





