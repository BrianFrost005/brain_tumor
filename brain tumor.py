import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

#load data
df = pd.read_csv("dataset/Brain Tumor.csv")
#data analysis
df.describe()
#drop Coarseness as it has almost zero average and no effect on the dataset.
df.isnull().sum()
df = df.drop(columns=['Coarseness'])


#correlation matrix and heatmap using the seaborn library
int_cols = df.select_dtypes(include=['int64','float64']).columns
int_df = df[int_cols]
corr_matrix = int_df.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


#split data into input (X) and output (y) variables.
X = df.drop(columns=["Class","Image"])
y = df["Class"]
#split the data into training and testing sets using the train_test_split() method.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=24)
#define the number of folds for cross-validation and create a k-fold cross-validation object
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=64)


#logistic regression model using scikit-learn
#Create pipeline with StandardScaler and logistic regression model
#define hyperparameters to search over
#create a grid search object with k-fold cross-validation
#fit the grid search object to the training data 
#create a new model with the best hyperparameters
#fit the new model to training data and make predictions on the test data
#evaluate the model using classification report
#plot the confusion matrix using seaborn.
lr_pipeline = make_pipeline(StandardScaler(),LogisticRegression(random_state=160,max_iter=1000))
lr_params = {'logisticregression__C': [1, 10, 100],
             
             }

lr_grid = GridSearchCV(lr_pipeline, lr_params, cv=kf, scoring='accuracy')
lr_grid.fit(X_train, y_train)
best_params = lr_grid.best_params_

lr_model = make_pipeline(StandardScaler(),LogisticRegression(multi_class='ovr',solver='lbfgs',C=best_params['logisticregression__C'], random_state=160,max_iter=1000))
lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred))

sns.set_theme()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()


#random forest classifier model
#create a pipeline with a StandardScaler and a random forest classifier model
#define hyperparameters to search over
#create a grid search object with k-fold cross-validation
#Fit the grid search object to the training data and get the best model from the grid search
#use the best model to make predictions on the test data
#evaluate model using classification report
#plot theconfusion matrix using seaborn
rf_pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=160))
rf_params = {
             'randomforestclassifier__criterion' :['gini', 'entropy'],
             'randomforestclassifier__max_features': ['sqrt', 'log2'],
             'randomforestclassifier__max_depth' : [6,7,8],}
rf_grid = GridSearchCV(rf_pipeline, rf_params, scoring='accuracy', cv=5)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
y_pred = rf_model.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()