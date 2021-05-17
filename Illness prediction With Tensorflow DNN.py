# -*- coding: utf-8 -*-
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import seaborn as sns
import os

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import binarize
from sklearn.linear_model import LogisticRegression


from sklearn import model_selection
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve, precision_score, roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import resample

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from mlxtend.classifier import StackingClassifier

import tensorflow as tf
import argparse

#from evalClassModel import evalClassModel

df = pd.read_csv('toy_dataset.csv')
#print(df.head(10))
df.info()
#print(df.shape)
#print(df.columns)
#print(df.dtypes)
#print(df.City.unique())
#print(df.Gender.unique())
#print(df.Illness.unique())
#print(df.isnull().sum())

negative_income = df[df["Income"] < 0]
# print(negative_income)

df.drop(df[df.Income < 0].index, inplace = True)
# print(df.describe())



plt.figure(figsize=(10, 5))
values = df['Illness'].value_counts().values
counts = df['Illness'].value_counts().index
colors = ['green', 'red']
plt.pie(values, labels=counts, colors=['g', 'r'], autopct='%1.1f')
plt.title('Illness')
plt.legend()

plt.figure(figsize=(10, 5))
values = df['Gender'].value_counts().values
counts = df['Gender'].value_counts().index
colors = ['teal', 'orange']
plt.pie(values, labels=counts, colors=colors, autopct='%1.1f')
plt.title('Comparison of Gender')
plt.legend()


plt.figure(figsize=(10,5))
sns.barplot(x = df['City'].value_counts().values, y = df['City'].value_counts().index)
plt.title('Population per city')
plt.xlabel('Counts')
plt.ylabel('Cities')

plt.figure(figsize=(10, 5))
sns.countplot(x="Gender", hue="Illness", palette="rocket", data=df)

g = sns.FacetGrid(df, col='Illness', height=5)
g = g.map(sns.histplot, "Age")

plt.figure(figsize=(10, 5))
sns.countplot(x="City", hue="Gender", palette="rocket", data=df)

plt.figure(figsize=(10, 5))
sns.histplot(df["Age"], color='r')
plt.title("Age distribution")

plt.figure(figsize=(10, 5))
sns.distplot(df["Income"], color='g')
plt.title("Income distribution")

fig = plt.figure(figsize=(10, 5))
sns.histplot(df[df["Gender"] == "Male"]["Income"], color='b')
sns.histplot(df[df["Gender"] == "Female"]["Income"], color='r')
fig.legend(labels=['Male', 'Female'])
plt.title("Income distribution - Man and Woman")

cities = ['Dallas', 'New York City', 'Los Angeles', 'Mountain View', 'Boston', 'Washington D.C.', 'Austin', 'San Diego']
colors = ['orange', 'red', 'blue', 'teal', 'brown', 'turquoise', 'olive', 'plum']
fig = plt.figure(figsize=(10, 5))
for i, j in zip(cities, colors):
    n, bins, patches = plt.hist(df[df.City == i].Income, bins=150, color=j, label=i)
plt.legend()
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.title('Income frequency per city')

plt.figure(figsize=(10,5))
sns.barplot(x=df["City"], y=df["Income"], data=df)

plt.show()


df['Male'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df['Illness'] = df['Illness'].apply(lambda x: 1 if x == 'Yes' else 0)
cities = pd.get_dummies(df['City'], drop_first=False)

df_new = pd.concat([df, cities], axis=1)
df_new.drop('City', axis=1, inplace=True)
df_new.drop('Gender', axis=1, inplace=True)
df_new.drop('Number', axis=1, inplace=True)

df_new.hist(bins=10, figsize=(15, 10), color="#2c5af2")
corr_matrix = df_new.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, linewidths=.5, cmap="Blues")
plt.show()

#print(df_new.shape)
#print(df.describe())
#print(df_new.columns)
#print(df_new.head(10))

y = df_new.Illness
X = df_new
'''----------------------------------Before oversampling--------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X.drop('Illness', axis=1), y, test_size=0.25, random_state=0)
print(y_test.value_counts())
print(y_test.mean())
'''
# --------------------------------------Oversampling----------------------------------------------------------
X_train1, X_test, y_train1, y_test = train_test_split(X.drop('Illness', axis=1), y, test_size=0.25, random_state=0)
print(y.shape)
print(X.shape)
print(X.head(10))

not_ill = X[X.Illness == 0]
ill = X[X.Illness == 1]
ill_upsampled = resample(ill,
                         replace=True,
                         n_samples=len(not_ill),
                         random_state=100)
upsampled = pd.concat([not_ill, ill_upsampled])
print(upsampled.Illness.value_counts())
print(upsampled.shape)
y = upsampled.Illness
X = upsampled.drop('Illness', axis=1)

print(y.shape)
print(X.shape)
X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.25, random_state=0)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
sns.countplot(y_train)
plt.title('Balanced training data')
plt.show()

# -------------------------------------------------------------------------------------------------------------
def allModels():
    models = []
    models.append(('Logistic Regression..............', LogisticRegression()))
    models.append(('Linear Discriminant Analysis.....', LinearDiscriminantAnalysis()))
    models.append(('K-Nearest Neighbors..............', KNeighborsClassifier()))
    models.append(('Decision Tree Classifier.........', DecisionTreeClassifier()))
    models.append(('Naive Bayes......................', GaussianNB()))
    results = []
    names = []
    for name, model in models:
        KFold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=KFold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = '%s: %f (%f)' % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Algorithms comparison
    fig = plt.figure()
    fig.suptitle('Algorithm comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(['LR', 'LDA', 'KNN', 'DTC', 'NB'])
    plt.show()

# Create dictionaries for final graph
methodDict = {}
rmseDict = ()

def evalClassModel(model, y_test, y_pred_class, plot=False):

    print('Accuracy:', accuracy_score(y_test, y_pred_class))
    print('Null accuracy:\n', y_test.value_counts())
    print('Percentage of ones:', y_test.mean())
    print('Percentage of zeros:', 1 - y_test.mean())
    print('True:', y_test.values[0:25])
    print('Pred:', y_pred_class[0:25])
    confusion = confusion_matrix(y_test, y_pred_class)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    # visualize Confusion Matrix
    sns.heatmap(confusion, annot=True, fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    accuracy = accuracy_score(y_test, y_pred_class)
    print('Classification Accuracy:', accuracy)
    print('Classification Error:', 1 - accuracy_score(y_test, y_pred_class))
    false_positive_rate = FP / float(TN + FP)
    print('False Positive Rate:', false_positive_rate)
    print('Precision:', f1_score(y_test, y_pred_class, average='weighted', labels=np.unique(y_pred_class)))
    print('AUC Score:', roc_auc_score(y_test, y_pred_class))
    print('Cross-validated AUC:', cross_val_score(model, X, y, cv=10, scoring='roc_auc').mean())
    print('First 10 predicted responses:\n', model.predict(X_test)[0:10])
    print('First 10 predicted probabilities of class members:\n', model.predict_proba(X_test)[0:10])
    print(model.predict_proba(X_test)[0:10, 1])
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    if plot == True:
        plt.rcParams['font.size'] = 12
        plt.hist(y_pred_prob, bins=8)
        plt.xlim(0, 1)
        plt.title('Histogram of predicted probabilities', )
        plt.xlabel('Predicted Illness probability')
        plt.ylabel('Frequency')

    y_pred_prob = y_pred_prob.reshape(-1, 1)
    y_pred_class = binarize(y_pred_prob, threshold=0.3)[0]

    print('First 10 predicted probabilities:\n', y_pred_prob[0:10])

    # ROC curve
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    if plot == True:
        plt.figure()

        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve for Illness classifier')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.legend(loc="lower right")
        plt.show()

    # One way of setting threshold
    predict_mine = np.where(y_pred_prob > 0.50, 1, 0)
    confusion = confusion_matrix(y_test, predict_mine)
    print(confusion)

    return accuracy


def logisticRegression():  # train a logistic regression model on the training set
    print('---------------------- Logistic regression analysis ---------------------\n')
    log_reg = LogisticRegression(solver='liblinear')
    log_reg.fit(X_train, y_train)
    y_pred_class = log_reg.predict(X_test)
    accuracy_score = evalClassModel(log_reg, y_test, y_pred_class, True)

    methodDict['Log. Reg.'] = accuracy_score * 100  # Data for final graph


def KNN():  # train a K-Nearest Neighbors model on the training set
    print('---------------------- K-Nearest Neighbors analysis ---------------------\n')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_class = knn.predict(X_test)
    accuracy_score = evalClassModel(knn, y_test, y_pred_class, True)

    methodDict['KNN'] = accuracy_score * 100  # Data for final graph


def treeClassifier():  # train a Decision Tree Classifier model on the training set
    print('---------------------- Decision Tree Classifier analysis ----------------\n')
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    y_pred_class = tree.predict(X_test)
    accuracy_score = evalClassModel(tree, y_test, y_pred_class, True)

    methodDict['Tree Class.'] = accuracy_score * 100  # Data for final graph


def randomForest(): # train a Random Forest Classifier model on the training set
    print('---------------------- Random Forests Classifier analysis ----------------\n')
    forest = RandomForestClassifier(max_depth=None, min_samples_leaf=8,
                                    min_samples_split=2, n_estimators=20, random_state=1)
    my_forest = forest.fit(X_train, y_train)
    y_pred_class = my_forest.predict(X_test)
    accuracy_score = evalClassModel(forest, y_test, y_pred_class, True)

    methodDict['R. Forest'] = accuracy_score * 100

def featuresImportance():
    xgbmodel = XGBClassifier()
    xgbmodel.fit(X_train, y_train)
    feat_importances = pd.Series(xgbmodel.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(12).plot(kind='barh')

def plotSuccess():
    my_dict = methodDict
    perc = [float(i) for i in my_dict.values()]
    sns.barplot(x=list(my_dict.keys()), y=perc)
    plt.show()


X_train = X_train.rename(columns={"Los Angeles": "LA", "Mountain View": "MW", "New York City": "NYC",
                                      "San Diego": "SD", "Washington D.C.": "WDC"})
X_test = X_test.rename(columns={"Los Angeles": "LA", "Mountain View": "MW", "New York City": "NYC",
                                    "San Diego": "SD", "Washington D.C.": "WDC"})
def neuralNet():


    batch_size = 100
    train_steps = 1000

    def train_input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        return dataset.shuffle(1000).repeat().batch(batch_size)

    def eval_input_fn(features, labels, batch_size):
        features=dict(features)
        if labels is None:
            inputs = features
        else:
            inputs = (features, labels)
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)
        return dataset


    age = tf.feature_column.numeric_column("Age")
    male = tf.feature_column.numeric_column("Male")
    income = tf.feature_column.numeric_column("Income")
    boston = tf.feature_column.numeric_column("Boston")
    austin = tf.feature_column.numeric_column("Austin")
    dallas = tf.feature_column.numeric_column("Dallas")
    la = tf.feature_column.numeric_column("LA")
    mw = tf.feature_column.numeric_column("MW")
    nyc = tf.feature_column.numeric_column("NYC")
    sandiego = tf.feature_column.numeric_column("SD")
    wdc = tf.feature_column.numeric_column("WDC")

    feature_columns = [age, male, income, boston, austin, dallas, la, mw, nyc, sandiego, wdc]

    model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[10, 10],
                                        optimizer='Adam')
    model.train(input_fn=lambda: train_input_fn(X_train, y_train, batch_size), steps=train_steps)
    eval_result = model.evaluate(
        input_fn=lambda: eval_input_fn(X_test, y_test, batch_size))

    print('\nTest set accuracy for neural network: {accuracy:0.2f}\n'.format(**eval_result))

    # Data for final graph
    accuracy = eval_result['accuracy'] * 100
    methodDict['NN DNNClasif.'] = accuracy


allModels()
logisticRegression()
KNN()
randomForest()
treeClassifier()
neuralNet()
plotSuccess()
featuresImportance()
plt.show()
