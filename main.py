import pandas as pd
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
data = pd.read_csv("LoanPrediction.csv")

def sample(data):
    #sample = resample(data, replace=False, n_samples=200, random_state=0)
    #data = data.dropna(True)
    males = data[data.Gender == 'Male']
    females = data[data.Gender == 'Female']
    sample = resample(males, n_samples=len(females), replace=False, random_state=0)
    data = pd.concat([sample, females])

    data['Gender'] = np.where(data['Gender'] == 'Female', 1, 0)
    data['Married'] = np.where(data['Married'] == 'Yes', 1, 0)
    data['Self_Employed'] = np.where(data['Self_Employed'] == 'Yes', 1, 0)
    data['Loan_Status'] = np.where(data['Loan_Status'] == 'Y', 1, 0)

    lb_Mar = LabelEncoder()
    data['Dependents'] = lb_Mar.fit_transform(data['Dependents'])
    data['Education'] = lb_Mar.fit_transform(data['Education'])
    data['Property_Area'] = lb_Mar.fit_transform(data['Property_Area'])

    data = data.drop(columns="Loan_ID")
    data = data.fillna(data.mean())

    return data

def show(data):
    sns.set(style="ticks")
    sns.pairplot(data, hue="Loan_Status")
    #sns.boxplot(data=data, orient="v", palette="Set2")
    plt.show()

def analyze(data):
    print(f"Length: {len(data)}")
    print(f'Loan Declines | Total {len(data.loc[data.Loan_Status == "N"])}',
          "| Male ", len(data.loc[(data.Loan_Status == "N") & (data.Gender == "Male")]),
          "| Female ", len(data.loc[(data.Loan_Status == "N") & (data.Gender == "Female")])
          )
    print(f'Loan Approvals | Total {len(data.loc[data.Loan_Status == "Y"])}',
          "| Male ", len(data.loc[(data.Loan_Status == "Y") & (data.Gender == "Male")]),
          "| Female ", len(data.loc[(data.Loan_Status == "Y") & (data.Gender == "Female")]),
          len(data.loc[data.Gender == "Male"]), len(data.loc[data.Gender == "Female"])
          )
    print(f'Column Names: {data.columns}')
    print(f'Column Distinct Values: {data.loc[data.Loan_Status == "Y"].nunique()}')

    show(data)


def process(data):
    data = data.select_dtypes(exclude=['object'])
    # Standardization
    #data = (data - data.mean()) / data.std()
    # Normalization
    data = (data - data.min()) / (data.max() - data.min())
    #print(data.describe())
    #sns.set(style="ticks")
    #sns.pairplot(data, hue="Loan_Status")
    #plt.show()
    return data

def partion(data):
    X = data.drop('Loan_Status', axis=1)
    Y = data['Loan_Status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        stratify=Y, test_size=0.25, random_state=0)
    print(len(X_train), len(X_test), len(Y_train), len(Y_test))

def Knearest(data):
    X = data.drop('Loan_Status', axis=1)
    Y = data['Loan_Status']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        stratify=Y, test_size=0.25, random_state=0)
    report = pd.DataFrame(columns=['Model', 'Acc.Train', 'Acc.Test'])

    # To build the classifier
    knnmodel = KNeighborsClassifier(n_neighbors=7)
    knnmodel.fit(X_train, Y_train)
    Y_train_pred = knnmodel.predict(X_train)

    # To examine the quality of the classifier we can calculate the confusion matrix (reference/prediction) and the accuracy
    cmtr = confusion_matrix(Y_train, Y_train_pred)
    print("Confusion Matrix Training:\n", cmtr)
    acctr = accuracy_score(Y_train, Y_train_pred)
    print("Accurray Training:", acctr)

    # To test the generalizability of the classifier, it must be applied to the test data
    Y_test_pred = knnmodel.predict(X_test)
    cmte = confusion_matrix(Y_test, Y_test_pred)
    print("Confusion Matrix Testing:\n", cmte)
    accte = accuracy_score(Y_test, Y_test_pred)
    print("Accurray Test:", accte)
    Y_test_pred = knnmodel.predict_proba(X_test)
    #print(Y_test_pred)

    # To find the optimal value for k, we try different k’s using a loop and print the results
    accuracies = []
    for k in range(1, 21):
        knnmodel = KNeighborsClassifier(n_neighbors=k)
        knnmodel.fit(X_train, Y_train)
        Y_test_pred = knnmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        print(k, accte)
        accuracies.append(accte)


data = process(sample(data))
#print(data)
Knearest(data)
#analyze(data)
#show(data)



