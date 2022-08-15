import pandas as pd
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import numpy as np
import itertools
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


### Functions
# Show data
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
         plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center", fontsize=16, color="white" if cm[i, j] > thresh else "black")
         plt.ylabel('True label')
         plt.xlabel('Predicted label')
         plt.tight_layout()
    plt.show()

# Prepare data
def clean(data):
    data = data.drop(columns="Loan_ID")
    data['Gender'] = np.where(data['Gender'] == 'Female', 1, 0)
    data['Married'] = np.where(data['Married'] == 'Yes', 1, 0)
    data['Self_Employed'] = np.where(data['Self_Employed'] == 'Yes', 1, 0)
    data['Loan_Status'] = np.where(data['Loan_Status'] == 'Y', 1, 0)
    data['Education'] = np.where(data['Education'] == 'Graduate', 1, 0)
    data['Dependents'] = np.where(data['Dependents'] == '3+', 3, data['Dependents'])
    data['Dependents'] = data['Dependents'].astype(np.float)
    data['Property_Area']= data['Property_Area'].replace(to_replace=["Urban", "Semiurban", "Rural"], value=[0, 1, 2])
    data = data.fillna(data.mean())
    return data

def preProcess(data):
    data = clean(data)
    X_ = data.drop('Loan_Status', axis=1)
    Y = data['Loan_Status']
    X = (X_ - X_.min()) / (X_.max() - X_.min())

    return train_test_split(X, Y, stratify=Y, test_size=0.255, random_state=0)

# Create models
def trainModel(X_train, Y_train, model, param_grid):
    CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=10)
    CV_model.fit(X_train, Y_train)
    model = model.set_params(**CV_model.best_params_)
    model.fit(X_train, Y_train)
    return model

# Evalualte models
def accuracy(X_train, X_test, Y_train, Y_test, model):
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    model_name = model.__class__.__name__

    # To examine the quality of the classifier we can calculate the confusion matrix (reference/prediction) and the accuracy
    cmtr = confusion_matrix(Y_train, Y_train_pred)
    cmte = confusion_matrix(Y_test, Y_test_pred)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accte = accuracy_score(Y_test, Y_test_pred)

    labelEncoder = LabelEncoder()
    Y_test_code = labelEncoder.fit_transform(Y_test)
    Y_test_pred_code = labelEncoder.fit_transform(Y_test_pred)
    f1score = f1_score(Y_test_code, Y_test_pred_code)

    report = pd.DataFrame(columns=['Model', 'Accuracy Training', 'Accuracy Testing', 'F1 Score'])
    report.loc[len(report)] = [model_name, acctr, accte, f1score]
    #print(report)

    #plot_confusion_matrix(cmtr, classes=['no', 'yes'], title=f'Train: Confusion matrix {model_name}')
    #plot_confusion_matrix(cmte, classes=['no', 'yes'], title=f'Test: Confusion matrix {model_name}')
    return f1score

# Predict new data
def user_input(data):
    user_gender = input(user_questions['gender'].iloc[0])
    user_married = input(user_questions['married'].iloc[0])
    user_dependents = input(user_questions['dependents'].iloc[0])
    user_graduate = input(user_questions['education'].iloc[0])
    user_self_employed = input(user_questions['self_empolyed'].iloc[0])
    user_applicant_income = int(input(user_questions['applicant_income'].iloc[0]))
    if user_married == "y":
        user_coapplicant_income = int(input(user_questions['coapplicant_income'].iloc[0]))
    else:
        user_coapplicant_income = 0
    user_loan_amount = int(input(user_questions['loan_amount'].iloc[0]))
    user_loan_amount_term = int(input(user_questions['loan_amount_term'].iloc[0])) * 12
    user_credit_history = input(user_questions['credit_history'].iloc[0])
    user_property_area = input(user_questions['property_area'].iloc[0])
    choice = [user_gender, user_married, user_dependents, user_graduate, user_self_employed, user_applicant_income,
              user_coapplicant_income, user_loan_amount, user_loan_amount_term, user_credit_history, user_property_area]

    # Generalize user's inputs for preprocessing
    choice = pd.DataFrame([choice], columns=[
        "Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome",
        "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area"])
    choice = choice.replace(to_replace=["y", "n"], value=[1, 0])
    choice = choice.replace(to_replace=["f", "m"], value=[1, 0])
    choice = choice.replace(to_replace=["Urban", "Semiurban", "Rural"], value=[0, 1, 2])
    choice = choice.replace(to_replace=["3+"], value=[3])
    choice = choice.astype(np.int)

    # Import cleaned data to preprocess user's choice
    data = clean(data).drop('Loan_Status', axis=1)

    # Normalize and return user's choice
    choice = (choice - data.min()) / (data.max() - data.min())
    return choice


### Execution

# 1. Import data
data = pd.read_csv("LoanPrediction.csv")
with open('user_questions.json') as json_file:
    _data = json.load(json_file)
user_questions = pd.json_normalize(_data)
print(user_questions['welcome_text'].iloc[0])
print(" ")
print(user_questions['wait_text'].iloc[0])
print(" ")

# 2. Preprocess data
X_train, X_test, Y_train, Y_test = preProcess(data)

# 3. Create and train models
classifiers = [KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression(), MLPClassifier()]
param_grids = [{'n_neighbors': range(1,20)}, {'criterion': ['entropy', 'gini'], 'max_depth': range(1, 21)},
               {'max_depth': range(4, 8, 2), 'n_estimators': range(10, 210, 50)}, {},
               {'solver' : ['lbfgs', 'sgd'], 'hidden_layer_sizes': range(8, 20, 2)}]
bestScore = 0
for i in range(4):
    model = trainModel(X_train, Y_train, classifiers[i], param_grids[i])
    score = accuracy(X_train, X_test, Y_train, Y_test, model)
    if score > bestScore:
        bestScore = score
        bestModel = model

# 4. Make predictions
choice = user_input(data)
result = int(bestModel.predict(choice).item(0))

print(" ")
if result == 0:
    print(user_questions['decline_text'].iloc[0])
else:
    print(user_questions['confirm_text'].iloc[0])