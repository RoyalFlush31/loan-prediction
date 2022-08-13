import pandas as pd
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

### Functions
# Show data
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

    sns.set(style="ticks")
    sns.pairplot(data, hue="Loan_Status")
    # sns.boxplot(data=data, orient="v", palette="Set2")
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`. """
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
def cleanGN(data):
    males = data[data.Gender == 'Male']
    females = data[data.Gender == 'Female']
    sample = resample(males, n_samples=len(females), replace=False, random_state=0)
    data = pd.concat([sample, females])

    data['Gender'] = np.where(data['Gender'] == 'Female', 1, 0)
    data['Married'] = np.where(data['Married'] == 'Yes', 1, 0)
    data['Self_Employed'] = np.where(data['Self_Employed'] == 'Yes', 1, 0)
    data['Loan_Status'] = np.where(data['Loan_Status'] == 'Y', 1, 0)
    data['Education'] = np.where(data['Education'] == 'Graduate', 1, 0)

    lb_Mar = LabelEncoder()
    data['Dependents'] = lb_Mar.fit_transform(data['Dependents'])
    data['Property_Area'] = lb_Mar.fit_transform(data['Property_Area'])

    data = data.drop(columns="Loan_ID")
    data = data.fillna(data.mean())
    data = data.select_dtypes(exclude=['object'])

    return data, lb_Mar

def clean(data):
    data['Gender'] = np.where(data['Gender'] == 'Female', 1, 0)
    data['Married'] = np.where(data['Married'] == 'Yes', 1, 0)
    data['Self_Employed'] = np.where(data['Self_Employed'] == 'Yes', 1, 0)
    data['Loan_Status'] = np.where(data['Loan_Status'] == 'Y', 1, 0)
    data['Education'] = np.where(data['Education'] == 'Graduate', 1, 0)

    lb_Mar = LabelEncoder()
    data['Dependents'] = lb_Mar.fit_transform(data['Dependents'])
    data['Property_Area'] = lb_Mar.fit_transform(data['Property_Area'])

    data = data.drop(columns="Loan_ID")
    data = data.fillna(data.mean())
    data = data.select_dtypes(exclude=['object'])

    return data, lb_Mar

def split(data, target):
    X = data.drop(target, axis=1)
    Y = data['Loan_Status']
    return train_test_split(X, Y, stratify=Y, test_size=0.25, random_state=0)

# Create models
def kNearest(X_train, X_test, Y_train, Y_test):
    # To find the optimal number of neighbors n, we try different n’s using a loop and print the results
    accuracies_tr = []
    accuracies_te = []
    for n in range(0, 20):
        knnmodel = KNeighborsClassifier(n_neighbors=n + 1)
        knnmodel.fit(X_train, Y_train)

        Y_train_pred = knnmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies_tr.append(acctr)

        Y_test_pred = knnmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies_te.append(accte)

    # Find best n fitting for train & test
    opt_n_tr = accuracies_tr.index(max(accuracies_tr)) + 1
    opt_n_te = accuracies_te.index(max(accuracies_te)) + 1
    opt_n = int((opt_n_te + opt_n_tr) / 2)

    # To build the classifier
    knnmodel = KNeighborsClassifier(n_neighbors=opt_n)
    knnmodel.fit(X_train, Y_train)
    return knnmodel

def decisionTree(X_train, X_test, Y_train, Y_test):
    # To find the optimal max depth, we try different k’s using a loop and print the results
    accuracies_tr = []
    accuracies_te = []
    for k in range(0, 20):
        etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=k + 1)
        etmodel.fit(X_train, Y_train)

        Y_train_pred = etmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies_tr.append(acctr)

        Y_test_pred = etmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies_te.append(accte)

    # Find best k fitting for train & test
    opt_k_tr = accuracies_tr.index(max(accuracies_tr)) + 1
    opt_k_te = accuracies_te.index(max(accuracies_te)) + 1
    opt_k = int((opt_k_te + opt_k_tr) / 2)

    # To build the classifier
    etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=opt_k)
    etmodel.fit(X_train, Y_train)
    return etmodel

def randomForest(X_train, X_test, Y_train, Y_test):
    # To find the optimal number of neighbors n, we try different n’s using a loop and print the results
    accuracies_tr = []
    accuracies_te = []
    ntrees = (np.arange(20) + 1) * 10
    for k in range(0, 20):
        rfmodel = RandomForestClassifier(random_state=0, n_estimators=ntrees[k])
        rfmodel.fit(X_train, Y_train)

        Y_train_pred = rfmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies_tr.append(acctr)

        Y_test_pred = rfmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies_te.append(accte)

    # Find best n fitting for train & test
    opt_k_tr = accuracies_tr.index(max(accuracies_tr)) + 1
    opt_k_te = accuracies_te.index(max(accuracies_te)) + 1
    opt_k = int((opt_k_te + opt_k_tr) / 2)

    # To build the classifier
    rfmodel = RandomForestClassifier(random_state=0, n_estimators=ntrees[opt_k])
    rfmodel.fit(X_train, Y_train)
    return rfmodel

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
    print(report)

    plot_confusion_matrix(cmtr, classes=['no', 'yes'], title=f'Train: Confusion matrix {model_name}')
    plot_confusion_matrix(cmte, classes=['no', 'yes'], title=f'Test: Confusion matrix {model_name}')

# Predict new data
def user_input(data, labelEncoder):
    user_gender = input("What is your gender? (m/f) ")
    user_married = input("Are you married? (y/n) ")
    user_dependents = input("How many dependents do you have? (0, 1, 2 or 3+) ")
    user_graduate = input("Do you have gratuated? (y/n) ")
    user_self_employed = input("Are you self-employed? (y/n) ")
    user_applicant_income = int(input("How high is your income per month? "))
    if user_married == "y":
        user_coapplicant_income = int(input("How high is the income of your coapplicant per month? "))
    else:
        user_coapplicant_income = 0
    user_loan_amount = int(input("How much do you want to loan per month? "))
    user_loan_amount_term = int(input("How many years should your loan last? ")) * 12
    user_credit_history = input("Do you already have a credit history? (y/n) ")
    user_property_area = input("Where do you live? (Rural, Semiurban, Urban) ")
    choice = [user_gender, user_married, user_dependents, user_graduate, user_self_employed, user_applicant_income,
              user_coapplicant_income, user_loan_amount, user_loan_amount_term, user_credit_history, user_property_area]

    #choice = [0, 0, '0', 1, 0, 10000, 0, 48, 60, 1, 'Urban'] # Used for testing

    choice = pd.DataFrame([choice], columns=[
        "Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome",
        "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area"])

    choice = choice.replace(to_replace=["y", "n"], value=[1, 0])
    choice = choice.replace(to_replace=["f", "m"], value=[1, 0])

    aI = 'ApplicantIncome'
    cAI = "CoapplicantIncome"
    LA = "LoanAmount"
    lAT = "Loan_Amount_Term"
    choice[aI] = (choice[aI] - data[aI].min()) / (data[aI].max() - data[aI].min())
    choice[cAI] = (choice[cAI] - data[cAI].min()) / (data[cAI].max() - data[cAI].min())
    choice[LA] = (choice[LA] - data[LA].min()) / (data[LA].max() - data[LA].min())
    choice[lAT] = (choice[lAT] - data[lAT].min()) / (data[lAT].max() - data[lAT].min())

    choice['Property_Area'] = labelEncoder.fit_transform(choice['Property_Area'])
    choice['Dependents'] = labelEncoder.fit_transform(choice['Dependents'])

    return choice.values


### Execution

# 1. Import data
data = pd.read_csv("LoanPrediction.csv")
with open('user_questions.json') as json_file:
    _data = json.load(json_file)
user_questions = pd.json_normalize(_data)
print(user_questions['welcome_text'].iloc[0])

# 2. Clean the data
GN = input(user_questions['gender_neutral'].iloc[0])
if GN == "y":
    data, labelEncoder = cleanGN(data)
elif GN == "n":
    data, labelEncoder = clean(data)
else:
    print("Wrong answer")
    pass

# 3. Normalize data
ndata = (data - data.min()) / (data.max() - data.min())

# 4. Split the data into train/test sets
X_train, X_test, Y_train, Y_test = split(ndata, 'Loan_Status')

# 5. Create and train a model
kNmodel = kNearest(X_train, X_test, Y_train, Y_test)
dTmodel = decisionTree(X_train, X_test, Y_train, Y_test)
rFmodel = randomForest(X_train, X_test, Y_train, Y_test)

# 6. Measure accuracy
accuracy(X_train, X_test, Y_train, Y_test, kNmodel)
accuracy(X_train, X_test, Y_train, Y_test, dTmodel)
accuracy(X_train, X_test, Y_train, Y_test, rFmodel)


# 7. Make predictions
#choice = user_input(data, labelEncoder)
#result = int(model.predict(choice).item(0))

result = 0
print(" ")
if result == 0:
    print(user_questions['decline_text'].iloc[0])
else:
    print(user_questions['confirm_text'].iloc[0])


