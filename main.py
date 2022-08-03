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
from sklearn.tree import DecisionTreeClassifier

### Functions

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

def show(data):
    sns.set(style="ticks")
    sns.pairplot(data, hue="Loan_Status")
    #sns.boxplot(data=data, orient="v", palette="Set2")
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

def kNearest(X_train, X_test, Y_train, Y_test):
    # To find the optimal number of neighbors n, we try different n’s using a loop and print the results
    accuracies = []
    for n in range(1, 21):
        knnmodel = KNeighborsClassifier(n_neighbors=n)
        knnmodel.fit(X_train, Y_train)
        Y_test_pred_ = knnmodel.predict(X_test)
        accte_ = accuracy_score(Y_test, Y_test_pred_)
        #print(n, accte_)
        accuracies.append(accte_)

    # To build the classifier
    opt_k = accuracies.index(max(accuracies)) + 1
    knnmodel = KNeighborsClassifier(n_neighbors=opt_k)
    knnmodel.fit(X_train, Y_train)
    return knnmodel

def accuracy(X_train, X_test, Y_train, Y_test, model):
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    # To examine the quality of the classifier we can calculate the confusion matrix (reference/prediction) and the accuracy
    cmtr = confusion_matrix(Y_train, Y_train_pred)
    cmte = confusion_matrix(Y_test, Y_test_pred)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accte = accuracy_score(Y_test, Y_test_pred)
    print("Confusion Matrix Training:\n", cmtr)
    print("Confusion Matrix Testing:\n", cmte)

    labelEncoder = LabelEncoder()
    Y_test_code = labelEncoder.fit_transform(Y_test)
    Y_test_pred_code = labelEncoder.fit_transform(Y_test_pred)
    f1te = f1_score(Y_test_code, Y_test_pred_code)

    report = pd.DataFrame(columns=['Model', 'Acc.Train', 'Acc.Test', 'F1.score'])
    report.loc[len(report)] = ['k-NN', acctr, accte, f1te]
    print(report)

    np.set_printoptions(precision=2)
    class_names = ['no', 'yes']
    plt.figure()
    plot_confusion_matrix(cmtr, classes=class_names, title='Confusion matrix KNN train')
    plt.show()

def user_input(data, labelEncoder):
    print("")
    print("Welcome to the FUTURE Bank AG. Fill out the survey to get an automated answer if you could likely receive a loan.")
    print("")

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

def decisionTree(X_train, X_test, Y_train, Y_test):
    # find optimal max_depth
    accuracies = np.zeros((2, 20), float)
    for k in range(0, 20):
        etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=k + 1)
        etmodel.fit(X_train, Y_train)
        Y_train_pred = etmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[0, k] = acctr
        Y_test_pred = etmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[1, k] = accte

    plt.plot(range(1, 21), accuracies[0, :])
    plt.plot(range(1, 21), accuracies[1, :])
    plt.xlim(1, 20)
    plt.xticks(range(1, 21))
    plt.xlabel('Max_depth')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Accuracies (Entropy)')
    plt.show()

    etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=5)
    etmodel.fit(X_train, Y_train)

    return etmodel

### Execution

# 1. Import data
data = pd.read_csv("LoanPrediction.csv")

# 2. Clean the data
GN = input("Do you want to conduct the application gender neutrally? (y/n) ")
if GN == "y":
    data, labelEncoder = cleanGN(data)
elif GN == "n":
    data, labelEncoder = clean(data)
else:
    print("Wrong answer")
    pass

# 3. Normalize data
ndata = (data - data.min()) / (data.max() - data.min())

# 3. Split the data into train/test sets
X_train, X_test, Y_train, Y_test = split(ndata, 'Loan_Status')

# 4. Create and train a model
test = input("Do you want to conduct the application based on kNearest or DecistionTree? We recommend kN (kN/dT) ")
if test == "kN":
    model = kNearest(X_train, X_test, Y_train, Y_test)
elif test == "dT":
    model = decisionTree(X_train, X_test, Y_train, Y_test)
else:
    print("Wrong answer")
    pass

# 5. Measure accuracy (USED FOR TESTING)
accurancy_faq = input("Do you want to see the accurancy of this application? (y/n) ")
if accurancy_faq == "y":
    accuracy(X_train, X_test, Y_train, Y_test, model)
else:
    pass

# 6. Make predictions
choice = user_input(data, labelEncoder)
result = int(model.predict(choice).item(0))

print(" ")
if result == 0:
    print("We are sorry to tell you that your application is not likey to become granted.")
else:
    print("We are very delighted to tell you that your application is likey to become granted! Apply now on our Website!")


