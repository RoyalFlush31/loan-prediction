import pandas as pd
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn as sk

def cleanGN(data):
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

    data = data.select_dtypes(exclude=['object'])
    # Standardization
    # data = (data - data.mean()) / data.std()
    # Normalization
    data = (data - data.min()) / (data.max() - data.min())

    return data

def clean(data):
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

    data = data.select_dtypes(exclude=['object'])
    # Standardization
    # data = (data - data.mean()) / data.std()
    # Normalization
    data = (data - data.min()) / (data.max() - data.min())

    return data

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
                                                        stratify=Y, test_size=0.2, random_state=0)
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

    # in case of indifference (yes/no) which can appear with even numbers, the algorithm chooses the solution randomly

    opt_k = 9 #manuelle eingefügt
    print('Optimal k =', opt_k)

    accte = accuracy_score(Y_test, Y_test_pred)
    report.loc[len(report)] = ['k-NN', acctr, accte]
    print(report)

    '''#####################'''

    '''
    # calculate f1 score
    from sklearn.preprocessing import LabelEncoder
    lb_churn = LabelEncoder()
    Y_test_code = lb_churn.fit_transform(Y_test)
    Y_test_pred_code = lb_churn.fit_transform(Y_test_pred)
    from sklearn.metrics import f1_score
    f1te = f1_score(Y_test_code, Y_test_pred_code)
    print("F1-score",f1te)

    # calculate ROC and AUC and plot the curve
    Y_probs = knnmodel.predict_proba(X_test)
    print("Y_probs:", Y_probs[0:6, :])
    Y_test_probs = np.array(np.where(Y_test == 1, 1, 0))
    print("Y_test_probs:", Y_test_probs[0:6])
    from sklearn.metrics import roc_curve
    fpr, tpr, threshold = roc_curve(Y_test_probs, Y_probs[:, 1])
    print("fpr:",fpr, "tpr:", tpr,"threshold:", threshold)
    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)
    print("roc and auc:",roc_auc)

    from sklearn.tree import DecisionTreeClassifier
    etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0)
    etmodel.fit(X_train, Y_train)
    Y_train_pred = etmodel.predict(X_train)
    cmtr = confusion_matrix(Y_train, Y_train_pred)
    print("Confusion Matrix Training:\n", cmtr)
    acctr = accuracy_score(Y_train, Y_train_pred)
    print("Accurray Training:", acctr)
    Y_test_pred = etmodel.predict(X_test)
    cmte = confusion_matrix(Y_test, Y_test_pred)
    print("Confusion Matrix Testing:\n", cmte)
    accte = accuracy_score(Y_test, Y_test_pred)
    print("Accurray Test:", accte)

    Y_train_pred_prob = etmodel.predict_proba(X_train)

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
    Y_train_pred = etmodel.predict(X_train)
    cmtr = confusion_matrix(Y_train, Y_train_pred)
    print("Confusion Matrix Training:\n", cmtr)
    acctr = accuracy_score(Y_train, Y_train_pred)
    print("Accurray Training:", acctr)
    Y_test_pred = etmodel.predict(X_test)
    cmte = confusion_matrix(Y_test, Y_test_pred)
    print("Confusion Matrix Testing:\n", cmte)
    accte = accuracy_score(Y_test, Y_test_pred)
    print("Accurray Test:", accte)
    report.loc[len(report)] = ['Tree (Entropy)', acctr, accte]

    # plot tree
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(30, 12))
    plot_tree(etmodel, feature_names=list(X), filled=True, rounded=True, max_depth=4, fontsize=10)
    plt.show()

    ##### # plot tree using graphviz
    import graphviz
    dot_data = sk.tree.export_graphviz(etmodel, out_file=None,
                                       feature_names=list(X),
                                       filled=True, rounded=True,
                                       special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render("Churn_entropy")
    
    '''
    #     Gini      #
    from sklearn.tree import DecisionTreeClassifier
    gtmodel = DecisionTreeClassifier(random_state=0)
    gtmodel.fit(X_train, Y_train)
    Y_train_pred = gtmodel.predict(X_train)
    cmtr = confusion_matrix(Y_train, Y_train_pred)
    print("Confusion Matrix Training:\n", cmtr)
    acctr = accuracy_score(Y_train, Y_train_pred)
    print("Accurray Training:", acctr)
    Y_test_pred = gtmodel.predict(X_test)
    cmte = confusion_matrix(Y_test, Y_test_pred)
    print("Confusion Matrix Testing:\n", cmte)
    accte = accuracy_score(Y_test, Y_test_pred)
    print("Accurray Test:", accte)

    accuracies = np.zeros((2, 20), float)
    for k in range(0, 20):
        gtmodel = DecisionTreeClassifier(random_state=0, max_depth=k + 1)
        gtmodel.fit(X_train, Y_train)
        Y_train_pred = gtmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[0, k] = acctr
        Y_test_pred = gtmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[1, k] = accte
    plt.plot(range(1, 21), accuracies[0, :])
    plt.plot(range(1, 21), accuracies[1, :])
    plt.xlim(1, 20)
    plt.xticks(range(1, 21))
    plt.xlabel('Max_depth')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Accuracies (Gini)')
    plt.show()

#print(data)

#analyze(data)
#show(data)

data = pd.read_csv("LoanPrediction.csv")
# 1. Import data
GN = str(input('Please enter if you want the Analysis performed Gender-Neutral? (Y/N)'))
if GN == 'Y':
    data = cleanGN(data)
    Knearest(data)
elif GN == 'N':
    data = clean(data)
    Knearest(data)
else:
    print('Input Error')



# 2. Clean the data
#data = clean(data)

# 3. Split the data into train/test sets

# 4. Create a model

#Knearest(data)
# 5. Train the model
# 6. Make predictions
# 7. Evaluate and improve



