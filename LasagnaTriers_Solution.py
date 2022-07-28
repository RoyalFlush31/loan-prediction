
import os
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


os.chdir("C:\Users\tf_02\OneDrive\Desktop\Frankfurt + Uni\6 Semester\Data Analysis using Machine lerning\LasagnaTriers")
data_ori = pd.read_csv('LasagnaTriers.csv')
print(data_ori.shape)
# types
print(data_ori.dtypes)
# feature names
print(list(data_ori))
# head
print(data_ori.head(6))


# change precision to 2 places
pd.set_option("display.precision", 2)
# descriptions
print(data_ori.describe())


# check for missing values
print(data_ori.isnull().sum())


# ckeck numerical features for outliers
import seaborn as sns
sns.boxplot(data=data_ori['Age'], orient="v", palette="Set2")
sns.boxplot(data=data_ori['Weight'], orient="v", palette="Set2")
sns.boxplot(data=data_ori['Income'], orient="v", palette="Set2")
print(data_ori[['Person' ,'Income']].sort_values(by='Income', ascending=0).head(10))
sns.boxplot(data=data_ori['Car_Value'], orient="v", palette="Set2")
sns.boxplot(data=data_ori['CC_Debt'], orient="v", palette="Set2")
sns.boxplot(data=data_ori['Mall Trips'], orient="v", palette="Set2")
print(data_ori[['Person' ,'Mall Trips']].sort_values(by='Mall Trips', ascending=0).head(10))


# create dataframe with the numerical variables (ignore Person)
data = data_ori.loc[:, ['Age', 'Weight', 'Income', 'Car_Value', 'CC_Debt', 'Mall Trips']]


# convert categorical variables to numerical and add to dataframe
print(data_ori['Gender'].describe())
data['Gender'] = np.where(data_ori['Gender' ]=='Female', 1, 0)

print(data_ori['Live_Alone'].describe())
data['Live_Alone'] = np.where(data_ori['Live_Alone' ]=='Yes', 1, 0)

print(data_ori['Dwell_Type'].describe())
from sklearn.preprocessing import LabelBinarizer
bi_Dwell_Type = LabelBinarizer()
bi_dummys = bi_Dwell_Type.fit_transform(data_ori['Dwell_Type'])
print(data_ori['Dwell_Type'].value_counts())     # fit_transform orders categories alphabetically
print(bi_dummys.sum(axis=0))
DT_dummys = pd.DataFrame(bi_dummys, columns=['DT_Apt', 'DT_Condo', 'DT_Home'])
DT_dummys.head(10)
DT_dummys = DT_dummys.drop('DT_Apt', axis=1)
DT_dummys.head(10)
data = pd.concat([data, DT_dummys], axis=1)

print(data_ori['Pay_Type'].describe())
data['Pay_Type'] = np.where(data_ori['Pay_Type' ]=='Salaried', 1, 0)

print(data_ori['Nbhd'].describe())
from sklearn.preprocessing import LabelBinarizer
bi_Nbhd = LabelBinarizer()
bi_dummys = bi_Nbhd.fit_transform(data_ori['Nbhd'])
print(data_ori['Nbhd'].value_counts())     # fit_transform orders categories alphabetically
print(bi_dummys.sum(axis=0))
NH_dummys = pd.DataFrame(bi_dummys, columns=['NH_East', 'NH_South', 'NH_West'])
NH_dummys.head(10)
NH_dummys = NH_dummys.drop('NH_South', axis=1)
NH_dummys.head(10)
data = pd.concat([data, NH_dummys], axis=1)


# add target to dataframe
data['Have_Tried'] = data_ori['Have_Tried']


# check for balanced state
print(data['Have_Tried'].value_counts())
# Separate majority and minority classes
data_majority = data[data_ori.Have_Tried =='Yes']
data_minority = data[data_ori.Have_Tried =='No']
# Downsample majority class
from sklearn.utils import resample
data_majority_downsampled = resample(data_majority,
                                     replace=False,    # sample without replacement
                                     n_samples=len(data_minority),     # to match minority class
                                     random_state=0) # reproducible results
# Combine minority class with downsampled majority class
data_downsampled = pd.concat([data_majority_downsampled, data_minority])
data_downsampled = data_downsampled.reset_index(drop=True)
print(data_downsampled['Have_Tried'].value_counts())


# Separate X and Y
X = data_downsampled.drop('Have_Tried', axis=1)
Y = data_downsampled.Have_Tried


# Normalize
from sklearn import preprocessing
nscaler = preprocessing.MinMaxScaler()
X.iloc[: ,:] = nscaler.fit_transform(X.iloc[: ,:])  # Trick to keep the dataframe
print(X.head(6))
print(X.describe())


# Partition into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, test_size=0.5, random_state=0)
print(Y_train.value_counts())
print(Y_test.value_counts())

# create report dataframe
report = pd.DataFrame(columns=['Model' ,'Acc.Train' ,'Acc.Test'])


from sklearn.neighbors import KNeighborsClassifier
knnmodel = KNeighborsClassifier(n_neighbors=7)
knnmodel.fit(X_train, Y_train)

Y_train_pred = knnmodel.predict(X_train)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)

Y_test_pred = knnmodel.predict(X_test)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

# find optimal k, try different k and find their accuracy
accuracies = []
for k in range(1, 31, 2):
    knnmodel = KNeighborsClassifier(n_neighbors=k)
    knnmodel.fit(X_train, Y_train)
    Y_test_pred = knnmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    print(k, accte)
    accuracies.append(accte)

print(accuracies)

# in case of indifference (yes/no) which can appear with even numbers, the algorithm chooses the solution randomly
plt.plot(range(1, 31, 2), accuracies)
plt.xlim(1 ,30)
plt.xticks(range(1, 31, 2))
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies')
plt.show()

opt_k = np.argmax(accuracies ) *2 + 1
print('Optimal k =', opt_k)

knnmodel = KNeighborsClassifier(n_neighbors=opt_k)
knnmodel.fit(X_train, Y_train)
Y_train_pred = knnmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
Y_test_pred = knnmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
report.loc[len(report)] = ['k-NN', acctr, accte]
print(report)

# visualize confusion matrix
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
plot_confusion_matrix(knnmodel, X_test, Y_test, labels= ['yes', 'no'],
                      cmap=plt.cm.Blues, values_format='d')

# calculate f1 score
from sklearn.preprocessing import LabelEncoder
lb_churn = LabelEncoder()
Y_test_code = lb_churn.fit_transform(Y_test)
Y_test_pred_code = lb_churn.fit_transform(Y_test_pred)
from sklearn.metrics import f1_score
f1te = f1_score(Y_test_code, Y_test_pred_code)
print(f1te)

# calculate ROC and AUC and plot the curve
Y_probs = knnmodel.predict_proba(X_test)
print(Y_probs[0:6 ,:])
Y_test_probs = np.array(np.where(Y_test =='yes', 1, 0))
print(Y_test_probs[0:6])
from sklearn.metrics import roc_curve
fpr, tpr, threshold = roc_curve(Y_test_probs, Y_probs[:, 1])
print (fpr, tpr, threshold)
from sklearn.metrics import auc
roc_auc = auc(fpr, tpr)
print(roc_auc)


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
accuracies = np.zeros((2 ,20), float)
for k in range(0, 20):
    etmodel = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth= k +1)
    etmodel.fit(X_train, Y_train)
    Y_train_pred = etmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0 ,k] = acctr
    Y_test_pred = etmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1 ,k] = accte
plt.plot(range(1, 21), accuracies[0 ,:])
plt.plot(range(1, 21), accuracies[1 ,:])
plt.xlim(1 ,20)
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

# plot tree using graphviz
import graphviz
dot_data = sk.tree.export_graphviz(etmodel, out_file=None,
                                   feature_names=list(X),
                                   filled=True, rounded=True,
                                   special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("Churn_entropy")


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

accuracies = np.zeros((2 ,20), float)
for k in range(0, 20):
    gtmodel = DecisionTreeClassifier(random_state=0, max_depth= k +1)
    gtmodel.fit(X_train, Y_train)
    Y_train_pred = gtmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0 ,k] = acctr
    Y_test_pred = gtmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1 ,k] = accte
plt.plot(range(1, 21), accuracies[0 ,:])
plt.plot(range(1, 21), accuracies[1 ,:])
plt.xlim(1 ,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Comparison of Accuracies (Gini)')
plt.show()

gtmodel = DecisionTreeClassifier(random_state=0, max_depth=8)
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
report.loc[len(report)] = ['Tree (Gini)', acctr, accte]
print(report)

# plot tree
from sklearn.tree import plot_tree
fig, ax = plt.subplots(figsize=(30, 12))
plot_tree(gtmodel, feature_names=list(X), filled=True, rounded=True, max_depth=4, fontsize=10)
plt.show()

import graphviz
dot_data = sk.tree.export_graphviz(gtmodel, out_file=None,
                                   feature_names=list(X),
                                   filled=True, rounded=True,
                                   special_characters=True)
graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render("Churn_gini")

# show feature importance
list(zip(X, gtmodel.feature_importances_))
index = np.arange(len(gtmodel.feature_importances_))
bar_width = 1.0
plt.bar(index, gtmodel.feature_importances_, bar_width)
plt.xticks(index,  list(X), rotation=90) # labels get centered
plt.show()


#################
# Random Forest #
#################

from sklearn.ensemble import RandomForestClassifier
rfmodel = RandomForestClassifier(random_state=0)
rfmodel.fit(X_train, Y_train)
Y_train_pred = rfmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = rfmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

# varying max_depth
accuracies = np.zeros((2 ,20), float)
for k in range(0, 20):
    rfmodel = RandomForestClassifier(random_state=0, max_depth= k +1)
    rfmodel.fit(X_train, Y_train)
    Y_train_pred = rfmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0 ,k] = acctr
    Y_test_pred = rfmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1 ,k] = accte
plt.plot(range(1, 21), accuracies[0 ,:])
plt.plot(range(1, 21), accuracies[1 ,:])
plt.xlim(1 ,20)
plt.xticks(range(1, 21))
plt.xlabel('Max_depth')
plt.ylabel('Accuracy')
plt.title('Random Forest')
plt.show()

# varying n_estimators
accuracies = np.zeros((2 ,20), float)
ntrees = (np.arange(20 ) +1 ) *10
for k in range(0, 20):
    rfmodel = RandomForestClassifier(random_state=0, n_estimators=ntrees[k])
    rfmodel.fit(X_train, Y_train)
    Y_train_pred = rfmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[0 ,k] = acctr
    Y_test_pred = rfmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[1 ,k] = accte
plt.plot(ntrees, accuracies[0 ,:])
plt.plot(ntrees, accuracies[1 ,:])
plt.xticks(ntrees, rotation=90)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest')
plt.show()

# varying max_depth and n_estimators, interaction effect
mdepth = np.linspace(4, 8, 5)
accuracies = np.zeros((4 , 5 *20), float)
row = 0
for k in range(0, 5):
    for l in range(0, 20):
        rfmodel = RandomForestClassifier(random_state=0, max_depth=mdepth[k], n_estimators=ntrees[l])
        rfmodel.fit(X_train, Y_train)
        Y_train_pred = rfmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2 ,row] = acctr
        Y_test_pred = rfmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3 ,row] = accte
        accuracies[0 ,row] = mdepth[k]
        accuracies[1 ,row] = ntrees[l]
        row = row + 1

print(accuracies)

from tabulate import tabulate
headers = ["Max_Depth", "n_Estimators", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n" ,table)

print(accuracies[3].max())
maxi = np.array(np.where(accuracies == accuracies[3].max()))
print(maxi[0 ,:], maxi[1 ,:])
print(accuracies[: ,maxi[1 ,:]])
table = tabulate(accuracies[: ,maxi[1 ,:]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n" ,table)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = accuracies[0 ,:]
y = accuracies[1 ,:]
z = accuracies[3 ,:]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Max_Depth')
ax.set_ylabel('n_Estimators')
ax.set_zlabel('accte')
plt.show()

rfmodel = RandomForestClassifier(random_state=0, max_depth=7, n_estimators=60)
rfmodel.fit(X_train, Y_train)
Y_train_pred = rfmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = rfmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Random Forest', acctr, accte]
print(report)
# View a list of the features and their importance scores
list(zip(X_train, rfmodel.feature_importances_))


#######################
# Logistic Regression #
#######################

from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression()
lrmodel.fit(X_train, Y_train)

Y_train_pred = lrmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)

Y_test_pred = lrmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Logistic Regression', acctr, accte]


##################
# Neural Network #
##################

from sklearn.neural_network import MLPClassifier
nnetmodel = MLPClassifier(solver='lbfgs', max_iter=3000 ,hidden_layer_sizes=(17,), random_state=0)
nnetmodel.fit(X_train, Y_train)
Y_train_pred = nnetmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = nnetmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)

accuracies = np.zeros((3 ,20), float)
for k in range(0, 20):
    nnetmodel = MLPClassifier(solver='lbfgs', max_iter=3000 ,hidden_layer_sizes=(7,), random_state=0)
    nnetmodel.fit(X_train, Y_train)
    Y_train_pred = nnetmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1 ,k] = acctr
    Y_test_pred = nnetmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2 ,k] = accte
    accuracies[0 ,k] = k+ 1
plt.plot(range(1, 21), accuracies[1, :])
plt.plot(range(1, 21), accuracies[2, :])
plt.xlim(1, 20)
plt.xticks(range(1, 21))
plt.xlabel('Hidden Neurons')
plt.ylabel('Accuracy')
plt.title('Neural Network')
plt.show()

from tabulate import tabulate

headers = ["Hidden Neurons", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)
maxi = np.array(np.where(accuracies == accuracies[2:].max()))
table = tabulate(accuracies[:, maxi[1, :]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)

nnetmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(7, 4), random_state=0)
nnetmodel.fit(X_train, Y_train)
Y_train_pred = nnetmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = nnetmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['Neural Network', acctr, accte]

##########################
# Support Vector Machine #
###########################

# linear kernel
from sklearn.svm import SVC

LinSVCmodel = SVC(kernel='linear', C=10, random_state=0)
LinSVCmodel.fit(X_train, Y_train)
Y_train_pred = LinSVCmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = LinSVCmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['SVM (Linear)', acctr, accte]

accuracies = np.zeros((3, 21), float)
costs = np.linspace(0, 40, 21)
costs[0] = 0.5
for k in range(0, 21):
    LinSVCmodel = SVC(kernel='linear', C=costs[k], random_state=0)
    LinSVCmodel.fit(X_train, Y_train)
    Y_train_pred = LinSVCmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1, k] = acctr
    Y_test_pred = LinSVCmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2, k] = accte
    accuracies[0, k] = costs[k]
plt.plot(costs, accuracies[1, :])
plt.plot(costs, accuracies[2, :])
plt.xlim(1, 20)
plt.xticks(costs, rotation=90)
plt.xlabel('Cost')
plt.ylabel('Accuracy')
plt.title('Linear SVM')
plt.show()

from tabulate import tabulate

headers = ["Cost", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)

# radial kernel
from sklearn.svm import SVC

accuracies = np.zeros((3, 21), float)
costs = np.linspace(0, 40, 21)
costs[0] = 0.5
for k in range(0, 21):
    RbfSVCmodel = SVC(kernel='rbf', C=costs[k], gamma=0.2, random_state=0)
    RbfSVCmodel.fit(X_train, Y_train)
    Y_train_pred = RbfSVCmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1, k] = acctr
    Y_test_pred = RbfSVCmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2, k] = accte
    accuracies[0, k] = costs[k]
plt.plot(costs, accuracies[1, :])
plt.plot(costs, accuracies[2, :])
plt.xlim(1, 20)
plt.xticks(costs, rotation=90)
plt.xlabel('Cost')
plt.ylabel('Accuracy')
plt.title('Radial SVM')
plt.show()

from tabulate import tabulate

headers = ["Cost", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)

accuracies = np.zeros((3, 21), float)
gammas = np.linspace(0, 4.0, 21)
gammas[0] = 0.1
for k in range(0, 21):
    RbfSVCmodel = SVC(kernel='rbf', C=1, gamma=gammas[k], random_state=0)
    RbfSVCmodel.fit(X_train, Y_train)
    Y_train_pred = RbfSVCmodel.predict(X_train)
    acctr = accuracy_score(Y_train, Y_train_pred)
    accuracies[1, k] = acctr
    Y_test_pred = RbfSVCmodel.predict(X_test)
    accte = accuracy_score(Y_test, Y_test_pred)
    accuracies[2, k] = accte
    accuracies[0, k] = gammas[k]
plt.plot(gammas, accuracies[1, :])
plt.plot(gammas, accuracies[2, :])
plt.xticks(gammas, rotation=90)
plt.xlabel('Gamma')
plt.ylabel('Accuracy')
plt.title('Radial SVM')
plt.show()

from tabulate import tabulate

headers = ["Gamma", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)

n = 21
accuracies = np.zeros((4, n * n), float)
costs = np.linspace(0, 20, n)
costs[0] = 0.5
gammas = np.linspace(0, 4.0, n)
gammas[0] = 0.1
row = 0
for k in range(0, n):
    for l in range(0, n):
        RbfSVCmodel = SVC(kernel='rbf', C=costs[k], gamma=gammas[l], random_state=0)
        RbfSVCmodel.fit(X_train, Y_train)
        Y_train_pred = RbfSVCmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2, row] = acctr
        Y_test_pred = RbfSVCmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3, row] = accte
        accuracies[0, row] = costs[k]
        accuracies[1, row] = gammas[l]
        row = row + 1

from tabulate import tabulate

headers = ["Cost", "Gamma", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)

maxi = np.array(np.where(accuracies == accuracies[3].max()))
print(maxi[1, :])
print(accuracies[:, maxi[1, :]])
table = tabulate(accuracies[:, maxi[1, :]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = accuracies[0, :]
y = accuracies[1, :]
z = accuracies[3, :]
ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
ax.set_xlabel('Cost')
ax.set_ylabel('Gamma')
ax.set_zlabel('accte')
plt.show()

RbfSVCmodel = SVC(kernel='rbf', C=14, gamma=0.4, random_state=0)
RbfSVCmodel.fit(X_train, Y_train)
Y_train_pred = RbfSVCmodel.predict(X_train)
cmtr = confusion_matrix(Y_train, Y_train_pred)
print("Confusion Matrix Training:\n", cmtr)
acctr = accuracy_score(Y_train, Y_train_pred)
print("Accurray Training:", acctr)
Y_test_pred = RbfSVCmodel.predict(X_test)
cmte = confusion_matrix(Y_test, Y_test_pred)
print("Confusion Matrix Testing:\n", cmte)
accte = accuracy_score(Y_test, Y_test_pred)
print("Accurray Test:", accte)
report.loc[len(report)] = ['SVM (Radial)', acctr, accte]

# polynomial kernel
n = 21
accuracies = np.zeros((4, n * 5), float)
costs = np.linspace(0, 20, n)
costs[0] = 0.5
degrees = np.linspace(1, 5, 5)
row = 0
for k in range(0, n):
    for l in range(0, 5):
        PolySVCmodel = SVC(kernel='poly', C=costs[k], degree=degrees[l], random_state=0)
        PolySVCmodel.fit(X_train, Y_train)
        Y_train_pred = PolySVCmodel.predict(X_train)
        acctr = accuracy_score(Y_train, Y_train_pred)
        accuracies[2, row] = acctr
        Y_test_pred = PolySVCmodel.predict(X_test)
        accte = accuracy_score(Y_test, Y_test_pred)
        accuracies[3, row] = accte
        accuracies[0, row] = costs[k]
        accuracies[1, row] = degrees[l]
        row = row + 1

from tabulate import tabulate

headers = ["Cost", "Degree", "acctr", "accte"]
table = tabulate(accuracies.transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)

maxi = np.array(np.where(accuracies == accuracies[3:].max()))
print(maxi[1, :])
print(accuracies[:, maxi[1, :]])
table = tabulate(accuracies[:, maxi[1, :]].transpose(), headers, tablefmt="plain", floatfmt=".3f")
print("\n", table)

################
# Final Report #
################

print(report)