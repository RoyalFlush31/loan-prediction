import pandas as pd
import json

data = pd.read_csv("LoanPrediction.csv")
print(f"Datalänge: {len(data)}")
print(f'Loan Declines: {len(data.Loan_Status.loc[data.Loan_Status == "N"])}')
print(f'Loan Approvals: {len(data.Loan_Status.loc[data.Loan_Status == "Y"])}')
print(f'Data Columns: {data.columns}')

#pdata = pd.pivot_table(data[data.Loan_Status=='Y'], index = 'Gender', values = 'Loan_ID', aggfunc = 'sum')
#print(pdata)
print(data.loc[(data.Loan_Status == "Y")&(data.Gender == "Female")])


#for i in range(len(data.columns)-1):
    #print(data[data.columns[i]])
