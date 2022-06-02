import pandas as pd
import json

data = pd.read_csv("LoanPrediction.csv")
print(f"Datalänge: {len(data)}")
print(f'Loan Declines: {len(data.Loan_Status.loc[data.Loan_Status == "N"])}')
print(f'Loan Approvals: {len(data.Loan_Status.loc[data.Loan_Status == "Y"])}')
print(f'Data Columns: {data.columns}')

