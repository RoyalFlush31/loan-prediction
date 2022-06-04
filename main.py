import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("LoanPrediction.csv")


print(f"Length: {len(data)}")
print(f'Loan Declines: {len(data.loc[data.Loan_Status == "N"])}')
print(f'Loan Approvals: {len(data.loc[data.Loan_Status == "Y"])}')
print(f'Column Names: {data.columns}')
print(f'Column Distinct Values: {data.loc[data.Loan_Status == "Y"].nunique()}')


data["Loan_Status"].replace({"Y": 0, "N": 1}, inplace=True)
sns.set(style="ticks")
sns.pairplot(data, hue="Loan_Status")
plt.show()

