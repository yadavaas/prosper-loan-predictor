import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle


# loading the dataset
data = pd.read_csv('updated_data.csv')
df = data[['AvailableBankcardCredit', 'BankcardUtilization', 'BorrowerAPR',
          'BorrowerRate', 'DebtToIncomeRatio', 'DelinquenciesLast7Years',
          'EmploymentStatus', 'EmploymentStatusDuration',
          'EstimatedEffectiveYield', 'EstimatedLoss', 'EstimatedReturn',
          'IncomeRange', 'Investors', 'LenderYield', 'LoanOriginalAmount',
          'LoanOriginationQuarter', 'LoanStatus', 'Occupation',
          'OpenRevolvingMonthlyPayment',
          'ProsperScore', 'RevolvingCreditBalance', 'StatedMonthlyIncome', 'Term',
          'TotalCreditLinespast7years', 'TotalTrades']]


# one hot encoding
# Listing the columns with object datatype
col = df.dtypes[df.dtypes == 'object'].index

df_num = pd.get_dummies(data=df, columns=col, drop_first=True)
df = df_num

# Dependent variable
y = df['LoanStatus']
# Independent variable
X = df.drop(['LoanStatus'], axis=1)
print(X)
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
# Create principal components
pca = PCA()
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)


LR = LogisticRegression(solver='liblinear')

LR = LR.fit(X_pca, y)

pickle.dump(LR, open('LR_pickle.pkl', 'wb'))
pickle.dump(pca, open('pca_pickle.pkl', 'wb'))
pickle.dump(scaler, open('scaler_pickle.pkl', 'wb'))