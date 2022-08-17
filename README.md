# prosper-loan-predictor
# Predictive Analysis Using Social Profile in Online P2P Lending Market
Online peer-to-peer (P2P) lending markets enable individual consumers to borrow from, and lend money to, one another directly.
In this project we predicted the status of the borrower, that whether the borrower should be Accepted or should be considered as High Risk, based on the information provided by them, especially by analyzing their financial and social strength.

## Understanding the Dataset
The dataset we are working on is the **prosperLoanData** which contains information from **2005** to **2014**.

- This dataset contains the information about *financial and social strength* of the borrower from **2005** to **2014**. 

- Some columns store values given as user input by the borrower, such as **IsBorrowerHomeowner**, **IncomeRange** etc.

- Some columns store values retrived by the **Prosper** company at the time the credit profile was pulled, such as **CurrentDelinquencies**, **BankcardUtilization** etc.

- Rest of the columns store the current status of the loan, such as **LoanStatus**, **LoanCurrentDaysDelinquent** etc.

- The label of the dataset is whether the borrower should be **Accepted** (labeled as **1**) or should be considered as **High Risk** (labeled as **0**). 


## Preprocessing

First of all, we checked the *missing values* in the dataset and found out that there were almost **15 %** values missing.

- To fix the missing values we first *removed* the columns containing more than **80 %** values.

- Then we looked for missing values in *object* data type columns and imputed them with suitable values such as **not applicable**, **other** etc. 

- Finally we imputed rest columns with **mean** values.

Then we took the **LoanStatus** and converted it's values into **Accepted** or **High Risk** categories as per following conversion.
- *Current*, *Completed* and *Final Payment in Progress* into ***Accepted***
- *Cancelled*, *Defaulted*, *Charged off* and all *Past dues* into ***High Risk***.

Also, we subsetted the dataset by removing the all the rows created before 2009 due to some discrepency and high volume of data, as we know there is a financial crisis in 2008 which affects the whole borrower condition and quality of loans, therefore it was more appropriate to subset this data set by post-crisis years. So, we substted the dataset and named it **cleaned_data**.


## EDA and Feature Engineering
**Exploratory Data Analysis (EDA) is an approach to analyze the data using visual techniques. It is used to discover trends, patterns, or to check assumptions with the help of statistical summary and graphical representations.**


**Introduction:**

- **cleaned_data** dataset comprises of 84853 rows and 69 columns.
- Dataset comprises of *int*, *float*, *object* and *bool* data types. 

**Analysis:**

- Plotted histograms and barplots to see the distribution of data for each column and found that most of the variables are normally distributed, while others were skewed.

- As, we have already selected **LoanStatus** as our **Target** variable, we plotted all our ***histograms*** and ***barplots/countplots*** for *LoanStatus* = ***Accepted*** and *LoanStatus* = ***High Risk*** side by side and compared the results.

- Based on our Analysis we found that some variables are highly related to the Target variable, such as ***LoanCurrentDaysDelinquent***, ***LP_CustomerPayments***, ***LP_CustomerPrincipalPayments*** etc. but when we read the document about the variable definition we found that these variables show the current status of on-going loan. Therefor, we decided to drop all these variables from our analyses, since they might lead to the **Target Leakage**.

- Aprt from these variables we found some other variables which showed some relationship with *Target Variable* and help to figure out *Accepted* and *High Risk* values. We chose those variables for our model building and to try out different combinations of them and choose the *best possible variables* for better accuracy. 

**Correlation Plot of Numerical Variables:**

We obtained the **correlation** of all the numerical variables and plotted them using **heat map** to highlight the highly correlated variables.

**Visualisation of Variables:**

We performed the **scatter plot** to visualize the relation between the variables.

**Outlier Detection:**

We also detected the outliers using **box plot** on variables which were highly skewed and were shoing long tails, such as **MonthlyStatedIncome**, **MonthlyLoanPayment** etc.
 
 
## Preprocessing Again

Now, after observing the outliers, we fixed them by boundary substitution.

Also, at last again we dropped some variables which we found unuseful for our model building such as all *Date columns* and *Unique key columns* etc.

Finally, we subsetted our dataset to **updated_data**.


## Model Building
In this Project we first built the three models which are: **Logistic Regression**, **Decision Tree Classifier** and **Naive Bayes**. Then we compared the results of these models. Description of Models and Evaluation metrcis are as follows:


#### Metrics considered for Model Evaluation
**Accuracy** , **F1 Score** and **AUC score**
- **Accuracy**: What proportion of actual positives and negatives is correctly classified?
- **F1 Score** : Harmonic mean of Precision and Recall. Where: *Precision*: What proportion of predicted positives are truly positive ? and *Recall*: What proportion of actual positives is correctly classified ?
- **AUC score** : Area under the curve of ROC curve (Receiver Operating Characteristic curve). It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) 

### Our Models
#### Logistic Regression
- Logistic Regression helps find how probabilities are changed with actions.
- The function is defined as P(y) = 1 / 1+e^-(A+Bx) 
- Logistic regression involves finding the **best fit S-curve** where A is the intercept and B is the regression coefficient. The output of logistic regression is a probability score.

#### Decision Tree Classifier
- Decision Tree is a **Supervised learning technique** that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where **internal nodes represent the features of a dataset, branches represent the decision rules** and **each leaf node represents the outcome.**
- In a Decision tree, there are two nodes, which are the **Decision Node** and **Leaf Node**. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.
- It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.
- A decision tree simply asks a question, and based on the answer **(Yes/No)**, it further split the tree into subtrees.

#### Naive Bayes
- Naïve Bayes algorithm is a supervised learning algorithm, which is based on **Bayes theorem** and used for solving classification problems.
- It is a probabilistic classifier, which means it predicts on the basis of the **probability** of an object.

  --- ***Bayes theorem*** ---
  - Bayes' theorem is also known as **Bayes' Rule** or **Bayes' law**, which is used to determine the probability of a hypothesis with prior knowledge. It depends on the **conditional probability**.
  - The formula for Bayes' theorem is given as:
  - **P(A|B) =  (P(B|A)P(A)) / P(B)** 
  - where: *P(A|B) is Posterior probability:* Probability of hypothesis A on the observed event B.
  - *P(B|A) is Likelihood probability:* Probability of the evidence given that the probability of a hypothesis is true.
  - *P(A) is Prior Probability:* Probability of hypothesis before observing the evidence.
  - *P(B) is Marginal Probability:* Probability of Evidence.


### Testing Models
We tested all three models for different set of variables and with and without feature engineering techniques, and we found that feature engineering techniques were giving better reults, so, we used them in our final model.

||Accuracy|F1 Score|AUC score|
| :---------------| :----:|:----:| :---:|
|**Logistic Regression**|0.90|0.86|0.82|
|**Decision Tree Classifier**|0.89|0.86|0.79|
|**Naive Bayes**|0.87|0.86|0.78|


From the above table and based on the **confusion matrix** we found that **Logistic Regression** model is giveng best results, So, we chose that for deployment.

### Approach to Final Model Building for Deployment
#### 1. Applying One Hot Encoding to categorical columns
After separating dependent and independent variables as **X** and **y**, we used *pd.get_dummies()* function to perform one hot encoding to all categorical variables to convert them into numerical form.


#### 2. Scaling the data points
We applied *Standard Scaler* to scale the data point.
```
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```


#### 3. PCA transformation
Then we applied PCA transformation
~~~
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
~~~

#### 4. Applying Logistic regression
Then we trained the x_pca with Logistic Regression function.
```
from sklearn.linear_model import LogisticRegression     
LR = LogisticRegression( solver='liblinear').fit(X,y)
```

#### 5. Dumping into pickle file
After fitting the model we dupmed it into pickle file. The files dumped itno pickle file are:
- LR_pickle.pkl
- pca_pickle.pkl
- scaler_pickle.pkl

we dumped *pca* and *scaler* along with *LR* because they can be used to transform the user input parameters based on trained model.


## Deployment
you can access our app by following this link [prosper-loan-status-predictor](https://prosper-loan-predictor-team-a.herokuapp.com/)

### Streamlit
- It is a tool that lets you creating applications for your machine learning model by using simple python code.
- We have written a python code for our app using Streamlit; the app asks the user to enter the several data.
- The output of our app will be **Accepted** or **High Risk** borrower, and if the borrower has been predicted as *Accepted* it also calculates **ROI** (Return on Investment) which you can find out by clicking on the check box.
- Our app also shows the **probability** of both *High Risk* and *Accepted* type.
- The app runs on local host.
- To deploy it on the internt we have to deploy it to Heroku.

### Heroku
We deploy our Streamlit app to [ Heroku.com](https://www.heroku.com/). In this way, we can share our app on the internet with others. 
We prepared the needed files to deploy our app sucessfully:
- Procfile: contains run statements for app file and setup.sh.
- setup.sh: contains setup information.
- requirements.txt: contains the libraries must be downloaded by Heroku to run app file (prosper-app.py)  successfully 
- prosper-app.py.py: contains the python code of a Streamlit web app.
- LR_pickle.pkl: contains the pickle file of trained Logistic regression
- pca_pickle.pkl: contains the pickle file of fitted pca
- scaler_pickle.pkl: contains the pickle file of fitted scaler transformation
- updated_data.csv: contains the dataset we used for model building. this is needed to transform the user input.
- model-building.py: this python file is our model building file.
