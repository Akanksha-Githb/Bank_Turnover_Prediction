#!/usr/bin/env python
# coding: utf-8

# ###  Importing important Libraries

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import sklearn as sk
import warnings 
warnings.filterwarnings("ignore")


# In[2]:


# Importing your dataset
data = pd.read_csv('Churn_Modelling.csv')
data.head()


# ### Data Preprocessing

# In[3]:


# Dropping the unneccessary columns 
data_1 = data.drop(['RowNumber','CustomerId','Surname'],axis = 1)


# In[4]:


# Checking the null values of all columns
data_1.info()


# In[5]:


# count of null values
data_1.isnull().sum()


# In[6]:


# Remove any white spacing in the column
data_1.columns = data_1.columns.str.strip()


# In[7]:


# Getting column name of all dataset
data_1.columns


# In[8]:


# checking data satistics 
data_1.describe()


# In[9]:


data_1.select_dtypes(include=['object']).columns


# ## Exploratory Data Analysis

# In[47]:


# List of Categorical variables to plot 
plt.figure(figsize = (16,9))
grouped_data = data_1.groupby(['Geography', 'Exited']).size().unstack()

# Create a bar plot
ax = grouped_data.plot.bar(stacked=False)
# Set labels and title
plt.xlabel('Geography')
plt.ylabel('Count')
plt.title('Exited Customers by Geography (0 vs 1)')

# Set the legend
plt.legend(['Not Exited (0)', 'Exited (1)'])

# Rotate x-axis labels if needed
plt.xticks(rotation=45)
# Display the plot 
plt.show()


# In[11]:


plt.figure(figsize = (16,9))
grouped_data = data_1.groupby(['Gender', 'Exited']).size().unstack()

# Create a bar plot
ax = grouped_data.plot.bar(stacked=False)

# Set labels and title
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Exited Customers by Gender (0 vs 1)')

# Set the legend
plt.legend(['Not Exited (0)', 'Exited (1)'])

# Rotate x-axis labels if needed
plt.xticks(rotation=45)
# Display the plot
#plt.tight_layout()
plt.show()


# In[51]:


# List of Categorical variables to plot 
plt.figure(figsize = (16,9))
grouped_data = data_1.groupby(['HasCrCard', 'Exited']).size().unstack()

# Create a bar plot
ax = grouped_data.plot.bar(stacked=False)
# Set labels and title
plt.xlabel('HasCrCard')
plt.ylabel('Count')
plt.title('Exited Customers by HasCrCard (0 vs 1)')
# Rotate x-axis labels if needed
plt.xticks(rotation=360)
# Set the legend
plt.legend(['Not Exited (0)', 'Exited (1)'])

# Display the plot 
plt.show()


# In[55]:


# List of Categorical variables to plot 
plt.figure(figsize = (16,9))
grouped_data = data_1.groupby(['IsActiveMember', 'Exited']).size().unstack()

# Create a bar plot
ax = grouped_data.plot.bar(stacked=False)
# Set labels and title
plt.xlabel('IsActiveMember')
plt.ylabel('Count')
plt.title('Exited Customers by IsActiveMember (0 vs 1)')
# Rotate x-axis labels if needed
plt.xticks(rotation=360)
# Set the legend
plt.legend(['Not Exited (0)', 'Exited (1)'])

# Display the plot 
plt.show()


# In[14]:


# Get the hist plot of all variable to see their distribution
data_1.hist(figsize = (10,15)) # here we checking normality 
plt.show()


# In[15]:


# get list of categorical variables
cat_vars = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

# create figure with subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
axs = axs.flatten()

# create histplot for each categorical variable
for i, var in enumerate(cat_vars):
    sns.histplot(x=var, hue='Exited', data=data_1, ax=axs[i], multiple="fill", kde=False, element="bars", fill=True, stat='density')
    axs[i].set_xticklabels(data_1[var].unique(), rotation=90)
    axs[i].set_xlabel(var)

# adjust spacing between subplots
fig.tight_layout()

# show plot
plt.show()


# In[16]:


# Geographical distribution of different countries 
plt.pie(data_1['Geography'].value_counts(normalize = True), labels=data_1['Geography'].unique(), autopct='%1.1f%%', startangle=90)
# Title of the plot
plt.title('Geography Distribution')
# Call tight_layout to adjust the positions
fig.tight_layout()
# Display the figure
plt.show()


# In[17]:


#Gender Distribution
plt.pie(data_1['Gender'].value_counts(normalize = True), labels=data_1['Gender'].unique(), autopct='%1.1f%%', startangle=90)
# Title of the plot
plt.title('Gender Distribution')
# Call tight_layout to adjust the positions
fig.tight_layout()
# Display the figure
plt.show()


# In[18]:


# Percentage if people that has card 
plt.pie(data_1['HasCrCard'].value_counts(normalize = True), labels=data_1['HasCrCard'].unique(), autopct='%1.1f%%', startangle=90)
plt.title('HasCrCard Distribution')
fig.tight_layout()
plt.show()


# In[19]:


plt.pie(data_1['IsActiveMember'].value_counts(normalize = True), labels=data_1['IsActiveMember'].unique(), autopct='%1.1f%%', startangle=90)
plt.title('IsActiveMember Distribution')
fig.tight_layout()
plt.show()


# In[21]:


# plotting of numerical variable to see their distribution and know about the outliers
# Defining the numerical variables
num_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts','EstimatedSalary']
# Defining the number of plot in the given layout
fig,axs = plt.subplots(nrows=2,ncols = 3,figsize=(20,10))
# Flatten the axs array
axs = axs.flatten()
# iterating over the numerical variables
for i ,var in enumerate(num_vars):
    # PLotting the box plot of numerical variables
    sns.boxplot(x = var,data = data_1,ax =axs[i])
# Call tight_layout to adjust the positions    
fig.tight_layout()
# Display the plot
plt.show()


# In[22]:


# plotting of numerical variable to see their distribution with exited column and know about the outliers
# Defining the numerical variables
num_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts','EstimatedSalary']
# Create a 2x3 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
# Flatten the axs array
axs = axs.flatten()
# Iteration over the subplots using a single loop
for i, var in enumerate(num_vars):
    # Perform same operation on each subplot
    sns.boxplot(y=var, x='Exited', data=data_1, ax=axs[i])
# Call tight_layout to adjust the positions
fig.tight_layout()
# Display the plot
plt.show()


# In[23]:


# plotting of numerical variable to see their distribution with violineplot
# Defining the numerical variables
num_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts','EstimatedSalary']
# Create a 3x3 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
# Flatten the axs array
axs = axs.flatten()
# Iteration over the subplots using a single loop
for i, var in enumerate(num_vars):
     # Perform same operation on each subplot
    sns.violinplot(x=var, data=data_1, ax=axs[i])
fig.tight_layout()
plt.show()


# In[24]:


num_vars = ['CreditScore', 'Age', 'Tenure', 'Balance', 
            'NumOfProducts','EstimatedSalary']
# Create a 3x3 grid of subplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
# Flatten the axs array
axs = axs.flatten()
# Iteration over the subplots using a single loo
for i, var in enumerate(num_vars):
    sns.violinplot(y=var, data=data_1, x='Exited', ax=axs[i])
fig.tight_layout()
plt.show()


# In[25]:


#Correlation Heatmap (print the correlation score each variables)
plt.figure(figsize=(20, 16))
sns.heatmap(data_1.corr(), fmt='.2g', annot=True)


# In[26]:


from sklearn import preprocessing
# loop over each column in the DataFrame where dtype is 'object'
for col in data_1.select_dtypes(include=['object']).columns:
    # Initization of label encoder object 
    label_encoder = preprocessing.LabelEncoder()
    # Fit the encoder to the unique values in the column
    label_encoder.fit(data_1[col].unique())
    # Transform the column using the encoder
    data_1[col] = label_encoder.transform(data_1[col])
    # print the column name and the unique encoded values
    print(f"{col}:{data_1[col].unique()}")


# ## Train Test Split

# In[27]:


from sklearn.model_selection import train_test_split
# Select the features (X) and the targets variables(y)
   x = data_1.drop('Exited',axis = 1)
   y = data_1['Exited']
   
# Split the data into training and testing dataset (order matter)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[28]:


x_train.shape, y_train.shape,x_test.shape,y_test.shape


# ## Remove the outliers from train data using the Z-Score

# In[29]:


data_1.describe()


# In[30]:


from scipy import stats
# Define the column that you want to remove the oulliers
selected_columns = ['Age','NumOfProducts','CreditScore']
# Calculate the Z-scores for the selected columns in the training data
z_scores = np.abs(stats.zscore(x_train[selected_columns]))
# set a threshold value for outliers detections(e.g 3)
threshold = 3
# Find the indices of outliers based on the threshold
outlier_indices = np.where(z_scores > threshold)[0]

# Remove the outliers from the training data
x_train = x_train.drop(x_train.index[outlier_indices])
y_train = y_train.drop(y_train.index[outlier_indices])


# In[31]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
model_1  = DecisionTreeClassifier(class_weight = 'balanced')
# This step is used for hyperparameter tuining and  optimize the complexity of the model 
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4],
    'random_state': [0, 42]
}


# In[32]:


# Perform a grid search with cross valdiation to find best hyperparameter
grid_search = GridSearchCV(model_1,param_grid,cv = 5)
grid_search.fit(x_train,y_train)
#print the best hyperparameter
print(grid_search.best_params_)


# In[33]:


# Provide model with these hyperparameter
model_1 = DecisionTreeClassifier(random_state=0,max_depth=3,min_samples_leaf=1,min_samples_split=2,class_weight = 'balanced')
model_1.fit(x_train,y_train)


# In[34]:


# R-square value of the model
model_1.score(x_train,y_train)


# In[35]:


# Accuracy score of the model
from sklearn.metrics import accuracy_score
y_pred = model_1.predict(x_test)
print('Accuracy Score:',round(accuracy_score(y_test,y_pred)*100,2),"%")


# In[36]:


# Selecting important features  from the decision tree model
imp_df = pd.DataFrame({'Feature Name': x_train.columns,'Importance': model_1.feature_importances_})
fi = imp_df.sort_values(by = 'Importance', ascending = False)
fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Top 10 Feature Importance Each Attributes (Decision Tree)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()


# In[37]:


# A look into confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[39]:


# Plotting heat maps for confusion matrix
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidths=.5, annot=True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score for Decision Tree: {0}'.format(model_1.score(x_test, y_test))
plt.title(all_sample_title, size = 15)


# ### Random Forest

# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rfc = RandomForestClassifier(class_weight = 'balanced')
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2', None],
    'random_state': [0, 42]
}

# Perform a grid search with cross-validation to find the best hyperparameters
grid_search = GridSearchCV(rfc, param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Print the best hyperparameters
print(grid_search.best_params_)


# In[41]:


from sklearn.ensemble import RandomForestClassifier
model_2 = RandomForestClassifier(random_state=0, max_features='sqrt', n_estimators=100, class_weight='balanced')
model_2.fit(x_train, y_train)


# In[42]:


model_2.score(x_train,y_train)


# In[43]:


model_2.score(x_test,y_test)


# In[44]:


y_pred = model_2.predict(x_test)


# In[45]:


y_pred


# In[46]:


imp_df = pd.DataFrame({
    "Feature Name": x_train.columns,
    "Importance": model_2.feature_importances_
})
fi = imp_df.sort_values(by="Importance", ascending=False)

fi2 = fi.head(10)
plt.figure(figsize=(10,8))
sns.barplot(data=fi2, x='Importance', y='Feature Name')
plt.title('Top 10 Feature Importance Each Attributes (Random Forest)', fontsize=18)
plt.xlabel ('Importance', fontsize=16)
plt.ylabel ('Feature Name', fontsize=16)
plt.show()  

