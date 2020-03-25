# Data analysis
import numpy as np
import pandas as pd
# Data visualization
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
# Cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
# Modelling
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Ignore warnings
import warnings
warnings.simplefilter("ignore")

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Exploratory data analysis
# train_data.head()

# Data Dictionary
# -Survived: 0 = No, 1 = Yes
# -Pclass: Ticket class 1=1st;class 2=2nd; class 3=3rd
# -SibSp: Number of siblings/spouse aboard
# -Parch: Number of parents/children aboard
# -Titanic: ticket number -Cabin: cabin number
# -Embarked: Port of embarkation C=Cherbourg, Q=Queenstown, S=Southampton

# test_data.head()
# Size of datasets
# train_data.shape, test_data.shape
# train_data.info()
# test_data.info()

# We can clearly see, there are missing values for the features:
# -Age -Cabin -Embarked -Fare

# Lets get the counts of the missing values for each feature

# train_data.isnull().sum()
# test_data.isnull().sum()