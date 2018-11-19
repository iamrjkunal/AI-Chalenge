import numpy as np
import pandas as pd
#######################train############################
# Importing train dataset
dataset = pd.read_csv('train.csv')
nonzeros_train =dataset[dataset != 0].count()
columns = ['var_d3','var_d4','var_d5','var_d6','var_d7','var_l_1','var_l_2','var_l_3','var_l_4','var_l_5','var_l_6','var_l_7','var_l_8','var_l_62','var_l_63','var_l_81','var_l_82','var_l_83','var_l_84']
dataset.drop(columns, inplace=True, axis=1)

# Taking care of missing data
nan_val_train = dataset.isnull().astype(int).sum()
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

imputer = DataFrameImputer()
imputer = imputer.fit(dataset)
dataset = imputer.transform(dataset)
X = dataset.iloc[:, 3:].values
y = dataset.iloc[:, 1].values

##########################test####################################
# Import test dataset
test_dataset = pd.read_csv('test.csv')
nonzeros_test =test_dataset[test_dataset != 0].count()
test_dataset.drop(columns, inplace=True, axis=1)

# Taking care of missing data
nan_val_test = test_dataset.isnull().astype(int).sum()
from sklearn.base import TransformerMixin
test_dataset = imputer.transform(test_dataset)
X_out=test_dataset.iloc[:, 3:].values 
outformat =test_dataset.iloc[:, 1].values
outformat= test_dataset['APP_ID_C'].astype('str')

# Encoding categorical data 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder().fit(X[:, 1])
X[:, 1] = labelencoder_X.transform(X[:, 1])
X_out[:, 1] = labelencoder_X.transform(X_out[:, 1])
labelencoder_X = LabelEncoder().fit(X[:, 3])
X[:, 3] = labelencoder_X.transform(X[:, 3])
X_out[:, 3] = labelencoder_X.transform(X_out[:, 3])
X= X.astype(float)
X_out = X_out.astype(float)
'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X)
y_train = y
X_test = sc_X.transform(X_out)


from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(n_estimators=200, max_depth=2)
#from sklearn.linear_model import LogisticRegression
#testing C=0.001, 0.01, 0.1, 1, 10, 100, 1000
#classifier = LogisticRegression(C=3.1, n_jobs=-1)
#from sklearn.svm import LinearSVC
#classifier = LinearSVC(max_iter=100, dual=False)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_score = classifier.predict_proba(X_test)
y_score = y_score[:,1]

'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
per=(cm[0,0] + cm[1,1])/(cm[0,0] + cm[0,1] + cm[1,0] + cm[1,1])
#print(per)
from sklearn.metrics import roc_auc_score
roc_wt= roc_auc_score(y_test, y_score,average='weighted')
#print(roc_wt)
'''
'''
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 1000,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
'''

outformat = list(outformat)
outformat = np.array([outformat])
outformat = np.transpose(outformat)
y_score = np.array([y_score])
y_score = np.transpose(y_score)
finaloutput= np.concatenate((outformat, y_score), axis=1)
ans = pd.DataFrame({'APP_ID_C':finaloutput[:, 0],'target':finaloutput[:, 1]})
ans.to_csv('ans.csv', encoding='utf-8', index=False)










