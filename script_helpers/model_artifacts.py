import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost
from xgboost import XGBClassifier

#from category_encoders import OneHotEncoder ## TODO: this might be a better encoder (preserves names)

data_train = pd.read_csv('/pkghome/training_data.csv',low_memory=False, dtype={'Service':'str', 'Anest Type':'str'} )
labels_train = pd.read_csv('/pkghome/training_labels.csv',low_memory=False )
## sklearn's methods assume array-like (all columns numeric)
Y = labels_train['Mortality_30d']


#pipeline = Pipeline([
    #("selector", ColumnTransformer([
        #('drop', make_column_selector(pattern="xwv" ))
    #], remainder="passthrough")) , 
#])
#data_train.drop(columns="caseid", inplace=True)


ct = make_column_transformer( 
     (OneHotEncoder(), make_column_selector(dtype_include=object)) ,  
     remainder='passthrough')

data_train2 = ct.fit_transform(data_train)
pickle.dump(ct,open('/pkghome/transform.p', 'wb'))


X_train, X_test, y_train, y_test = train_test_split(data_train2, Y, test_size=0.33, random_state=101)

X_train = X_train[np.isfinite(y_train)  ]
y_train = y_train [np.isfinite(y_train)  ]

X_test = X_test[np.isfinite(y_test)  ]
y_test = y_test [np.isfinite(y_test)  ]


xgbmodel = XGBClassifier(n_estimators=25,max_depth=15, learning_rate=.2, random_state=101, gamma=1.5 ,objective='binary:logistic')

xgbmodel.fit(X=X_train, y=y_train)

y_pred = xgbmodel.predict_proba(X_test)[:,1]
print("Accuracy:",metrics.roc_auc_score(y_test, y_pred))
xgbmodel.save_model('/pkghome/Mortality_30d_xgb.xgb')

imp = IterativeImputer(max_iter=5, random_state=0)

imp = imp.fit(data_train2)

pickle.dump(imp,open('/pkghome/impute.p', 'wb'))


data_train2 = imp.transform(data_train2 )
X_train, X_test, y_train, y_test = train_test_split(data_train2, Y, test_size=0.33, random_state=101)

X_train = X_train[np.isfinite(y_train)  ]
y_train = y_train [np.isfinite(y_train)  ]

X_test = X_test[np.isfinite(y_test)  ]
y_test = y_test [np.isfinite(y_test)  ]

## fit and save the sklearn method

lr_reg_model = LogisticRegression(solver='liblinear', random_state=0)
lr_reg_model.fit(X_train, y_train)
lr_reg_model.coef_
y_pred=lr_reg_model.predict_proba(X_train)[:,1]
print("Accuracy:",metrics.roc_auc_score(y_train, y_pred))
y_pred = lr_reg_model.predict_proba(X_test)[:,1]
print("Accuracy:",metrics.roc_auc_score(y_test, y_pred))
pickle.dump(lr_reg_model,open('/pkghome/Mortality_30d_lr.p', 'wb'))

regressor = RandomForestClassifier(n_estimators=100, random_state=101, max_depth=5, min_samples_split=50 )
regressor.fit(X_train, y_train)

y_pred = regressor.predict_proba(X_train)[:,1]
print("Accuracy:",metrics.roc_auc_score(y_train, y_pred))

y_pred = regressor.predict_proba(X_test)[:,1]
print("Accuracy:",metrics.roc_auc_score(y_test, y_pred))

pickle.dump(regressor,open('/pkghome/Mortality_30d_rf.p', 'wb'))




