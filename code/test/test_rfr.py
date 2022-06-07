import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # some plotting!
import seaborn as sns # so pretty!
from scipy import stats # I might use this
from sklearn.ensemble import RandomForestClassifier # checking if this is available
# from sklearn import cross_validation
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import os
cwd = os.getcwd()

def main(service):
    rfr(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, 
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=60, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)

    test = pd.read_csv(os.path.join(cwd,"test.csv"))



    # apply the model to the test data and get the output...
    X_test = test[included_features]
    for col in list(X_test):
        if X_test[col].dtype=='object':
            X_test = getObjectFeature(X_test, col, datalength=1459)
    # print(X_test.head(20))
    y_output = model.predict(X_test.fillna(0)) # get the results and fill nan's with 0
    print(y_output)

    # transform the data to be sure
    y_output = np.exp(y_output)
    print(y_output)

    # define the data frame for the results
    saleprice = pd.DataFrame(y_output, columns=['SalePrice'])
    # print(saleprice.head())
    # saleprice.tail()
    results = pd.concat([test['Id'],saleprice['SalePrice']],axis=1)
    results.head()

    # and write to output
    results.to_csv('housepricing_submission.csv', index = False)
    
if __name__ == "__main__":
    main()
