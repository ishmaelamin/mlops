"""import json

from azureml.core import Webservice


def main(service):
    # Creating input data
    print("Creating input data")
    data = {"data": [[ 1,2,3,4 ], [ 10,9,8,7 ]]}
    input_data = json.dumps(data)

    # Calling webservice
    print("Calling webservice")
    output_data = service.run(input_data)
    predictions = output_data.get("predict")
    assert type(predictions) == list


if __name__ == "__main__":
    main()
    
"""
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
import json

from azureml.core import Webservice

#We will tweak the pre-processing function from before to handle missing data better, too...

# re-define a function to convert an object (categorical) feature into an int feature
# 0 = most common category, highest int = least common.
def getObjectFeature(df, col, datalength=1460):
    if df[col].dtype!='object': # if it's not categorical..
        print('feature',col,'is not an object feature.')
        return df
    else:
        df1 = df
        counts = df1[col].value_counts() # get the counts for each label for the feature
#         print(col,'labels, common to rare:',counts.index.tolist()) # get an ordered list of the labels
        df1[col] = [counts.index.tolist().index(i) 
                    if i in counts.index.tolist() 
                    else 0 
                    for i in df1[col] ] # do the conversion
        return df1 # make the new (integer) column from the conversion
    
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
