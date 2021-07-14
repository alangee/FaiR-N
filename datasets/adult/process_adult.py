from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import numpy as np
import pandas as pd

def load_adult(dataFile):
    df = pd.read_csv(dataFile, header=None, delimiter=r",\s+",)
    # add header
    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    
    # convert income
    df["Income"] = df["Income"].map({ "<=50K": 0, ">50K": 1 })
    
    
    y_all = df["Income"].values
    df.drop("Income", axis=1, inplace=True,)
    
    df.drop("CapitalGain", axis=1, inplace=True,)
    df.drop("CapitalLoss", axis=1, inplace=True,)
 
    print(df.head())
    
    df.Age = df.Age.astype(float)
    df.fnlwgt = df.fnlwgt.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.HoursPerWeek = df.HoursPerWeek.astype(float)
    
    
    df = pd.get_dummies(df, columns=[
        "WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship",
        "Race", "Gender", "NativeCountry",
    ])
    
    standard_scaler_cols = ["Age", "fnlwgt", "EducationNum", "HoursPerWeek",]
    other_cols = list(set(df.columns) - set(standard_scaler_cols))
    mapper = DataFrameMapper(
        [([col,], StandardScaler(),) for col in standard_scaler_cols] +
        [(col, None,) for col in other_cols]
    , df_out = True)
    

    # write transformed datasets to file
    map_fit= mapper.fit(df)
    scaled_df = mapper.transform(df)


    # add back in label
    training_label_df = pd.DataFrame(y_all, columns = ['Income'])

    scaled_df.to_csv ('adult_train_processed.csv', index = None, header=True)
    training_label_df.to_csv ('adult_train_label.csv', index = None, header=True)

    # ------------------
    # get test data and transform
    test_df = pd.read_csv('adult.test', header=None, delimiter=r",\s+",)

    # add header
    test_df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    
    # convert income
    test_df["Income"] = test_df["Income"].map({ "<=50K": 0, ">50K": 1 })
    
    y_all = test_df["Income"].values
    print(y_all)


    test_df.drop("Income", axis=1, inplace=True,)
    
    test_df.drop("CapitalGain", axis=1, inplace=True,)
    test_df.drop("CapitalLoss", axis=1, inplace=True,)
    
    test_df.Age = test_df.Age.astype(float)
    test_df.fnlwgt = test_df.fnlwgt.astype(float)
    test_df.EducationNum = test_df.EducationNum.astype(float)
    test_df.HoursPerWeek = test_df.HoursPerWeek.astype(float)
    
    test_df = pd.get_dummies(test_df, columns=[
        "WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship",
        "Race", "Gender", "NativeCountry",
    ])

    # set test data to the same order as the transformation
    test_df = test_df[mapper.transformed_names_]

    scaled_test_df = mapper.transform(test_df)
    print(scaled_df.head())
    print(scaled_test_df.head())
   
    # add back in label
    test_label_df = pd.DataFrame(y_all, columns = ['Income'])

    scaled_test_df.to_csv('adult_test_processed.csv', index = None, header = True)
    test_label_df.to_csv('adult_test_label.csv', index = None, header = True)
  
    
    return df, mapper


if __name__ == "__main__":

    file = "adult.data"
    df, mapper  = load_adult(file)
    #pd.set_option('display.max_columns', None)
    '''
    map_fit= mapper.fit(df)
    trasnformed_df = mapper.transform(df)
    print(df.head())
    print(trasnformed_df.head())
    '''
