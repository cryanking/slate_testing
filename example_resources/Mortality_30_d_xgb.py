# --------------------------------------------------
# © 2018 - 2020 Epic Systems Corporation.
# Chronicles® is a registered trademark of Epic Systems Corporation.
# ---------------------------------------------------

import os
from xml.etree import ElementTree

from xgboost import XGBClassifier
from xgboost import DMatrix

import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

import pandas as pd
import numpy as np

# Parcel is an Epic released packaging/formatting helper
from parcel import Parcel
from parcel import converters

from epic_lib.cloud import (
    SecretProvider,
    WebCallout
)

# Any additional imports must either be in the Epic image or stored in pip_packages as part of your model

"""
Below is a predict (ondemand) method. It can be named whatever you specify in the definition.json.

This model code is what is included in the file uploaded to ECCP, and is what is invoked when
a request for a prediction occurs.
"""


def predict(data):
 

    # ordered_columns simply lists the feature names, and the type they should be cast to.
    # The type can be a string or numpy dtype.
    # In this example, they're hardcoded, but could easily be read from disk, or derived.
    #ordered_columns = [("Feature1", "int"), ("Feature2", "float")]
    colnames = pd.read_csv(os.path.join(os.getcwd(), "resources", 'colname_data.csv' ),low_memory=False )
    icmconv = converters.InterconnectMissingValueConverter()
    used_cols = list(map(tuple, colnames[["EpicName"]].to_numpy()))
    ordered_columns = list(map(tuple, colnames[["EpicName","dtype"]].to_numpy()))

    # unpack_input() separates metadata (chronicles_info) from the dataframe of features
    dataframe, chronicles_info = Parcel.unpack_input( data, ordered_columns, dict(zip(used_cols, [icmconv]*len(used_cols))))
    
    ##############################################
    ### Load other resources ###
    ##############################################
    
    xgb_model = XGBClassifier()
    xgb_model.load_model(os.path.join(os.getcwd(), "resources", "Mortality_30d_xgb.xgb") )
    ct=pickle.load(open(os.path.join(os.getcwd(), "resources", 'transform.p') , "rb" ) )


    ##############################################
    ### Use WebCallouts to get additional data ###
    ##############################################


    ##############################################
    ###     Make your predictions here         ###
    ##############################################

    
    bad_holder = dataframe['Anest Type'] == '0'
    dataframe = dataframe[dataframe['Anest Type'] != '0' ]
    dataframe.replace(to_replace=-1, value=np.nan, inplace=True)

    ## transform some special missing data
    dataframe.replace(to_replace="-1", value=np.nan, inplace=True)

    ## special missing value (unlike most categoricals this is too uncommon to leave as-is)
    dataframe['Sex'] = pd.to_numeric(dataframe['Sex'], errors="coerce") 
    dataframe.replace(to_replace={'Sex':3}, value=np.nan, inplace=True)
    dataframe['Sex'] -=1

    ## transform lvef to quasinumeric
    dataframe.replace(to_replace={'LVEF': {1:.08, 2:0.15, 3:0.25, 4:0.35, 5:0.45, 6:0.55, 7:0.65, 8:0.7, 101:0.6, 102:0.4, 103:0.33, 104:0.2, 999:np.nan} },  inplace=True)

    ## transform diastolic to quasinumeric
    dataframe.replace(to_replace={'Diastolic': {888:2, 999:np.nan} },  inplace=True)

    dataframe.replace(to_replace={'METs': {4:1, 3:2 , 2:3, 1:4, 888:5, 889:5, 999:np.nan} },  inplace=True)

    ## TODO: load this map fro a csv instead of code
    dataframe.replace(to_replace={'Anest Type': {'2':'1', '4':'2', '6':'5', '7':'4', '8':'6', '9':'2', '10':'2', '11':'2', '12':'1', '5':'1', '999':'other'} },  inplace=True)
    dataframe['Anest Type'] = dataframe['Anest Type'].fillna('Other')

    ## this is some lumping that happened due to small categories
    dataframe.replace(to_replace={'Anest Type': {'5':'Other', '6':'Other'} },  inplace=True)

    ## TODO: go back to MV and create a "spine" cat
    dataframe.replace(to_replace={'Service': {'480':'250'} },  inplace=True)

    
    data_train2 = ct.transform(dataframe)

    raw_predictions = xgb_model.predict_proba(data_train2)[:,1]

    shap_values = xgb_model.get_booster().predict(DMatrix(data_train2), pred_contribs=True)

    
    feature_contributions = dict()
    for i, thisname in enumerate(colnames[["EpicName"]].to_numpy()):
        feature_contributions[thisname[0]] = {"Contributions":shap_values.T[i].tolist() }


    # This optional parameter to pack_output() can be configured to be displayed in hover bubbles, etc.
    #feature_contributions = {
        ## each entry of these lists corresponds to the contribution of that feature to
        ## each sample/patient (e.g. 0.2 is the contribution of Feature1 to the second sample)
        #"Feature1": {
            #"Contributions": [0.34, 0.2]
        #},
        #"Feature2": {
            #"Contributions": [0.56, 0.8]
        #},
        #"NewFeature": {
            #"Contributions": [0.10, 0.0]
        #}
    #}

    # This optional parameter to pack_output() allows you to include additional features only calculated in python
    #added_features = {
        #"NewFeature": {
            #"Values": [patient_name, "John Doe"]
        #}
    #}
    # Though multiple keys are allowed within the "Outputs" node, the one that you'd like to
    # display (e.g. the score, or probability of the positive class) is specified by the
    # the score_displayed parameter in Parcel.pack_output().
    #
    # Note that the key specified in score_displayed is what end users will read.
    formatted_predictions = {
        "Classification1":  # this level corresponds to the Classification1 in the return schema shown above
        {
            # This ModelScore_DisplayName corresponds to
            # RegressionOutputOrClass1Probabilities in the return schema above
            "Death_in_30d": [str(probability) for probability in raw_predictions],
            "OtherMetaData": [str(feat1_contribution > 0.2) for feat1_contribution in feature_contributions[colnames["EpicName"][0]]["Contributions"]]
        }
    }


    return Parcel.pack_output(
        mapped_predictions=formatted_predictions,  # Dictionary with display names corresponding to predictions per sample
        score_displayed="Death_in_30d",
        # The output key you'd like end users to look at (e.g. the positive class name)
        chronicles_info=chronicles_info,  # Metadata that should be passed through
        # This optional parameter can be configured to be displayed in hover bubbles, etc.
        feature_contributions=feature_contributions,
        #additional_features=added_features
    )
