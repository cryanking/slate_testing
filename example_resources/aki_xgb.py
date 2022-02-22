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

from epic_lib import get_logger
import re

# Any additional imports must either be in the Epic image or stored in pip_packages as part of your model

"""
Below is a predict (ondemand) method. It can be named whatever you specify in the definition.json.

This model code is what is included in the file uploaded to ECCP, and is what is invoked when
a request for a prediction occurs.
"""


def predict(data):
 
    logger = get_logger()
    try:
    # ordered_columns simply lists the feature names, and the type they should be cast to.
    # The type can be a string or numpy dtype.
    # In this example, they're hardcoded, but could easily be read from disk, or derived.
    #ordered_columns = [("Feature1", "int"), ("Feature2", "float")]
      colnames = pd.read_csv(os.path.join(os.getcwd(), "resources", 'colname_data.csv' ),low_memory=False )
      icmconv = converters.InterconnectMissingValueConverter()
      used_cols = list(map(tuple, colnames[["EpicName"]].to_numpy()))
      ordered_columns = list(map(tuple, colnames[["EpicName","dtype"]].to_numpy()))
      if "modelInput" in data:
        data = data.get("modelInput")
    # unpack_input() separates metadata (chronicles_info) from the dataframe of features
      dataframe, chronicles_info = Parcel.unpack_input( data, ordered_columns, dict(zip(used_cols, [icmconv]*len(used_cols))))
    
    ##############################################
    ### Load other resources ###
    ##############################################
    
      xgb_model = XGBClassifier()
      xgb_model.load_model(os.path.join(os.getcwd(), "resources", "aki_xgb.xgb") )
      ct=pickle.load(open(os.path.join(os.getcwd(), "resources", 'transform.p') , "rb" ) )
      ct.named_transformers_.onehotencoder.handle_unknown = 'ignore'



    ##############################################
    ### Use WebCallouts to get additional data ###
    ##############################################


    ##############################################
    ###     Make your predictions here         ###
    ##############################################

    
      bad_holder = dataframe['AnesthesiaType'] == '0'
      dataframe = dataframe[dataframe['AnesthesiaType'] != '0' ]
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
      dataframe.replace(to_replace={'AnesthesiaType': {'2':'1', '4':'2', '6':'5', '7':'4', '8':'6', '9':'2', '10':'2', '11':'2', '12':'1', '5':'1', '999':'other'} },  inplace=True)
      dataframe['AnesthesiaType'] = dataframe['AnesthesiaType'].fillna('Other')

    ## this is some lumping that happened due to small categories
      dataframe.replace(to_replace={'AnesthesiaType': {'5':'Other', '6':'Other'} },  inplace=True)

    ## TODO: go back to MV and create a "spine" cat
      dataframe.replace(to_replace={'Service': {'nan':'NA'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'480':'250'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'20':'10'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'30':'40'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'60':'10'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'70':'250'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'80':'10'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'90':'255'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'130':'NA'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'140':'200'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'150':'340'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'160':'255'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'170':'10'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'180':'255'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'190':'666'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'280':'NA'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'350':'250'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'395':'10'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'490':'666'} },  inplace=True)
      dataframe.replace(to_replace={'Service': {'':'NA'} },  inplace=True)

    
      data_train2 = ct.transform(dataframe)

      raw_predictions = xgb_model.predict_proba(data_train2)[:,1]*100.

      shap_values = xgb_model.get_booster().predict(DMatrix(data_train2), pred_contribs=True).T
      
      ## find the OHE encoded groups
      myre = re.compile("__x\\d+_" )
      vargroups = list(re.search(myre , val ).group(0) for val in ct.get_feature_names() if  re.search(myre, val) )
      vargroups = list(set(vargroups ))
    ## collapse those groups

      for matchname in vargroups :
        localset = list(i for i,val in enumerate(ct.get_feature_names() ) if re.search(matchname, val)) 
        if len(localset) > 1:
          shap_values[ localset[0] ] = shap_values[ localset ].sum(axis=0)
          shap_values=np.delete(shap_values, localset[1:], axis=0)

## shap output from xgboost is on log-odds scale
      def expit(x):
        return(np.exp(x)/(1+np.exp(x)))
      
      def logit(x):
        return(np.log(x) - np.log(1-x))
      
      def convert_shap_to_prob_margin(shap, prob):
        return(prob - expit(logit(prob)-shap) )
      
      shap_values = convert_shap_to_prob_margin(shap_values, raw_predictions/100. )*100.
        
   
      feature_contributions = dict()
      for i, thisname in enumerate(colnames[["EpicName"]].to_numpy()):
          feature_contributions[thisname[0]] = {"Contributions":shap_values[i].tolist() }


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
      added_features = {
          "NewFeature": {
              "Values": ["John Doe"]
          }
      }
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
              "AKI": [str(probability) for probability in raw_predictions],
              "OtherMetaData": [str(feat1_contribution*.02 ) for feat1_contribution in feature_contributions[colnames["EpicName"][0]]["Contributions"]]
          }
      }


      return Parcel.pack_output(
          mapped_predictions=formatted_predictions,  # Dictionary with display names corresponding to predictions per sample
          score_displayed="AKI",
          # The output key you'd like end users to look at (e.g. the positive class name)
          chronicles_info=chronicles_info,  # Metadata that should be passed through
          # This optional parameter can be configured to be displayed in hover bubbles, etc.
          feature_contributions=feature_contributions
          #, additional_features=added_features
      )

    except ValueError as error:
       return(f"raising an exception and the error was {error}.")
# log.exception(f"raising an exception and the error was {error}.")
