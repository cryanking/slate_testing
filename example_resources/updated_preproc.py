## this file contains transformation functions which map epic data from the back-end representation to the processed representation from clarity.

import os
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

from itertools import filterfalse

import pandas as pd
import numpy as np
import json

import gensim

import re

## map discrete lab values to factor index
with open(os.path.join(os.getcwd(), "resources", 'factor_encoding.json') ) as f:
  lab_trans = json.load(f)
  
## map from clarity names to epic names
with open(os.path.join(os.getcwd(), "resources", 'clarity_epic_map.json') ) as f:
  name_map = json.load(f)


## generated helper functions from GPT
## many paf are defined to return -1 to protect against the dumb CCP behavior of always nulling if any inputs are null
def replace_m1_with_nan(col):
    if isinstance(col, pd.Series):
        # For pandas series, use .where() method to replace -1 with np.nan
        return col.where(col != -1, np.nan)
    elif isinstance(col, np.ndarray):
        # For numpy arrays, use np.where() function to replace -1 with np.nan
        return np.where(col != -1, col, np.nan)
    else:
        # For scalar inputs, return np.nan if input is -1, otherwise return input
        return np.nan if col == -1 else col

## apply re to strip non-numbers
def only_numbers(x):
    pattern=r"[^0-9.]"
    replacement = ""
    """Substitute using a regular expression, keeping all number-type characters"""
    if isinstance(x, pd.Series):
        return x.str.replace(pattern, replacement)
    elif isinstance(x, np.ndarray):
        return np.char.replace(x, pattern, replacement)
    elif isinstance(x, str):
        return re.sub(pattern, replacement, x)
    else:
        raise ValueError("Unsupported input type")

def apply_dict_mapping(x, mapping, default=None):
    """
    Applies a fixed dictionary mapping to an input.

    Parameters:
        x (pandas.Series, numpy.ndarray, or scalar): The input to map.
        mapping (dict): The dictionary mapping to apply.
        default: (optional) The default value to use for keys not in the mapping.

    Returns:
        pandas.Series, numpy.ndarray, or scalar: The mapped output.
    """

    if isinstance(x, pd.Series):
        # If x is a Pandas Series, use the map() method to apply the mapping
        return x.map(mapping).fillna(default)

    elif isinstance(x, np.ndarray):
        # If x is a NumPy array, use the vectorized version of the mapping
        # TODO: I think there are more efficent ways to do this, but also don't care that much because I expect that I will always convert to pandas first
        mapping_func = np.vectorize(mapping.get)
        return np.where(np.isin(x, mapping), mapping_func(x), default)

    else:
        # If x is a scalar, apply the mapping directly
        return mapping.get(x, default)

def extract_group(input_obj, group=1):
    """Extracts a regular expression group from input_obj using pattern and group number.
    input_obj can be a pandas series, a numpy array, or a string.
    Returns a pandas series with the extracted group or None if no match found.
    """
    tempre = re.compile("([0-9]+): Anesthesia Start")
    def first_group (x):
      match = re.search(tempre, x)
      if match:
        return int(match.group(group)[:2])*60 + int(match.group(group)[2:])
      else
        return np.nan
      
    if isinstance(input_obj, pd.Series):
        temp = input_obj.str.extract(tempre, expand=False)
        return pd.to_numeric(temp.str[:2], errors='coerce') * 60 + pd.to_numeric(temp.str[2:], errors='coerce')
    elif isinstance(input_obj, np.ndarray):
        return np.vectorize(first_group)(input_obj) # surprisingly, there is no built in vectorized regex
    elif isinstance(input_obj, str):
        return first_group(input_obj)
    return None
  
  
## this set of variables should all have -1 replaced with NaN
negative_1_impute = [
"Height in Inches" ,
"Weight in Ounces",
"Age (Years)" ,
"ALT" ,  
"Albumin" ,    
"AlkPhos" , 
"aPTT" , 
"Bilirubin" , 
"Creatinine" , 
"Glucose" , 
"Hematocrit" , 
"INR" ,      
"Platelet count" , 
"Potassium" , 
"Sodium" ,   
"BUN" , 
"WBC" ,  
"Coombs" ,  
"a1c" ,         
"ast" ,  
"bicarb" , 
"bilidir" , 
"bnp" ,  
"chloride" ,   
"chol" ,               
"ck" ,             
"crp" ,       
"esr" ,  
"ferritin" , 
"Fibrinogen" , 
"hdl" , 
"hemoglobin" , 
"ionca" , 
"iron" , 
"ironsat" , 
"lactate" , 
"ldl" , 
"lipase" , 
"pHArt" , 
"phos" , 
"rdw" , 
"tibc" , 
"totalprot" , 
"transferrin" , 
"trig" , 
"TropI" , 
"tsh" , 
"vanc" , 
"vitd" , 
"mentalvars" , 
"ASA" , 
"Diastolic" , 
"METs" , 
"LVEF"
]

nan_to_zero = (
'hasBlock'
,'emergency'
,'coombs'
)


transformation_dict = {
  "Weight in Ounces": lambda x: x*.02835 
  , "TropI": lambda x :x*.001 
  , "Emergent": lambda x: x=="E"
  , "Race": lambda x: apply_dict_mapping(x, {1:0, 2:1, 19:-1}, -1 )
  , "Ethnicity": lambda x: apply_dict_mapping(x, {8:1 , 9:0, 12:-1}, -1 )
  , "plannedDispo": lambda x: apply_dict_mapping(x, {"ER":1,"Outpatient":1,"23 hour admit":2, "Floor":2,"Obs. unit":3,"ICU":4} , np.nan )
  , "Service": lambda x: apply_dict_mapping(x, {
      'Acute Critical Care Surgery':1,'General Surgery':1,'Trauma':1,
      'Cardiothoracic' : 2 ,
      'Colorectal' : 3 ,
      'Anesthesiology':4, 'Dental':4, 'Gastroenterology':4, 'Minor Procedures':4, 'Pain Management':4, 'Pulmonary':4,'Transplant Hepatology':4, 'Cardiovascular':4, 'Radiation Oncology':4,
      'Hepatobiliary' : 5 ,
      'Minimally Invasive Surgery' : 6 ,
      'Neurosurgery': 7, 'Neurosurgery Spine': 7 ,
      'Obstetrics / Gynecology' : 8 ,
      'Oncology' : 9 ,
      'Ophthalmology' : 10 ,
      'Orthopaedic Spine':11, 'Orthopaedics':11, 'Podiatry Foot / Ankle':11,
      'Otolaryngology':12, 'Oral / Maxillofacial':12 ,
      'Plastics' : 13 ,
      'Transplant' : 14 ,
      'Urology' : 15 ,
      'Vascular' : 16 
      } , 0 )
  , "Last Tobac Use Status": lambda x: apply_dict_mapping(x, {3:-1, 1:2 , 4:1 , 2:0 }, -1 )
  , "cancerdeets" : lambda col: 
        patterns = {"metastatic cancer":"4",  "current cancer":"3" , "in remission":"2", "s/p radiation":"2", "s/p chemo":"2", "skin cancer only":"1"  }
        if isinstance(col, pd.Series):
          for pattern in patterns.keys():
            col[ col.str.contains(pattern) ] = patterns[pattern]
          col[ ~col.isin(patterns.values()) ] = "-1"
          col = pd.to_numeric(col, errors='coerce').astype(int)
          return col
        elif isinstance(col, np.ndarray):
          for pattern in patterns.keys():
            col = np.where( np.char.contains(col, pattern), patterns[pattern], col )
          col = np.where( np.isin(col, patterns.values()), col, "-1" )
          col = col.astype(int)
          return col
        else 
          for pattern in patterns.keys():
            if re.search(pattern, col):
              return int(patterns[pattern])
          return -1
  , "Diastolic": lambda x: apply_dict_mapping(x, {0:0, 1:1, 2:2,3:3, 888:1, 999:1  }, np.nan )
  , "Pain Score":  only_numbers
  , "CAM": lambda x: 
     temp = x>=1
     temp = temp.astype(int) # to allow nan
     temp[x<=0] = np.nan
     return temp
  , "mental_status": lambda x: x > 0
  
}



multi_trans_dict = {
  "AN Type": lambda col:
    if isinstance(col, pd.Series):
      plannedAnesthesia = col.str.contains("general")
      hasBlock = col.str.contains("|".join(["regional", "shot", "block", "epidural"]))
    elif isinstance(col, np.ndarray):
      plannedAnesthesia = np.char.contains(col, "general")
      hasBlock = np.char.contains(col, "|".join(["regional", "shot", "block", "epidural"]))
    else 
      plannedAnesthesia = re.search("general", col)
      hasBlock = any(re.search(s, col) for s in ["regional", "shot", "block", "epidural"])
    return [plannedAnesthesia , hasBlock ]   
  , "mentalhx" : lambda col:
    if isinstance(col, pd.Series):
      MentalHistory_anxiety= col.str.contains("anxiety")
      MentalHistory_bipolar= col.str.contains("bipol")
      MentalHistory_depression= col.str.contains("depr")
      MentalHistory_schizophrenia= col.str.contains("schiz")
    elif isinstance(col, np.ndarray):
      MentalHistory_anxiety= np.char.contains(col, "anxiety")
      MentalHistory_bipolar= np.char.contains(col, "bipol")
      MentalHistory_depression= np.char.contains(col, "depr")
      MentalHistory_schizophrenia= np.char.contains(col, "schiz")
    else 
      MentalHistory_anxiety = re.search("anxiety", col)
      MentalHistory_bipolar = re.search("bipol", col)
      MentalHistory_depression = re.search("depr", col)
      MentalHistory_schizophrenia = re.search("schiz", col)
    return[ MentalHistory_anxiety,MentalHistory_bipolar, MentalHistory_depression, MentalHistory_schizophrenia ]
}

set_trans_array = (
  [["DVT", "PE"], lambda data: pd.DataFrame(data["DVT"] | data["PE"]).rename(columns={0:"DVT_PE"})  ]
  , [["Coombs", "Coombs_SDE"] , lambda data: pd.DataFrame(data["Coombs"] | data["Coombs_SDE"]).rename(columns={0:"Coombs"})  ]
  )

## assumes a pandas series
def lab_processing(AW_labs):
  AW_labs = AW_labs.str.lower()

  AW_labs.loc[AW_labs.str.contains('not', na = False)] = '0'
  AW_labs.loc[AW_labs.str.contains('none', na = False)] = '0'
  AW_labs.loc[AW_labs.str.contains('undetected', na = False)] = '0'
  AW_labs.loc[AW_labs.str.contains(r'\w{3,}\snegative', na = False, regex=True)] = "0"
  AW_labs.loc[AW_labs.str.contains(r'\w{3,}\spositive', na = False, regex=True)] = "1"
  AW_labs.loc[AW_labs.str.contains(r'negative\s\w{3,}', na = False, regex=True)] = "0"
  AW_labs.loc[AW_labs.str.contains(r'positive\s\w{3,}', na = False, regex=True)] = "1"

  conditions = [
      AW_labs.eq("negative"),
      AW_labs.eq("trace"),
      AW_labs.eq("positive"),
      AW_labs.eq('presumptive'),
      AW_labs.eq("presumptive positive"),
      AW_labs.eq("detected"),
      AW_labs.eq("reactive"),
      AW_labs.eq("repeatedly reactive"),
      AW_labs.isin(["1+", "2+", "3+", "4+"]),
      AW_labs.isin(["small", "moderate"]),
      AW_labs.eq('large')
  ]
  choices = ['0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1']
  AW_labs = np.select(conditions, choices, default=AW_labs)


def make_wv_hx(data)


def do_maps(data):
  ## fixed transformations, usually mappings
  for target in transformation_dict.keys():
    if target in data.columns:
      data[target] = transformation_dict[target](data[target])
  ## variables with -1 as a wrapper for null
  for target in negative_1_impute:
    if target in data.columns:
      data[target] = replace_m1_with_nan(data[target])
  ## generate expected columns
  for target in multi_trans_dict.keys():
    if target in data.columns:
      data = pd.concat([data] + lab_trans[target](data[target]) , axis=1)
  ## remap names so that I can apply the existing lab transformations
  data.rename(columns=name_map)
  ## some semi-qualitative labs
  for target in lab_list:
    if target in data.columns:
      data[target] = lab_processing(data[target])
  ## quantitative labs to their anticipated integer index
  for target in lab_trans.keys():
    if target in data.columns:
      data[target] = lab_trans[target](data[target])  
  ## some transformations that use > 1 variable
  for vset, fun in set_trans_array:
    if set(vset) <= set(data.columns):
      data = pd.concat(data.rename( columns={v:v+"old" for v in vset}), fun(data[vset] ) )
  return data

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
    
     #with open(os.path.join(os.getcwd(), "resources", 'ondemand.json' ), 'r') as file:
       #test_data = json.load(file )
     #test_data['Data']['EntityId'] = None
     #test_data['Data']['PredictiveContext'] = None
     #test_data2 = pd.DataFrame.from_dict(test_data['Data'])
    
    
    
      xgb_model = XGBClassifier()
      xgb_model.load_model(os.path.join(os.getcwd(), "resources", "Mortality_30d_xgb.xgb") )
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
      
        ## the categorical features
      feature_contributions = dict()
      for i, thisname in enumerate(ct.transformers_[0][2] ):
          feature_contributions[thisname] = {"Contributions":shap_values[i].tolist() }
      
      offset = len(ct.transformers_[0][2])
      
      ## the numeric features
      varnames = list( filterfalse( myre.search , ct.get_feature_names())) 
      
      for i, thisname in enumerate(varnames):
          feature_contributions[thisname] = {"Contributions":shap_values[i+offset].tolist() }




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
              "Death_in_30d": [str(probability) for probability in raw_predictions],
              #"OtherMetaData": [str(feat1_contribution*.02 ) for feat1_contribution in feature_contributions[colnames["EpicName"][0]]["Contributions"]]
          }
      }


      return Parcel.pack_output(
          mapped_predictions=formatted_predictions,  # Dictionary with display names corresponding to predictions per sample
          score_displayed="Death_in_30d",
          # The output key you'd like end users to look at (e.g. the positive class name)
          chronicles_info=chronicles_info,  # Metadata that should be passed through
          # This optional parameter can be configured to be displayed in hover bubbles, etc.
          feature_contributions=feature_contributions
          #, additional_features=added_features
      )

    except ValueError as error:
       return(f"raising an exception and the error was {error}.")
# log.exception(f"raising an exception and the error was {error}.")
