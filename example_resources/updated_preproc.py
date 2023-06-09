## this file contains transformation functions which map epic data from the back-end representation to the processed representation from clarity.

## 2c. import list of cols actually used, or add a drop column to mapping (which is almost the same as equality of names)
## 2d. there are category columns ... need to somehow import the identity /ActFastData/Epic_TS_Prototyping/preops_metadata.json after re-running sandhya's preproc -> also solves above
## 3. numeric coerce based on above
## 4. Import easiest  procedure model



import os
import pickle
import sys
## for testing in slate only
#sys.path.append('/home/eccp/epic_extract')
from parcel import Parcel
from parcel import converters

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

from itertools import filterfalse

import pandas as pd
import numpy as np
import json
import math


import re

## for testing only
if True:
  data = json.load(open("resources/ondemand.json"))
  
  


## NOTE: this function does not use in-place modification. It wasn't performance critical. For larger datasets, that might be necessary
def normalization(data0, mode, normalizing_value, contin_var):
    data = data0.copy()
    if mode == 'mean_std':
        mean = normalizing_value['mean']
        std = normalizing_value['std']
        data[contin_var] = data[contin_var] - mean
        data[contin_var] = data[contin_var] / std
    if mode == 'min_max':
        min_v = normalizing_value['min']
        max_v = normalizing_value['max']
        data[contin_var] = data[contin_var] - min_v
        data[contin_var] = data[contin_var] / max_v
    return data

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

def make_constants(x):
  x["case_year"] = 2021
  x["HCG,URINE, POC"] = 0
  x['URINE UROBILINOGEN'] = 1.0
  x['opioids_count'] = 0
  x['total_morphine_equivalent_dose'] = 0
  x['preop_los'] = 0.1
  x['delirium_history']= 0
  x['MentalHistory_other'] = 0
  x['MentalHistory_adhd'] = 0
  x['MRN_encoded'] = 0
  x["PNA"] =0
  x["delirium_history"] =0
  x["StartTime"] =0
  

def apply_dict_mapping(x, mapping, default=np.nan):
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
, "sbt"
, "Diastolic"
,"fall"
, "NoCHF"
, "NoROS"
,"NutritionRisk"
)

## suspect this is broke
def cancerdeetst(col):
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
  else: 
    for pattern in patterns.keys():
      if re.search(pattern, col):
        return int(patterns[pattern])
    return -1





## Service is a double mapping because the variable representation changed, and it was easier than redoing it

transformation_dict = {
    "TropI": lambda x :np.where( (x>0), x*.001 ,x)
  , "Emergent": lambda x: x=="E"
  , "Race": lambda x: apply_dict_mapping(x, {"1":0, "2":1, "19":-1}, -1 )
  , "gait": lambda x: apply_dict_mapping(x, {"0":1, "10":2, "20":3}, -1 )
  , "Ethnicity": lambda x: apply_dict_mapping(x, {8:1 , 9:0, 12:-1}, -1 )
  #, "dispo": lambda x: apply_dict_mapping(x.str.lower(), {"er":1,"outpatient":1,"23 hour admit":2, "floor":2,"obs. unit":3,"icu":4} , np.nan )
  , "Service": lambda x: apply_dict_mapping( apply_dict_mapping(x,  {'5' : 'Acute Critical Care Surgery' , 
      '10' : 'Anesthesiology' , 
      '40' : 'Cardiothoracic' , 
      '50' : 'Cardiovascular' , 
      '55' : 'Colorectal' , 
      '100' : 'Gastroenterology' , 
      '110' : 'General' , 
      '120' : 'General Surgery' , 
      '165' : 'Hepatobiliary' , 
      '185' : 'Minimally Invasive Surgery' , 
      '187' : 'Minor Procedures' , 
      '200' : 'Obstetrics / Gynecology' , 
      '230' : 'Ophthalmology' , 
      '230' : 'Ophthalmology' , 
      '250' : 'Orthopaedics' , 
      '255' : 'Otolaryngology' , 
      '340' : 'Plastics' , 
      '360' : 'Pulmonary' , 
      '390' : 'Transplant' , 
      '400' : 'Trauma' , 
      '410' : 'Urology' , 
      '420' : 'Vascular' , 
      '440' : 'Dental' , 
      '450' : 'Pain Management' , 
      '666' : 'Neurosurgery' , 
      'NaN' : 'Unknown' , 
      '480' : 'Orthopaedic Spine' ,
      '190' : 'Plastics' }, 'Unknown' ) , { 
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
  , "tobacco_sde": lambda x: x.fillna(0)
  , "cancerdeets" : cancerdeetst 
  , "Diastolic": lambda x: apply_dict_mapping(x, {0:0, 1:1, 2:2,3:3, 888:1, 999:1  }, np.nan )
  , "Pain Score":  only_numbers
  , "ono2_sde_new": lambda x: x.isin(["0", "nan"])
  , "CAM": lambda x: np.where( (x>=1), 1, np.where(x==0, np.nan, 0) )
  , "orientation": lambda x: pd.DataFrame([
      x.str.lower().str.contains("x4") *4
      , x.str.lower().str.contains("x3")*3
      , x.str.lower().str.contains("x2")*2
      , x.str.lower().str.contains("x1")*1
      , x.str.lower().str.contains("person").astype(int) + x.str.lower().str.contains("place").astype(int) + x.str.lower().str.contains("time").astype(int)+ x.str.lower().str.contains("situation").astype(int)
    ]).max(axis=0)
    , "An Start": lambda x: x/60.
    , "activeInfection" : lambda x : ~pd.isnull(x)
    , "ad8" : lambda x :  (x>0).fillna(False) 
    , "Barthel" : lambda x : (x<100).fillna(False) 
    , "DementiaCogIm" : lambda x: x.isin(["0", "nan"])
    , "fall" : lambda x : (x>0).fillna(False)
    #, "Mental Status" : lambda x : (x>0).fillna(False)
    , "DyspneaF" : lambda x : x=="NEVER"
    , "pastDialysis" : lambda x : ~pd.isnull(x)
    , "LVEF": lambda x : apply_dict_mapping(x , {-1:0.6, 1:.08, 2:0.15, 3:0.25, 4:0.35, 5:0.45, 6:0.55, 7:0.65, 8:0.7, 101:0.6, 102:0.4, 103:0.33, 104:0.2, 999:np.nan} )
    , "Resp. Support" : lambda x : ~x.isin(["NASAL CANNULA", np.nan])
}

# suspect incorrect data feed
# fev1percent -> does not exist (added), magnesium -> added
# Coombs_Lab urnitr colorur covidrna hepb hepc hivlab  urmucus claritryur -> new extension
# wheezing -> all zero (it's just rare)
# icu_request -> added
# gait has a factor map? -> yes, transformed to ints, which was probably a mistake 
# bactur hyalineur urleuk glucoseur epiur urketone urprot -> fixed (change to string) ; this is not exactly the same (it converts thing leading with numbers to that number and otherwise to 0)
# 

def genanest(col):
  if isinstance(col, pd.Series):
    plannedAnesthesia = col.str.lower().str.contains("general")
    hasBlock = col.str.lower().str.contains("|".join(["regional", "shot", "block", "epidural"]))
    return pd.DataFrame({'PlannedAnesthesia': plannedAnesthesia, 'hasBlock': hasBlock}) 
  elif isinstance(col, np.ndarray):
    plannedAnesthesia = np.char.contains(col, "general")
    hasBlock = np.char.contains(col, "|".join(["regional", "shot", "block", "epidural"]))
  else :
    plannedAnesthesia = re.search("general", col)
    hasBlock = any(re.search(s, col) for s in ["regional", "shot", "block", "epidural"])
  return [plannedAnesthesia , hasBlock ]  

def mentaltrans(col):
  if isinstance(col, pd.Series):
    MentalHistory_anxiety= col.str.lower().str.contains("anxiety")
    MentalHistory_bipolar= col.str.lower().str.contains("bipol")
    MentalHistory_depression= col.str.lower().str.contains("depr")
    MentalHistory_schizophrenia= col.str.lower().str.contains("schiz")
    return pd.DataFrame({'MentalHistory_anxiety': MentalHistory_anxiety, 'MentalHistory_bipolar': MentalHistory_bipolar , "MentalHistory_depression":MentalHistory_depression, "MentalHistory_schizophrenia":MentalHistory_schizophrenia })
  elif isinstance(col, np.ndarray):
    MentalHistory_anxiety= np.char.contains(col, "anxiety")
    MentalHistory_bipolar= np.char.contains(col, "bipol")
    MentalHistory_depression= np.char.contains(col, "depr")
    MentalHistory_schizophrenia= np.char.contains(col, "schiz")
  else :
    MentalHistory_anxiety = re.search("anxiety", col)
    MentalHistory_bipolar = re.search("bipol", col)
    MentalHistory_depression = re.search("depr", col)
    MentalHistory_schizophrenia = re.search("schiz", col)
  return[ MentalHistory_anxiety,MentalHistory_bipolar, MentalHistory_depression, MentalHistory_schizophrenia ]

multi_trans_dict = {
  "AN Type": genanest
  , "mentalhx" : mentaltrans
}

set_trans_array = (
  [["DVT", "PE"], lambda data: pd.DataFrame((data["DVT"] + data["PE"]) >0 ).rename(columns={0:"DVT_PE"})  ]
  , [["Coombs_Lab", "Coombs_SDE"] , lambda data: pd.DataFrame((data["Coombs_Lab"].isin(["Positive"])) | (data["Coombs_SDE"] == "1") ).rename(columns={0:"Coombs"})  ]
  , [["emergency_b" , "Emergent" ], lambda data: pd.DataFrame((data["emergency_b"] > 0) | data["Emergent"]).rename(columns={0:"emergency"})  ] 
  , [["tobacco_sde" , "Last Tobac Use Status" ], lambda data: pd.DataFrame( (data["tobacco_sde"] >0) | (data["Last Tobac Use Status"] >1) ).rename(columns={0:"TOBACCO_USE"})  ] 
  , [["ono2_sde_new" , "ono2" , "Resp. Support"], lambda data: pd.DataFrame( (data["Resp. Support"] ) | (data["ono2"].astype(float) > 0) | (data["ono2_sde_new"]) ).rename(columns={0:"on_o2"})  ] 
  )





## assumes a pandas series
## weird 0.1 0.3 etc in maps are because the orginal mapping contains values which are replaced by the below to the same value, the intermediate as.list and write_json method create these "off" keys, but since they  all map to the same value it doesn't matter
def lab_processing(AW_labs):
  if( pd.api.types.is_string_dtype(AW_labs.dtype)):
    AW_labs = AW_labs.str.lower()
    AW_labs = AW_labs.str.replace("<","")
    AW_labs = AW_labs.str.replace(">","")
    AW_labs.loc[AW_labs.str.contains('not', na = False)] = '0'
    AW_labs.loc[AW_labs.str.contains('none', na = False)] = '0'
    AW_labs.loc[AW_labs.str.contains('undetected', na = False)] = '0'
    AW_labs.loc[AW_labs.str.contains(r'\w{3,}\snegative', na = False, regex=True)] = "0"
    AW_labs.loc[AW_labs.str.contains(r'\w{3,}\spositive', na = False, regex=True)] = "1"
    AW_labs.loc[AW_labs.str.contains(r'negative\s\w{3,}', na = False, regex=True)] = "0"
    AW_labs.loc[AW_labs.str.contains(r'positive\s\w{3,}', na = False, regex=True)] = "1"
    AW_labs.loc[AW_labs.str.contains(r'nonreactive', na = False, regex=True)] = "0"
    AW_labs.loc[AW_labs.eq("nan")]= "0"
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
  return AW_labs


#subset = {key: lab_trans.get(key) for key in set(raw_data.columns).intersection( set(lab_trans.keys()) ) }

def do_maps(raw_data):
  ## fixed transformations, usually mappings
  for target in transformation_dict.keys():
    if target in raw_data.columns:
      raw_data[target] = transformation_dict[target](raw_data[target])
  ## variables with -1 as a wrapper for null
  for target in negative_1_impute:
    if target in raw_data.columns:
      raw_data[target] = replace_m1_with_nan(raw_data[target])
  ## generate expected columns
  for target in multi_trans_dict.keys():
    if target in raw_data.columns:
      raw_data = pd.concat([raw_data] + [multi_trans_dict[target](raw_data[target])] , axis=1)
  ## some transformations that use > 1 variable
  for vset, fun in set_trans_array:
    if set(vset) <= set(raw_data.columns):
      raw_data = pd.concat([raw_data.rename( columns={v:v+"old" for v in vset}), fun(raw_data[vset] )] , axis=1)
  ## remap names so that I can apply the existing lab transformations
  raw_data.rename(columns=name_map, inplace=True)
  ## TODO: note that this preserves nan! The source data has na's, and a consistent treatment has to match whatever the other did
  for target in lab_trans.keys():
    if target in raw_data.columns:
      raw_data[target] = pd.Series(lab_processing(raw_data[target])).replace( lab_trans[target] )
  make_constants(raw_data)
  return raw_data

def preprocess_inference(preops, metadata):
    preops.reset_index(drop=True, inplace=True)
    # reassigning to make sure that the train and test data have the same type of variables
    continuous_variables = metadata['norm_value_cont']['cont_names']
    ordinal_variables = metadata['norm_value_ord']['ord_names']
    encoded_variables = metadata["encoded_var"] 
    binary_variables = metadata["binary_var_name"] 
    categorical_variables = metadata["categorical_name"] 
    ordinal_variables = metadata["ordinal_variables"] 
    continuous_variables = metadata["continuous_variables"] 
    
    for i in binary_variables:
      preops[i].fillna(0, inplace=True)
    
    # this is kind of hardcoded; check your data beforehand for this; fixed this
    # this is done because there were two values for missing token (nan and -inf)
    # TODO: try the isfinite function defined above
    # this section creates NaNs only to be filled in later. it harmonizes the different kinds of not-a-number representations
    temp_list = [i for i in preops['PlannedAnesthesia'].unique() if np.isnan(i)] + [i for i in preops['PlannedAnesthesia'].unique() if math.isinf(i)]
    if temp_list !=[]:
        preops['PlannedAnesthesia'].replace(temp_list, np.NaN, inplace=True)  
    
    if 'plannedDispo' in preops.columns:
        preops['plannedDispo'].replace('', np.NaN, inplace=True)
        
    for a in preops.columns:
        if preops[a].dtype == 'bool':
            preops[a] = preops[a].astype('int32')
        if preops[a].dtype == 'int32':
            if (a in categorical_variables) and (a not in ordinal_variables):
                preops[a] = pd.Series(pd.Categorical( preops[a], categories= metadata['levels'][a] , ordered=False) )
        
    # one hot encoding
    # this is reverse from how I would have thought about it. It starts with the list of target columns, gets the value associated with that name, then scans for values matching the target
    # i probably would have used pd.get_dummies, concat, drop cols not present in the original, add constant 0 cols that are missing. I think this works as-is
    preops_ohe = preops.copy()
    encoded_var = metadata['encoded_var']
    for ev in encoded_var:
        preops_ohe[ev] = 0
        ev_name = ev.rsplit("_", 1)[0]
        ev_value = ev.rsplit("_", 1)[1]
        # print(ev, ev_name, ev_value)
        if ev_value != 'nan':
            if len(preops[ev_name].unique()) < 2:
                dtype_check = preops[ev_name].unique()[0]
            else:
                dtype_check = preops[ev_name].unique()[1]
            if type(dtype_check) == np.float64 or type(dtype_check) == np.int64:
                preops_ohe[ev] = np.where(preops_ohe[ev_name].astype('float') == float(ev_value), 1, 0)
            elif type(dtype_check) == bool:
                preops_ohe[ev] = np.where(preops[ev_name].astype('str') == ev_value, 1, 0)
            else:
                preops_ohe[ev] = np.where(preops_ohe[ev_name] == ev_value, 1, 0)
    # this for loop checks if the categorical variable doesn't have 1 in any non-NAN value column and then assigns 1 in the nan value column
    # this is done because the type of nans in python are complicated and different columns had different type of nans
    for i in categorical_variables:
        name = str(i) + "_nan"
        lst = [col for col in encoded_var if (i == col.rsplit("_", 1)[0]) and (col != name)]
        preops_ohe['temp_col'] = preops_ohe[lst].sum(axis=1)
        preops_ohe[name] = np.where(preops_ohe['temp_col'] == 1, 0, 1)
        preops_ohe.drop(columns=['temp_col'], inplace=True)
    preops_ohe.drop(columns=categorical_variables, inplace=True)
    # mean imputing and scaling the continuous variables
    preops_ohe[continuous_variables].fillna(dict(zip(metadata['norm_value_cont']['cont_names'], metadata["train_mean_cont"])), inplace= True)  ## warning about copy
    # this is done because nan that are of float type is not recognised as missing by above commands
    for i in continuous_variables:
        if preops_ohe[i].isna().any() == True:
            preops_ohe[i].replace(preops_ohe[i].unique().min(), dict(zip(metadata['norm_value_cont']['cont_names'], metadata["train_mean_cont"]))[i], inplace=True)
    preops_ohe = normalization(preops_ohe, 'mean_std', metadata['norm_value_cont'], continuous_variables)
    # median Imputing_ordinal variables
    # imputing
    for i in ordinal_variables:
        preops_ohe.loc[:,i] = pd.to_numeric(preops_ohe[i] , errors='coerce').fillna( dict(zip(metadata['norm_value_ord']['ord_names'], metadata["train_median_ord"]))[i]) 
        #replace(preops_ohe[i].unique().min(), dict(zip(metadata['norm_value_ord']['ord_names'], metadata["train_median_ord"]))[i], inplace=True) # min because nan is treated as min
    # normalization
    preops_ohe = normalization(preops_ohe, 'mean_std', metadata['norm_value_ord'] , ordinal_variables)
    preops_ohe = preops_ohe.reindex(metadata["column_all_names"], axis=1)
    return preops_ohe



def predict(data):
 
    logger = get_logger()
    try:
    # ordered_columns simply lists the feature names, and the type they should be cast to.
    # this is all the "simple" features
    #ordered_columns = [("Feature1", "int"), ("Feature2", "float")]
  ## map discrete lab values to factor index
    if True:  
      with open(os.path.join(os.getcwd(), "resources", 'factor_encoding.json') ) as f:
        lab_trans = json.load(f)
      with open(os.path.join(os.getcwd(), "resources", 'preops_metadata.json') ) as f:
        preops_meta = json.load(f)
      ## map from clarity names to epic names
      with open(os.path.join(os.getcwd(), "resources", 'rwb_map.csv') ) as f:
        colnames = pd.read_csv(f,low_memory=False)
      # This feature is a problem, drop it for now
      lab_trans["URINE UROBILINOGEN"] = None
      colnames = colnames.loc[~(colnames.ClarityFeature == "URINE UROBILINOGEN")]
      name_map = colnames[["RWBFeature","ClarityFeature"]].set_index('RWBFeature')['ClarityFeature'].to_dict()
      #colnames = pd.read_csv(os.path.join(os.getcwd(), "resources", 'rwb_map.csv' ),low_memory=False )
      icmconv = converters.InterconnectMissingValueConverter()
      used_cols = list(map(tuple, colnames[["RWBFeature"]].to_numpy()))
      ordered_columns = list(map(tuple, colnames[["RWBFeature","dtype"]].to_numpy()))
      if "modelInput" in data:
        data = data.get("modelInput")
    # unpack_input() separates metadata (chronicles_info) from the dataframe of features
      pred_data_pre, chronicles_info = Parcel.unpack_input( data, ordered_columns, dict(zip(used_cols, [icmconv]*len(used_cols))))
      ## occasionally, a column is all absent on a batch, which the above function will set to NaN and float type, even if it should be a string.
      for target in ordered_columns:
        if target[0] in pred_data_pre.columns:
          if (target[1] == 'str'):
            pred_data_pre.loc[:,pred_data_pre.columns == target[0]] = pred_data_pre[target[0]].astype('str')
      ## this block re-creates the processing to get to the same format as the raw training data
      pred_data_pre = do_maps(pred_data_pre)
      pred_data_pre.rename(columns={"DVTold":"DVT", "PEold":"PE"} , inplace=True)
      ## this applies the pre-processing used by the classifier (OHE, column selection, normalization)
      ## it so happens that at this time there is only 1 element in the processed data
      preop_data = preprocess_inference(pred_data_pre, preops_meta)
      
      
      
      ## Drop from training: PNA
    
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

## this is currently unused (was designed for extracting event times from a text column that isn't working)
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
      else:
        return np.nan
      
    if isinstance(input_obj, pd.Series):
        temp = input_obj.str.extract(tempre, expand=False)
        return pd.to_numeric(temp.str[:2], errors='coerce') * 60 + pd.to_numeric(temp.str[2:], errors='coerce')
    elif isinstance(input_obj, np.ndarray):
        return np.vectorize(first_group)(input_obj) # surprisingly, there is no built in vectorized regex
    elif isinstance(input_obj, str):
        return first_group(input_obj)
    return None
  
