## this file contains transformation functions which map epic data from the back-end representation to the processed representation from clarity.
## TODO: check column order vs ST, getting wacky results
## NOTE: requires a more modern xgb than default, (tested with 1.7.5)

import os
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import make_column_transformer
#from sklearn.compose import make_column_selector

import pandas as pd
import numpy as np
import json
import math
from itertools import repeat

from xgboost import XGBClassifier
from xgboost import DMatrix

import re
import ast

## for testing only
if False:
# if True:
  data = json.load(open("resources/ondemand.json"))
  import sys
  sys.path.append('/home/eccp/epic_extract')

from parcel import Parcel
from parcel import converters
from epic_lib import get_logger

## A function to strip out certain ID numbers, room numbers, punctuation, doctor names, and expand the most common abbreviations
def text_fixed_trans(text):
    if text is None:
      return None
    if isinstance(text, pd.Series):
      text = text.str.lower() 
      text = text.str.replace(r"[^a-z0-9]", " " ,regex=True) ## non word characters
      text = text.str.replace(r"\bunos\b", " ",regex=True) ## transplant id
      text = text.str.replace(r"\ba[a-z]{2,3}[0-9]{3,4}\b", " ",regex=True) ## transplant id
      text = text.str.replace(r"\bmoma", " ",regex=True) ## transplant id
      text = text.str.replace(r"\broom\s?\d+", " ",regex=True) ## room number
      text = text.str.replace(r'\b\d+\b', ' ',regex=True) ## isolated number
      text = text.str.replace(r'\bleft', ' ',regex=True) ## side is almost always unimportant
      text = text.str.replace(r'\bright', ' ',regex=True) ## side is almost always unimportant
      text = text.str.replace(r'\bpost\b', ' ',regex=True) ## almost always unimportant
      text = text.str.replace(r'\bhours', ' ',regex=True) ## time projection is hard to embed 
      text = text.str.replace(r'\bintra', ' ',regex=True) ## this phrase is ususally superflous
      text = text.str.replace(r'\bdr\s[a-z]+', "",regex=False) ## dr names
      text = text.str.replace(r"\bc\d+", "cervical spine ",regex=True) ## spine segment abrev
      text = text.str.replace(r"\bt\d+", "thoracic spine ",regex=True) ## spine segment abrev
      text = text.str.replace(r"\bl\d+", "lumbar spine ",regex=True) ## spine segment abrev
      text = text.str.replace(r'\b[a-z0-9]{1,2}\b', '', regex=True) ## 2 letter terms 
      text = text.str.replace(r'\W+', ' ', regex=True) ## collapse multiple spaces
    else:      
      text = text.lower() 
      # remove puntuation
      text = re.sub(r"[^a-z0-9]", ' ', text)
      # remove special chars
      text = re.sub(r'\b\d+\b', ' ', text)
      text = re.sub(r'\bleft', ' ', text)
      text = re.sub(r'\bright', ' ', text)
      text = re.sub(r'\bpost\b', ' ', text)
      text = re.sub(r'\bhours', ' ', text)
      text = re.sub(r'\bintra', ' ', text)
      text = re.sub(r'\W', ' ', text)
      #remove Dr. name
      text = text.replace("dr.", "")
      text = re.sub(r"\bc\d+", "cervical spine ", text)
      text = re.sub(r"\bt\d+", "thoracic spine ", text)
      text = re.sub(r"\bl\d+", "lumbar spine ", text)
      text = re.sub(r"\bunos", " ", text)
      text = re.sub(r"\bmoma", " ", text)
      text = re.sub(r"\broom", " ", text)
    return text


def read_python_list_from_file(file_name):
  with open(file_name, "r") as f:
    list_string = f.read()
  list_string = list_string.strip()
  list_of_python_objects = ast.literal_eval(list_string)
  return list_of_python_objects

## apply a fixed embedding on the words in the procedure, then sum them
## create a final NA column for when no text is present

def text_to_cbow(text, map_dict):
  def transfun(text):
    x = map(map_dict.get, text, repeat(map_dict.get("<unk>")))  
    return np.sum(np.stack( list(map(lambda y: np.array(y, dtype=float), x ) ) ), axis=0)
  
  embedded_proc = text_fixed_trans(text)
  embedded_proc = embedded_proc[embedded_proc !='' ].str.split().transform(transfun)
  embedded_proc_df = pd.DataFrame(embedded_proc.to_list(), index=embedded_proc.index)
  embedded_proc_df.columns = ["em_CBOW_" + str(x) for x in (embedded_proc_df.columns) ]
  bow_na = pd.Series(0,index=text.index, name="BOW_NA") ## this acts like an index holder so that dropped rows are filled in
  embedded_proc_df = embedded_proc_df.merge(bow_na, left_index=True, right_index=True, how='right')
  bow_cols = [col for col in embedded_proc_df.columns if 'BOW' in col]
  embedded_proc_df['BOW_NA'] = np.where(np.isnan(embedded_proc_df[bow_cols[0]]), 1, 0)
  embedded_proc_df.fillna(0, inplace=True)
  return(embedded_proc_df)


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
## many paf are defined to return -1 to protect against the dumb CCP behavior of nulling if any inputs are null
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

## There are a small number of missing variables in the data feed
## Fill in these blank values until they are fixed
def make_constants(x):
  if(~any(x.columns == 'case_year' ) ) :
    x["case_year"] = 2021
  if(~any(x.columns == 'HCG,URINE, POC' ) ) :
    x["HCG,URINE, POC"] = 0
  if(~any(x.columns == 'preop_vent_b' ) ) :
    x['preop_vent_b'] = False

## These were dropped from the classifier
  #x['opioids_count'] = 0
  #x['total_morphine_equivalent_dose'] = 0
  #x['preop_los'] = 0.1
  #x['delirium_history']= 0
  #x['MentalHistory_other'] = 0
  #x['MentalHistory_adhd'] = 0
  #x["PNA"] =0
  #x["delirium_history"] =0


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

#ordinal_variables = [ "TOBACCO_USE" , "AR" , "AS" , "MR" , "MS" , "TR" , "ASA" , "plannedDispo" , "LVEF" , "FunctionalCapacity" , "DyspneaFreq" , "DiastolicFunction" , "CHF_class" , "cancerStatus" , "pre_aki_status" , "NutritionRisk" , "Level_of_Consciousness" , "poorDentition" , "orientation_numeric" , "Pain Score" , "case_year" , "gait" , "EPITHELIAL CELLS, SQUAMOUS, URINE" , "HYALINE CAST" , "RED BLOOD CELLS, URINE" , "WHITE BLOOD CELLS, URINE" ]


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
"LVEF" ,
"Calcium" , 
"PO2" , 
"Pain Score" , 
"basos" , 
"eosin" , 
"immgran" , 
"ldh" , 
"lymphs" , 
"magnesium" , 
"monos" , 
"mpv" , 
"neutrophil" , 
"nrbc" , 
"pco2" , 
"pco2ven" , 
"phven" , 
"po2ven" , 
"rbc" , 
"so2" , 
"urph" , 
"urspecgrav" , 
"LVEF" ,
]


def cancerdeetst(col):
  patterns = {"metastatic cancer":4,  "current cancer":3 , "in remission":2, "s/p radiation":2, "s/p chemo":2, "skin cancer only":1  }
  if isinstance(col, pd.Series):
    col2 = pd.Series(0, index=col.index)
    for pattern in patterns.keys():
      col2[ col.str.lower().str.contains(pattern) ] = col2[ col.str.lower().str.contains(pattern) ].clip(lower=patterns[pattern])
    #col[ ~col.isin(patterns.values()) ] = "-1"
    return col2
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




## pre-processing mappings to apply to inputs
## Service is a double mapping because the variable representation changed, and it was easier than redoing it

transformation_dict = {
    "TropI": lambda x :np.where( (x>0), x*.001 ,x)
  #, "Diastolic":  lambda x: x.fillna(0) ## later dropped from the dataset
  , "Patient Sex ID": lambda x: (x==2).astype('int')
  , "NoCHF":  lambda x: x.fillna(0) ## later dropped from the dataset
  #, "NoROS":  lambda x: x.fillna(0) ## later dropped from the dataset
  , "Emergent": lambda x: x=="E"
  , "Race": lambda x: apply_dict_mapping(x, {"1":0, "2":1, "19":-1}, -1 )
  , "gait": lambda x: apply_dict_mapping(x, {"0":1, "10":2, "20":3}, np.nan )
  , "Ethnicity": lambda x: apply_dict_mapping(x, {"8":1 , "9":0, "12":-1}, -1 )
  #, "dispo": lambda x: apply_dict_mapping(x.str.lower(), {"er":1,"outpatient":1,"23 hour admit":2, "floor":2,"obs. unit":3,"icu":4} , np.nan )
  , "Service": lambda x: apply_dict_mapping( apply_dict_mapping(x.str.replace(" none", "", regex=False),  {'5' : 'Acute Critical Care Surgery' , 
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
  , "Last Tobac Use Status": lambda x: apply_dict_mapping(x.fillna(0).astype(int), {3:0, 1:2 , 4:1 , 2:0 }, 0 )
  , "tobacco_sde": lambda x: x.fillna(0).astype(int)
  , "cancerdeets" : cancerdeetst 
  , "Diastolic": lambda x: apply_dict_mapping(x, {0:0, 1:1, 2:2,3:3, 888:1, 999:1  }, np.nan )
  , "Pain Score":  only_numbers
  , "ono2_sde_new": lambda x: ~x.isin(["0", "nan"])
  , "CAM": lambda x: np.select( [pd.isnull(x), (x>=1), x==0],  ["nan", "True", "nan"], default="False" )
  , "orientation": lambda x: pd.DataFrame([
      x.str.lower().str.contains("x4") *4
      , x.str.lower().str.contains("x3")*3
      , x.str.lower().str.contains("x2")*2
      , x.str.lower().str.contains("x1")*1
      , x.str.lower().str.contains("person").astype(int) + x.str.lower().str.contains("place").astype(int) + x.str.lower().str.contains("time").astype(int)+ x.str.lower().str.contains("situation").astype(int)
    ]).max(axis=0)
    , "An Start": lambda x: x.mod(86400) /60.
    , "activeInfection" : lambda x : (x!=" none")
    , "ad8" : lambda x : np.select( [x==0, pd.isnull(x)], ["False", "nan"], "True" )
    #, "Barthel" : lambda x : (x<100).fillna(False) 
    , "DementiaCogIm" : lambda x: ~x.isin(["0", "nan"])
    , "ambulatory" : lambda x: np.select( [x==0, pd.isnull(x)], ["False", "nan"], "True" )
    , "fall" : lambda x: np.select( [x==0, pd.isnull(x)], ["False", "nan"], "True" )
    , "Mental Status" : lambda x: np.select( [x=="0", x=="nan"], ["False", "nan"], "True" )
    , "DyspneaF" : lambda x : np.select( [
        x.str.lower().str.contains("never"), 
        x.str.lower().str.contains("or less") , 
        x.str.lower().str.contains("not dail"),
        x.str.lower().str.contains("daily"),
        x.str.lower().str.contains("throughout") ] , [
        np.broadcast_to(0, x.shape), 
        np.broadcast_to(1, x.shape), 
        np.broadcast_to(2, x.shape), 
        np.broadcast_to(3, x.shape), 
        np.broadcast_to(4, x.shape) ] )  #
    #, "pastDialysis" : lambda x : (x.str.lower()=='past dialysis') # applied in hyperspace
    #, "LVEF": lambda x : apply_dict_mapping(x , {-1:0.6, 1:.08, 2:0.15, 3:0.25, 4:0.35, 5:0.45, 6:0.55, 7:0.65, 8:0.7, 101:0.6, 102:0.4, 103:0.33, 104:0.2, 999:np.nan} )
    , "Resp. Support" : lambda x : ~x.isin(["NASAL CANNULA strip", "nan", " strip", ""])
    , 'MEWS LOC Score' : lambda x: x==0 # the raw LOC has a lot more subtle values, but all bad, and they mapped higher = worse whereas i mapped 1 = normal
    , "dispo":  lambda x: apply_dict_mapping(x.str.replace(" none", "", regex=False), {"OUTPATIENT":0, '23 HOUR ADMIT':1, "FLOOR":1, "OBS. UNIT":2 , "ICU":3, "ER":0}, np.nan )
    , "epiur": lambda x: x.str.replace("\s","", regex=True)
    , "Blood Type": lambda x: x.str.replace(" strip","", regex=False)
    ,"dentition": lambda x:np.select( [
        x.str.lower().str.contains("loose") |
        x.str.lower().str.contains("poor") |
        x.str.lower().str.contains("missing") |
        x.str.lower().str.contains("chipped") ,
        x.str.lower().str.contains("edentulous") | 
        x.str.lower().str.contains("partials") | 
        x.str.lower().str.contains("dentures"),
         ] , [
        2, 1 ] )

}



## some inputs that turn into multiple variables.
## i could have combined this with the set transforms below

#general: 1,2
#block: 6,8,4,5,9,10,11,12

def genanest(col):
  if isinstance(col, pd.Series):
    #plannedAnesthesia = col.str.lower().str.contains("general").fillna(False).astype(int)
    #hasBlock = col.str.lower().str.contains("|".join(["regional", "shot", "block", "epidural"])).fillna(False)
    plannedAnesthesia = col.isin(["1","2"])
    hasBlock = col.isin(["6","8","4","5","9","10","11","12"])
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
    MentalHistory_anxiety= col.str.lower().str.contains("anxiety").fillna(False)
    MentalHistory_bipolar= col.str.lower().str.contains("bipol").fillna(False)
    MentalHistory_depression= col.str.lower().str.contains("depr").fillna(False)
    MentalHistory_schizophrenia= col.str.lower().str.contains("schiz").fillna(False)
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
  "AnesthesiaType": genanest
  , "mentalhx" : mentaltrans
}

## transformations that operate on sets of features

set_trans_array = (
  [["DVT", "PE"], lambda data: pd.DataFrame((data["DVT"] + data["PE"]) >0 ).fillna(False).rename(columns={0:"DVT_PE"})  ]
  , [["Coombs_Lab", "Coombs_SDE"] , lambda data: pd.DataFrame((data["Coombs_Lab"].str.lower().str.contains("positive")) | (data["Coombs_SDE"] == "1") ).fillna(False).rename(columns={0:"Coombs"})  ]
  , [["emergency_b" , "Emergent" ], lambda data: pd.DataFrame((data["emergency_b"] > 0) | data["Emergent"]).fillna(False).rename(columns={0:"emergency"})  ] 
  , [["tobacco_sde" , "Last Tobac Use Status" ], lambda data: pd.DataFrame( data["Last Tobac Use Status"].clip(lower=data["tobacco_sde"] *2).fillna(0).rename("TOBACCO_USE")).rename(columns={0:"TOBACCO_USE"} )  ] 
  #, [["tobacco_sde" , "Last Tobac Use Status" ], lambda data: pd.DataFrame( (data["tobacco_sde"] >0) | (data["Last Tobac Use Status"] >1) ).fillna(False).rename(columns={0:"TOBACCO_USE"})  ] 
  , [["NoCHF" , "CHF" ], lambda data: pd.DataFrame( (data["CHF"] > 0) & (data["NoCHF"]>0)  ).fillna(False).rename(columns={ 0:"CHF"})  ]
  , [["pastDialysis" , "Dialysis" ], lambda data: pd.DataFrame( (data["Dialysis"] > 0) & (data["pastDialysis"]>0)  ).fillna(False).rename(columns={ 0:"Dialysis"})  ]
  , [["ono2_sde_new" , "ono2" , "Resp. Support"], lambda data: pd.DataFrame( (data["Resp. Support"] ) | (data["ono2"].astype(float) > 0) | (data["ono2_sde_new"]).fillna(False) ).rename(columns={0:"on_o2"})  ] 
  )




## fix up some text values that can be numbers
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
    AW_labs.loc[AW_labs.str.contains('cancel', na = False)] = 'NaN'
    AW_labs.loc[AW_labs.str.contains(r'\w{3,}\snegative', na = False, regex=True)] = "0"
    AW_labs.loc[AW_labs.str.contains(r'\w{3,}\spositive', na = False, regex=True)] = "1"
    AW_labs.loc[AW_labs.str.contains(r'negative\s\w{3,}', na = False, regex=True)] = "0"
    AW_labs.loc[AW_labs.str.contains(r'positive\s\w{3,}', na = False, regex=True)] = "1"
    AW_labs.loc[AW_labs.str.contains(r'nonreactive', na = False, regex=True)] = "0"
    AW_labs.loc[AW_labs.eq("nan")]= "nan"
    conditions = [
        AW_labs.eq("negative"),
        AW_labs.eq("trace"),
        AW_labs.eq("rare"),
        AW_labs.eq("positive"),
        AW_labs.eq('presumptive'),
        AW_labs.eq("presumptive positive"),
        AW_labs.eq("detected"),
        AW_labs.eq("reactive"),
        AW_labs.eq("repeatedly reactive"),
        AW_labs.isin(["1+", "2+", "3+", "4+"]),
        AW_labs.isin(["small", "moderate"]),
        AW_labs.eq('large'),
        AW_labs.eq('present')
    ]
    choices = ['0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1' , '1']
    AW_labs = np.select(conditions, choices, default=AW_labs)
  return AW_labs


## this applies the mappings to get to the processed preops as they are in the depository
def do_maps(raw_data,name_map, lab_trans):
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
  ## NOTE: note that this preserves nan! The source data has na's, and a consistent treatment has to match whatever the other did
  for target in lab_trans.keys():
    if target in raw_data.columns:
      raw_data[target] = pd.Series(lab_processing(raw_data[target])).map(lab_trans[target]).fillna(0).astype(int)
  make_constants(raw_data)
  return raw_data

## this applies the transformations unique to the model setup
def preprocess_inference(preops, metadata):
    preops.reset_index(drop=True, inplace=True)
    # reassigning to make sure that the train and test data have the same type of variables
    #continuous_variables = metadata['norm_value_cont']['cont_names']
    #ordinal_variables = metadata['norm_value_ord']['ord_names']
    encoded_variables = metadata["encoded_var"] 
    binary_variables = metadata["binary_var_name"] 
    categorical_variables = metadata["categorical_name"] 
    ordinal_variables = metadata["ordinal_variables"] 
    continuous_variables = metadata["continuous_variables"] 
    
    preops_ohe = preops.copy()[ set(binary_variables + categorical_variables + ordinal_variables + continuous_variables) ]
    
    for i in binary_variables:
      preops_ohe[i].fillna(0, inplace=True)
    
    # this is kind of hardcoded; check your data beforehand for this; fixed this
    # this is done because there were two values for missing token (nan and -inf)
    # NOTE: try the isfinite function defined above
    # this section creates NaNs only to be filled in later. it harmonizes the different kinds of not-a-number representations
    temp_list = [i for i in preops_ohe['PlannedAnesthesia'].unique() if np.isnan(i)] + [i for i in preops_ohe['PlannedAnesthesia'].unique() if math.isinf(i)]
    if temp_list !=[]:
        preops_ohe['PlannedAnesthesia'].replace(temp_list, np.NaN, inplace=True)  
    
    if 'plannedDispo' in preops_ohe.columns:
        preops_ohe['plannedDispo'].replace('', np.NaN, inplace=True)
        
    for name in categorical_variables:
      preops_ohe[name] = preops_ohe[name].astype('category')
    for a in preops_ohe.columns:
        if preops_ohe[a].dtype == 'bool':
            preops_ohe[a] = preops_ohe[a].astype('int32')
        if preops_ohe[a].dtype == 'int32':
            if (a in categorical_variables) and (a not in ordinal_variables):
                preops_ohe[a] = pd.Series(pd.Categorical( preops_ohe[a], categories= metadata['levels'][a] , ordered=False) )
        
    # one hot encoding
    # this is reverse from how I would have thought about it. It starts with the list of target columns, gets the value associated with that name, then scans for values matching the target
    # i probably would have used pd.get_dummies, concat, drop cols not present in the original, add constant 0 cols that are missing. I think this works as-is
    encoded_var = metadata['encoded_var']
    for ev in encoded_var:
        preops_ohe[ev] = 0
        ev_name = ev.rsplit("_", 1)[0]
        ev_value = ev.rsplit("_", 1)[1]
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
   # if True:
      with open(os.path.join(os.getcwd(), "resources", 'factor_encoding.json') ) as f:
        lab_trans = json.load(f)
      with open(os.path.join(os.getcwd(), "resources", 'preops_metadataicu.json') ) as f:
        preops_meta = json.load(f)
      with open(os.path.join(os.getcwd(), "resources", 'filtered_cbow.json') ) as f:
        cbow_map = json.load(f)
      ## map from clarity names to epic names
      with open(os.path.join(os.getcwd(), "resources", 'rwb_map.csv') ) as f:
        colnames = pd.read_csv(f,low_memory=False)
      # This feature is a problem, drop it for now
      #lab_trans["URINE UROBILINOGEN"] = None 
      ## NOTE: this is dumb, but The column display names are not compatible with being mnemonics in nebula. This trick happens to work to run on either direct RWB exports and nebula report
      ## detect the data source
      mnemonic_name = "RWBFeature" if any(' ' in key for key in data['Data'].keys()) else "mnemonic"
      ## map to transform names to the format anticipated by the preprocessing and factor mapping
      name_map = colnames[["RWBFeature","ClarityFeature"]].set_index("RWBFeature")['ClarityFeature'].to_dict()
      ## map to transform real-time names to the offline RWB names
      name_map2 = colnames[["mnemonic","RWBFeature"]].set_index("mnemonic")['RWBFeature'].to_dict()
      
      icmconv = converters.InterconnectMissingValueConverter()
      used_cols = list(map(tuple, colnames[[mnemonic_name]].to_numpy()))
      ordered_columns = list(map(tuple, colnames[[mnemonic_name,"dtype"]].to_numpy()))
      if "modelInput" in data:
        data = data.get("modelInput")
    # unpack_input() separates metadata (chronicles_info) from the dataframe of features
      pred_data_pre, chronicles_info = Parcel.unpack_input( data, ordered_columns, dict(zip(used_cols, [icmconv]*len(used_cols))))
      ## this is just waiting for a column fix to hit production
      ## occasionally, a column is all absent on a batch, which the above function will set to NaN and float type, even if it should be a string.
      for target in ordered_columns:
        if target[0] in pred_data_pre.columns:
          if (target[1] == 'str'):
            pred_data_pre.loc[:,pred_data_pre.columns == target[0]] = pred_data_pre[target[0]].astype('str')
      ## swap the names to the front end names used to define transformations
      if mnemonic_name == "mnemonic":
        pred_data_pre.rename(columns=name_map2, inplace=True)
      ## this block re-creates the processing to get to the same format as the raw training data
      pred_data_pre = do_maps(pred_data_pre, name_map, lab_trans)
      ## split off the procedure text
      embedded_proc = text_to_cbow(pred_data_pre["procedureText"], cbow_map)
      ## these are in the old meta
      pred_data_pre.rename(columns={"DVTold":"DVT", "PEold":"PE"} , inplace=True)
      ## handle this one case until an upstream fix occurs to switch this to native str
      if 'ABORH PAT INTERP' in pred_data_pre.columns:
        pred_data_pre['ABORH PAT INTERP']=pred_data_pre['ABORH PAT INTERP'].map(lambda x: '{:.1f}'.format(float(x)) if isinstance(x, int) else x)
      ## this applies the pre-processing used by the classifier (OHE, column selection, normalization)
      ## it so happens that at this time there is only 1 element in the processed data
      preop_data = preprocess_inference(pred_data_pre, preops_meta)
      preop_data = pd.concat( [preop_data , embedded_proc] , axis=1)
      preop_data.drop(["person_integer"], inplace=True, axis=1)
      #preop_data.to_csv("proc_data.csv")
      #vnames = read_python_list_from_file(os.path.join(os.getcwd(), "resources", 'fitted_feature_names.txt'))

      
      
      
      
    
    ##############################################
    ### Load other resources ###
    ##############################################
    
     #with open(os.path.join(os.getcwd(), "resources", 'ondemand.json' ), 'r') as file:
       #test_data = json.load(file )
     #test_data['Data']['EntityId'] = None
     #test_data['Data']['PredictiveContext'] = None
     #test_data2 = pd.DataFrame.from_dict(test_data['Data'])
    
    
    
      xgb_model = XGBClassifier()
      xgb_model.load_model(os.path.join(os.getcwd(), "resources", "BestXgBoost_model_icu_wo_hm_None-None.json") )

    ##############################################
    ### Use WebCallouts to get additional data ###
    ##############################################


    ##############################################
    ###     Make your predictions here         ###
    ##############################################
      raw_predictions = xgb_model.predict_proba(preop_data)[:,1]*100.
      #raw_predictions = xgb_model.predict_proba(preop_data.loc[:,vnames])[:,1]*100.



      added_features = {
          "NewFeature": {
              "Values": ["John Doe"]
          }
      }
      formatted_predictions = {
          "Classification1":  # this level corresponds to the Classification1 in the return schema shown above
          {
            # This ModelScore_DisplayName corresponds to
            # RegressionOutputOrClass1Probabilities in the return schema above
              "ICU": [str(probability) for probability in raw_predictions],
              #"OtherMetaData": [str(feat1_contribution*.02 ) for feat1_contribution in feature_contributions[colnames["EpicName"][0]]["Contributions"]]
          }
      }


      return Parcel.pack_output(
          mapped_predictions=formatted_predictions,  # Dictionary with display names corresponding to predictions per sample
          score_displayed="ICU",
          # The output key you'd like end users to look at (e.g. the positive class name)
          chronicles_info=chronicles_info,  # Metadata that should be passed through
          # This optional parameter can be configured to be displayed in hover bubbles, etc.
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
  
