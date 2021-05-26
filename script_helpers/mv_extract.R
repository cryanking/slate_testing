## subsample the columns of ACT2 to match those for slate experiment
## docker run --rm -it -v '/mnt/ris/ActFastData/:/research/' -v '/home/christopherking/gitdirs/actfast_processing_container:/actfast_prep' -v "/home/christopherking/slate_test/:/pkghome/" cryanking/verse_plus /bin/bash

library(tidyverse)
library(magrittr)
library(data.table)

colsource <- "colname_data.csv"
data_pre_imputed <- FALSE

local_arg <- commandArgs(trailingOnly=TRUE)
if(length(local_arg) > 0L) {
  if( any(grepl(local_arg, pattern="^colsource=")) )
    colsource <-  local_arg %>% grep(pattern="^colsource=", value=TRUE) %>% extract2(1) %>% sub(pattern="^colsource=",replacement="") 
  if( any(grepl(local_arg, pattern="^preimpute=")) )
    data_pre_imputed <-  local_arg %>% grep(pattern="^preimpute=", value=TRUE) %>% extract2(1) %>% sub(pattern="^preimpute=",replacement="") %>% as.logical 

} 

de_i_location <- "/research/Actfast_deident/"


if(data_pre_imputed) {
  preops <- read_csv(paste0(de_i_location, "preops.csv") )
} else {
  preops <- read_csv(paste0(de_i_location, "unimputed_preops.csv") )
}

## the some versions lack CCI
if(is.null(preops[["CCI"]] )) {
  preops %<>% mutate(CCI = case_when(Age >= 80 ~4, Age>= 70 ~3, Age >=60~2, Age >= 50 ~1, TRUE ~0 ) + (CHF) + (PAD) +  (CV_TIA_STROKE) + (DEMENTIA)  + (CIRRHOSIS) + (DM) + (Outpatient_Insulin ) + (CKD)*2 + (CANCER_HX)*2 + (COPD) )
}

## transform CCI to prob of survival - this is the most common formula
preops %<>% mutate(CCI = 0.983 ^ exp(CCI*0.9 ) )

## Subset to requested variables
preops %<>% select(caseid, any_of(name_alignment$MVName) )
 
## Sex in the form epix expects
preops %<>% mutate(SEX = case_when(SEX=="1"~0L, SEX=="2"~1L, TRUE~NA_integer_ ))
 
##undo a lumping
preops %<>% mutate(CHF_Diastolic_Function = as.integer(CHF_Diastolic_Function) -1L)

# name_alignment$dtype <-  preops %>% sapply(class) %>% extract(name_alignment$MVName)
# name_alignment$dtype <- if_else(name_alignment$dtype=="numeric", "float" , "str" )


## surgicial service conversion table
epic_service_numerics <- read_csv("surgical_service_map.csv")

# epic_service_numerics <- c(
# "5" , "Acute Critical Care Surgery" , "ACCS" ,
# "10" , "Anesthesiology" , "Other" , 
# "40" , "Cardiothoracic" , "Cardiothoracic" , 
# "50" , "Cardiovascular" , "Other" , 
# "55" , "Colorectal" , "Colorectal" , 
# "100" , "Gastroenterology" , "Other" , 
# "110" , "General" , "Hepatobiliary" , 
# "120" , "General Surgery" , "Hepatobiliary" , 
# "165" , "Hepatobiliary" , "Hepatobiliary" , 
# "185" , "Minimally Invasive Surgery" , "Minimally Invasive Surgery" , 
# "187" , "Minor Procedures" , "Other" , 
# "200" , "Obstetrics / Gynecology" , "GYNECOLOGY" , 
# "230" , "Ophthalmology" , "Other" , 
# "230" , "Ophthalmology" , "Other" , 
# "250" , "Orthopaedics" , "Orthopaedic" , 
# "255" , "Otolaryngology" , "Otolaryngology" , 
# "340" , "Plastics" , "Plastic" , 
# "360" , "Pulmonary" , "Other" , 
# "390" , "Transplant" , "Transplant" , 
# "400" , "Trauma" ,  "ACCS" ,
# "410" , "Urology" , "Urology" , 
# "420" , "Vascular" , "Vascular" , 
# "440" , "Dental" , "Other" , 
# "450" , "Pain Management" , "Other" , 
# "666" , "Neurosurgery" , "Neurosurgery" , 
# NA_character_ , "Unknown" , "UNKNOWN" , 
# "480" , "Orthopaedic Spine" , "Orthopaedics" 
# ) %>% matrix(ncol=3, byrow=TRUE) %>% as_tibble %>% set_names(c("Service", "epic_name", "Surg_Type") )



## because the map is not 1-1 randomly break ties; this is better than mapping the epic data down since I will update with epic data soon
## this maps each case multiple times
preops %<>% left_join(epic_service_numerics %>% select(Service, Surg_Type), by="Surg_Type" ) 
preops %<>% select(-Surg_Type)
preops %<>% group_by(caseid) %>% slice_sample(n=1, replace=FALSE) %>% ungroup

## labs are in some weird unintelligible units soemtimes
## TODO: consider on a big sample from PRD, define a rank transformation of labs. Tree based estimators are invariant to rank transform. This is easier than manually checking the units.

## name the columns in the same way as epic - this is important so that column transformers work, we will also use this order for the output and for type declarations. there is almost certainly a less ugly tidy way to do this.

preops %<>% rename_with( function(x){ if_else(x %in% name_alignment$MVName, name_alignment$EpicName[match(x, name_alignment$MVName )], x) } )

## load labels

de_i_outs <- paste0(de_i_location ,"outcomes.csv")

outcomes <- read_csv(de_i_outs)

## this could also be an inner join followed by a select
## exclude with no preop
outcomes %<>% filter(!is.na(caseid))
outcomes %<>% group_by(caseid) %>% slice_sample(n=1, replace=FALSE) %>% ungroup
outcomes %<>% semi_join(preops, by="caseid")
outcomes %<>% arrange(caseid)

## exclude with no outcomes
preops %<>% semi_join(outcomes, by="caseid")
preops  %<>% filter(!is.na(caseid))
preops %<>% arrange(caseid)

outcomes %<>% select(-caseid)
outcomes %>% write_csv("/pkghome/training_labels.csv")

preops %<>% select(-caseid)
preops %<>% relocate(any_of(name_alignment$EpicName) )
preops %>% write_csv("/pkghome/training_data.csv")

