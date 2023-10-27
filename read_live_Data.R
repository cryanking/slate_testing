  library(readxl)
  library(dplyr)
  library(magrittr)
  library(jsonlite)
  raw_data <- read_xlsx("PME0921.xlsx", skip=4)

# raw_data %>% filter(is.na(`ERROR ID`))  %>% pull("LOG MESSAGE") %>% is.na %>% table
# .
# TRUE 
#  363 

# raw_data %>% filter(!is.na(`ERROR ID`)) %>% pull("LOG MESSAGE") %>% is.na %>% table
# .
# FALSE 
#   121 
  
raw_data %<>% filter(!is.na(`ERROR ID`))
raw_data %<>% select(log_data=`LOG MESSAGE`, instant_created=`INSTANT OF ENTRY`, date=`RECORD CREATION DATE`)



## create valid json (thanks, Epic)
temp <- raw_data[["log_data"]] %>% gsub(pattern="(?<![a-zA-Z])'", replacement='"', perl=T) %>% gsub(pattern="'(?![a-zA-Z])", replacement='"', perl=T) 


## write out the raw payloads and see if they work
dir.create("slate_real_time_output")

for (i in 1:nrow(raw_data)) {
  file_name <- paste0("slate_real_time_output/row_", raw_data$date[i] %>% gsub(pattern="/", replacement="_", fixed=T), "-", i, ".json")
    writeLines(temp[i], file_name)
}

## make a script
cat("#!/bin/bash", file="slate_real_time_output/run_payloads.sh", sep="\n")
mycplines <- vector(mode="character", length=nrow(raw_data)*2) 
mycplines[ seq(from=1, to=length(mycplines), by=2 )] <- paste0("cp /home/eccp/slate_real_time_output/row_", raw_data$date %>% gsub(pattern="/", replacement="_", fixed=T), "-" , seq.int(nrow(raw_data)), ".json /home/eccp/postop_death_test/resources/ondemand.json")
mycplines[ seq(from=2, to=length(mycplines), by=2 )] <- paste0("dsutils ondemand > /home/eccp/slate_real_time_output/row_" , seq.int(nrow(raw_data)) )

myfile <- file("slate_real_time_output/run_payloads.sh", "a")
writeLines(mycplines, con=myfile )
close(myfile)


temp %>% lapply( . %>% parse_json %>% extract2("Data") %>% 
  (function(x) x[ which(!(names(x) %in% c("OutputType","PredictiveContext","EntityId") ) ) ] ) %>% 
  sapply(  (function(x) {x[lengths(x)==0 ] <- NA_character_ ; unlist(x)} ) ) ) %>%
  lapply( (function(x) {if(is.matrix(x) ) {as_tibble(x)} else{tibble::as_tibble_row(x) } } ) ) %>%
  do.call(rbind, .)-> temp2

temp %>% lapply( . %>% parse_json %>% extract2("Data") %>% extract2("EntityId") %>% unlist %>%matrix(byrow=T, ncol=2)  %>% set_colnames(c("CSN", "IDtype")) %>% as_tibble ) -> temp3

for(i in seq_along(temp3)) { temp3[[i]]$batch <- i}

temp3 %<>% do.call(rbind, .) 

processed_data <- bind_cols( temp2, temp3)

processed_data %>% readr::write_csv("extracted_live_data.csv")

## I think there is evidence that we many need to trim_ws for strings and there may be double spaces that I don't expect ("PNEUMONIA  NONE") unless it is based on just contains()

## are the columns that don't transform to numerics well
## nothing surprising hee
processed_data %>% sapply( . %>% as.numeric %>% is.na %>% sum) %>% sort %>% tail(n=20)
processed_data %>% filter(is.na(as.numeric(Weight)) ) %>% select(Weight)
processed_data %>% filter(is.na(as.numeric(Ethnicity)) ) %>% select(Ethnicity)
processed_data %>% filter(is.na(as.numeric(PainScore)) ) %>% select(PainScore)
processed_data %>% filter(is.na(as.numeric(Race)) ) %>% select(Race)


score_results <- read_xlsx("RDI0921.xlsx", skip=4)

## The instants are overwhelmingly at the same time on a given date
## So, I think the next step is to loop over dates and data %>% tsv %>% shell(makeondemandpayload) %>% shell(ondemand) %>% shell(export)
## then get back in R


## load the slate results
my_results <- list.files(path="slate_real_time_output", pattern="\\d$", full.names=T)
slate_scores<- lapply(my_results , . %>% fromJSON() %>% extract2("Outputs") %>% extract2("Classification1") %>% extract2("Scores") %>% extract2("ICU") %>% extract2("Values"))
slate_scores<- data.frame( rownumber=rep( my_results %>% str_match(string=. , pattern="\\d+" ) %>% extract(,1) , times=lengths(slate_scores))  , scores= unlist(slate_scores))
slate_scores %<>% as_tibble

## merge them to time properties
slate_scores<- slate_scores %>% mutate(rownumber = as.integer(rownumber)) %>% left_join(raw_data %>% select(record_date=date,instant_created )%>% mutate(rownumber = seq.int(n())) , by="rownumber")

convert_epic_instant <- function(x) {
  this_timezone <- x %>% gsub(pattern="\\r.*", replacement="") %>% str_match(string=., pattern="[A-Z]+$")%>% extract(,1) 
  ## annoying overloaded abbrev (we aren't in cuba)
  this_timezone[this_timezone=="CDT"] <- "America/Chicago"
  this_longint <- x  %>% gsub(pattern="\\r.*", replacement="") %>% str_match(string=., pattern="^[0-9]+")%>% extract(,1)   %>% (bit64::as.integer64) %>% subtract(bit64::as.integer64(47116*60*60*24) ) 
  ## manually convert to unix era to reduce issues with long ints
  this_date<- as_datetime(as.integer(this_longint))
  tz(this_date) <- this_timezone 
  return(this_date)
}

slate_scores %<>% mutate(instant_created = instant_created %>% convert_epic_instant)

## compare the instants


score_results %>% select(c(contains("INSTANT") , contains("DATE")) ) %>% mutate(across( one_of("RULE FILING INSTANT UTC",  "SCORE FILING INSTANT UTC", "INSTANT OF ENTRY"),  . %>% convert_epic_instant %>% with_tz("America/Chicago"))) %>% mutate(delta1 = mdy(`CONTACT DATE`)  - mdy(`PATIENT DATE`), delta2 = mdy(`PATIENT DATE`) - mdy(`RECORD CREATION DATE`) ) %>% select(delta1, delta2) %>% table
## after setting the tz, these time components are basically the same
## the dates are usually 1 day, occasionally 0 or 2 days different, not clear why

score_results %>% select(c(contains("INSTANT") , contains("DATE")) ) %>% mutate(across( one_of("RULE FILING INSTANT UTC",  "SCORE FILING INSTANT UTC", "INSTANT OF ENTRY"),  . %>% convert_epic_instant %>% with_tz("America/Chicago"))) %>% mutate(delta1 = (`RULE FILING INSTANT UTC`)  - (`INSTANT OF ENTRY`), delta2 = date(`SCORE FILING INSTANT UTC`) - mdy(`RECORD CREATION DATE`) ) %>% select(delta1, delta2) %>% lapply(table)

score_results %>% select(record_date = `RECORD CREATION DATE`, entry_time=`INSTANT OF ENTRY` ,epic_score = `ACUITY RULE SCORE`) %>% mutate(entry_time = entry_time %>% convert_epic_instant) -> live_scores_only
# "SCORE FILING INSTANT UTC"



## loop through dates
for (this_date in live_scores_only$entry_time %>% date %>% unique %>% sort) {
  list(live_scores_only %>% filter(date(entry_time) == this_date) %>% pull("epic_score") %>% as.numeric %>% round(1) %>% sort
  , slate_scores %>% filter(date(instant_created) == this_date) %>% pull("scores") %>% as.numeric %>% round(1) %>% sort ) %>% print
  temp <- readline()
#   if (nchar(temp >0)) { break() }
}

## this doesn't work either - the distribution of hours is very different, and the number very different. The number based on any reasonable matching is very different

# score_results <- score_results %>% filter( `RECORD CREATION DATE` %in% raw_data$date)
# 
# overlaps <- function(x, y) { c(
# length(setdiff(x,y)) ,
# length(intersect(x,y)) ,
# length(setdiff(y,x)) 
# )}
# 
# overlaps(raw_data$instant_created, score_results[["INSTANT OF ENTRY"]] )
# 
# 
# convert_time_epic <- . %>% gsub(pattern="\\r.*", replacement="") %>% strsplit(.,split="[", fixed=T) %>% unlist %>% matrix(byrow=T, ncol=2) %>% extract(,1) %>% (bit64::as.integer64) %>% subtract(bit64::as.integer64("5762000000"))
# 
# ## these are very close
# qqplot(raw_data$instant_created %>% convert_time_epic %>% add(18000), score_results[["SCORE FILING INSTANT UTC"]] %>%  convert_time_epic)
# 
# ## both in CDT!
# median(raw_data %>% filter( date == "09/21/2023") %>% pull("instant_created") %>% convert_time_epic) -
# median( score_results%>% filter( `RECORD CREATION DATE`  == "09/21/2023") %>% pull("INSTANT OF ENTRY") %>% convert_time_epic )
# 
# qqplot(raw_data$instant_created %>% convert_time_epic, score_results[["INSTANT OF ENTRY"]] %>% convert_time_epic )
