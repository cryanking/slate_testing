## Getting slate installed
This assumes you have some familiarity with docker and have a docker daemon set up. Copy the image from bjcdfs02.bjc-nt.bjc.org/Shared/HIP_Infrastructure/Innovation_Lab. For me that's
```
sudo mount -t cifs -o username=christopherking,domain=accounts,rw //bjcdfs02.bjc-nt.bjc.org/Shared/HIP_Infrastructure/Innovation_Lab /mnt/temp
```
and copy the Slate_v2.0 directory to a location of your choosing. Then, load the image into your docker index:
```
docker load < dsrt-slate-v2.0.tar
```

## Going from the old dataset to model artifacts
Prepare the list of features you're going to use and the type which they have (features in training will need to have the same order as features in test) and save that as a csv. I have a file which extracts those features from the metavision dataset and renames the appropriately. There is a switch to use the unimputed or imputed datasets. An example in this directory is in "script_helpers/mv_extract.R"

Then run a python script to load the training data, create the preprocessing steps and imputation and train the classifier and regressor and save those artifacts. An example is in "script_helpers/model_artifacts.py"


## Using Slate
The slate image can be activated using docker. Passing a data directory as a volume is the best way to get data "into" slate. I happen to have an "eccp" user in my system, because that is the default for slate. Doing so avoids painful file permission issues. I have a "resources" folder which has the artifacts that I want to use in the model. For example,

```
docker run -it --rm  -v "/home/christopherking/slate_test/:/home/eccp/" --user eccp dsrt-slate:v2.0 /bin/bash
```

Once in slate, initialize a project skeleton:

```
dsutils new-project --org="WUSM-testing" --name="DataIntegration" --version=0.001 
```

If you have a reporting workbench sample, transform it to a json file:
```
dsutils make-ondemand-payload --from-RW --root-dir ./DataIntegration/ --samples=591 --file './resources/Test Reporting Workbench 2021-05-19.csv'
```

Then, copy the model artifacts etc, then enter the directory

```
cp resources/*.p ./DataIntegration/resources/
cp resources/*.csv ./DataIntegration/resources/
cp resources/*.xgb ./DataIntegration/resources/
cd DataIntegration
```
Copy over the model.py if you have already made it:
```
cp resources/Mortality_30_d_xgb.py ./DataIntegration/src/DataIntegration/model.py
```

Install package dependencies, for example,

```
dsutils install xgboost==1.4.2 scikit-learn==0.24.2 ibex
```

Make sure to add these to the requirements.txt:
```
echo "xgboost==1.4.2" >> requirements.txt
echo "scikit-learn==0.24.2" >> requirements.txt
echo "ibex" >> requirements.txt
```

The default "definition.json" needs to be modified for any callouts or web secrets. Test that the model works:
```
dsutils ondemand > test_output.kson
```

Modify your python until there are no errors, then package it up
```
dsutils archive
```

The result is stored in the "archive" folder. I have rclone in a seperate docker image to move the file to a box folder for sharing:
```
rclone copy archive/ remote:/slate_archives/
```
