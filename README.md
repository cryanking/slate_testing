## Getting slate installed
This assumes you have some familiarity with docker and have a docker daemon set up. Copy the image from bjcdfs02.bjc-nt.bjc.org/Shared/HIP_Infrastructure/Innovation_Lab. For me that's
```
sudo mount -t cifs -o username=christopherking,domain=accounts,rw //bjcdfs02.bjc-nt.bjc.org/Shared/HIP_Infrastructure/Innovation_Lab /mnt/temp
```
and copy the Slate_v2.0 directory to a location of your choosing. Then, load the image into your docker index:
```
docker load < dsrt-slate-v2.0.tar
```
- since I wrote this, Slate v 2.1 is available, substitute whatever version is current

## Going from the old dataset to model artifacts
Prepare the list of features you're going to use and the type which they have (sklearn's column transformers will fix the order of features in live data to match training but depends on column names) and save that as a csv. I have a file which extracts those features from the metavision dataset and renames them appropriately. An example in this directory is in "script_helpers/mv_extract.R"

Then run a python script to load the training data, create the preprocessing steps (including imputation if you need it, detection of "bad" data) and train the classifier and regressor and save those artifacts. An example is in "script_helpers/model_artifacts.py" Save the package versions to a "requirements.txt" to install in slate, since a common problem is incompatability between the training and evaluation environment.

- You can install the desired packages in slate itself and train the classifier there, but you will need to manage the versions installed in *slate's python* and the *exported archive*. Training in slate has the advantage of fixing the python version between training and evaluation.
- I recommend using a container other than slate itself to do classifier training, because you will be able to use HPC and other environments where you might not want to or even be able to get the proprietary slate image configured.
- pickling python objects for the analysis workflow is mostly unavoidable, and generally ok, since you can fix the python and package versions to be the same between in training and slate.
- - Very few python objects can be serialized in a truly "self-contained" way, for example column transformers or imputation routines. You will therefore probably have to rely on pickled objects
- - tf.Transform allegedly allows serializing complete pipelinelines if you are willing to write it all in tensorflow and figure out the Beam and Spark dependencies. 
- - xgboost and tensorflow support exporting classifiers to stable serialized formats. TorchScript is also a stable serialization format. Linear-type classifiers can of course be exported to a table of coefficents. sklearn-onnx will serialize many sklearn classifiers, but not the pipeline.
- -  ExtraTreesRegressor in sklean is an imputation option that can be serialized to ONNX



## Using Slate
The slate image can be activated using docker. Passing a data directory as a volume is the best way to get data "into" slate. I happen to have an "eccp" user in my system, because that is the default for slate. Doing so avoids painful file permission issues. I have a "slate_test/resources" folder which has the artifacts that I want to use in the model. For example,

```
docker run -it --rm  -v "($pwd)/slate_test/:/home/eccp/" --user eccp dsrt-slate:v2.0 /bin/bash
```

Once in slate, initialize a project skeleton:

```
dsutils new-project --org="wustl" --name="postop_death_test" --version=1.1 
```
Note that version MUST be integer major.minor. The org must be wustl. The name must match the name in epic. You can change these in the definition.json file. The skeleton will contain a definition.json, but it will need to be edited (or preferably overwritten)

- You have to delete "capabilities" like `webcallout` if they are not actually enabled in hyperspace
- the `platform` won't match slate's version number. That's just Epic being epic.
- you could overwrite the python wrapper code's location and entrypoint, but there is no obvious reason to do so




Then, copy the model artifacts etc:

```
cp /home/eccp/resources/*.p ./postop_death_test/resources/
cp /home/eccp/resources/*.csv ./postop_death_test/resources/
cp /home/eccp/resources/*.xgb ./postop_death_test/resources/
cp /home/eccp/resources/definition_mort.json  ./postop_death_test/definition.json
cp /home/eccp/resources/Mortality_30_d_xgb.py ./postop_death_test/src/postop_death_test/model.py
```

The model.py that comes with the project skeleton is pretty well commented to help you figure out how to structure this model code. The parcel package is visible in ./project_name/pip_packages/parcel/ and has the routines for packaging data to and from the CCP. The epic_lib package visible in ./project_name/pip_packages/epic_lib has routines like get_logger for returning logging information and the module `cloud` for handling secrets and external APIs. 


Install package dependencies using `dsutils install`, which is a clunky wrapper around pip. For example,
```
cd postop_death_test
cat ./requirements.txt /home/eccp/resources/postop_death_requirements.txt > requirements.txt
xargs dsutils install < requirements.txt
```
or manually,
```
cd postop_death_test
echo "xgboost==1.4.2" >> requirements.txt
echo "scikit-learn==0.24.1" >> requirements.txt
dsutils install xgboost==1.4.2 scikit-learn==0.24.1 ibex
```
the exact pandas and numpy version rarely matter, but you can set them as well.


If you have a reporting workbench sample, transform it to a json file:
```
dsutils make-ondemand-payload --from-RW --root-dir /home/eccp/postop_death_test/ --samples=8 --file '/home/eccp/resources/new_test.csv'
```

Test that the model works:
```
dsutils ondemand | less
```

Modify your python until there are no errors, then package it up. This can require finessing the file permissions of the target directory. The slate package really wants to run as the "eccp" user. On my server, I created an eccp user and added them to my file permission group. You can also chmod +666 the target.
```
dsutils archive
```

The result is stored in the "archive" folder. I have rclone in a seperate docker image to move the file to a box folder for sharing:
```
rclone copy archive/ remote:/slate_archives/
```

