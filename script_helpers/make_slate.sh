cp  /home/christopherking/gitdirs/slate_testing/example_resources/* ~/slate_test/resources/

docker run --rm -it -v '/mnt/ris/ActFastData/:/research/' -v '/home/christopherking/gitdirs/slate_testing/script_helpers:/actfast_prep' -v "/home/christopherking/slate_test/resources/:/pkghome/" cryanking/verse_plus R --file /actfast_prep/mv_extract.R

## this docker container is just jupyter/scipy-notebook:notebook-6.4.8 with xgboost installed
docker run --rm -it -v '/home/christopherking/gitdirs/slate_testing/script_helpers:/actfast_prep' -v "/home/christopherking/slate_test/resources/:/pkghome/" cryanking/slate_analysis:0.1 /bin/bash -c "pip install xgboost ; python /actfast_prep/model_artifacts.py"

docker run --rm -it -v '/home/christopherking/gitdirs/slate_testing/script_helpers:/actfast_prep' -v "/home/christopherking/slate_test/resources/:/pkghome/" cryanking/slate_analysis:0.1 /bin/bash -c "pip install xgboost ; python /actfast_prep/model_artifacts_aki.py"


docker run -it --rm  -v "/home/christopherking/slate_test/:/home/eccp/" --user eccp dsrt-slate:v2.1 /bin/bash

pip install xlsx2csv
python -c "from xlsx2csv import Xlsx2csv; Xlsx2csv('/home/eccp/resources/Test Reporting Workbench 2021-08-03b.xlsx', outputencoding='utf-8').convert('resources/new_test.csv')"

dsutils new-project --org="wustl" --name="postop_death_test_1_2" --version=1.2 

dsutils new-project --org="wustl" --name="postop_aki_test_1_2" --version=1.2 

cp /home/eccp/resources/*.p /home/eccp/postop_death_test_1_2/resources/
cp /home/eccp/resources/*.csv /home/eccp/postop_death_test_1_2/resources/
cp /home/eccp/resources/*.xgb /home/eccp/postop_death_test_1_2/resources/
cp /home/eccp/resources/Mortality_30_d_xgb.py /home/eccp/postop_death_test_1_2/src/postop_death_test_1_2/model.py
cp /home/eccp/resources/definition_mort.json  /home/eccp/postop_death_test_1_2/definition.json


cp /home/eccp/resources/*.p /home/eccp/postop_aki_test_1_2/resources/
cp /home/eccp/resources/*.csv /home/eccp/postop_aki_test_1_2/resources/
cp /home/eccp/resources/*.xgb /home/eccp/postop_aki_test_1_2/resources/
cp /home/eccp/resources/aki_xgb.py /home/eccp/postop_aki_test_1_2/src/postop_aki_test_1_2/model.py
cp /home/eccp/resources/definition_aki.json  /home/eccp/postop_aki_test_1_2/definition.json

dsutils make-ondemand-payload --from-RW --root-dir /home/eccp/postop_death_test_1_2/ --samples=8 --file '/home/eccp/resources/new_test.csv'
dsutils make-ondemand-payload --from-RW --root-dir /home/eccp/postop_aki_test_1_2/ --samples=8 --file '/home/eccp/resources/new_test.csv'


cd postop_death_test_1_2

echo "xgboost==1.5.2" >> requirements.txt
echo "scikit-learn==1.0.2" >> requirements.txt
dsutils install xgboost==1.5.2 scikit-learn==1.0.2
dsutils archive

cd ../postop_aki_test_1_2
echo "xgboost==1.5.2" >> requirements.txt
echo "scikit-learn==1.0.2" >> requirements.txt
dsutils install xgboost==1.5.2 scikit-learn==1.0.2
dsutils archive

exit


rclone copy /home/christopherking/slate_test/postop_death_test_1_2/archive/ remote:/slate_archives/
rclone copy /home/christopherking/slate_test/postop_aki_test_1_2/archive/ remote:/slate_archives/
