cp  /home/christopherking/gitdirs/slate_testing/example_resources/* ~/slate_test/resources/

docker run --rm -it -v '/mnt/ris/ActFastData/:/research/' -v '/home/christopherking/gitdirs/slate_testing/script_helpers:/actfast_prep' -v "/home/christopherking/slate_test/resources/:/pkghome/" cryanking/verse_plus R --file /actfast_prep/mv_extract.R


docker run --rm -it -v '/home/christopherking/gitdirs/slate_testing/script_helpers:/actfast_prep' -v "/home/christopherking/slate_test/resources/:/pkghome/" cryanking/torchscipy:11.1 /bin/bash -c "pip install xgboost ; python /actfast_prep/model_artifacts.py"



docker run -it --rm  -v "/home/christopherking/slate_test/:/home/eccp/" --user eccp dsrt
-slate:v2.0 /bin/bash

pip install xlsx2csv
python -c "from xlsx2csv import Xlsx2csv; Xlsx2csv('/home/eccp/resources/Test Reporting Workbench 2021-08-03b.xlsx', outputencoding='utf-8').convert('resources/new_test.csv')"

dsutils new-project --org="wustl" --name="postop_death_test" --version=1.1 

cp /home/eccp/resources/*.p /home/eccp/postop_death_test/resources/
cp /home/eccp/resources/*.csv /home/eccp/postop_death_test/resources/
cp /home/eccp/resources/*.xgb /home/eccp/postop_death_test/resources/
cp /home/eccp/resources/Mortality_30_d_xgb.py /home/eccp/postop_death_test/src/postop_death_test/model_code.py

dsutils make-ondemand-payload --from-RW --root-dir /home/eccp/postop_death_test/ --samples=8 --file '/home/eccp/resources/new_test.csv'

cd postop_death_test

echo "xgboost==1.4.2" >> requirements.txt
echo "scikit-learn==0.24.1" >> requirements.txt
echo "ibex" >> requirements.txt
dsutils install xgboost==1.4.2 scikit-learn==0.24.1 ibex
dsutils archive

exit


rclone copy /home/christopherking/slate_test/postop_death_test/archive/ remote:/slate_archives/
