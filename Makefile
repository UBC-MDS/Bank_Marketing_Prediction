# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-12-02

#This script retrieve data from source and generates all files and model for the Bank Marketing prediction project

#Usage: make all

all: doc/bank_marketing_prediction_report.md

# Download Banking dataset from source unzip and save it to data/raw:
data/raw/bank-additional/bank-additional-full.csv data/raw/bank-additional/bank-additional.csv	data/raw/bank-additional/bank-additional-names.txt : src/downloader.py 
	python src/downloader.py http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip --path=data/raw

# Run preprocessor script on downloaded data and save to data/processed, split the raw data into 80% training /20% test and clean target column y to 0 and 1:
data/processed/bank-additional-test.csv data/processed/bank-additional-train.csv :  data/raw/bank-additional/bank-additional-full.csv
	python src/data_preprocessing.py data/raw/bank-additional/bank-additional-full.csv data/processed --test_split=0.2

# Generate EDA tables and figures for the final report:
results/eda_barchart_by_target.png : data/processed/bank-additional-train.csv
	python src/eda.py data/processed/bank-additional-train.csv results

# Train and find best hyper-parameters for Random Forest and Logistic Regression models and output summary graphs and tables:
results/RFC_BestParams.csv results/BestRandomForest_ConMat_Test.jpg results/LR_BestParams.csv results/BestLogisticRegression_ConMat_Test.jpg results/BestLogisticRegression_Coefficients.csv : data/processed/bank-additional-train.csv data/processed/bank-additional-test.csv
	python src/Build_Models.py data/processed/bank-additional-test.csv data/processed/bank-additional-train.csv results

# Render the final report in Rmarkdown and html file
doc/bank_marketing_prediction_report.html doc/bank_marketing_prediction_report.md: doc/bank_marketing_prediction_report.Rmd doc/bank_marketing_refs.bib doc/Attribute_Info.csv results/eda_barchart_by_target.png results/RFC_BestParams.csv results/BestRandomForest_ConMat_Test.jpg results/LR_BestParams.csv
	Rscript -e "rmarkdown::render('doc/bank_marketing_prediction_report.Rmd')"
	
clean:
	rm -rf data/raw
	rm -rf data/processed	
	rm -rf results/
	rm -rf doc/bank_marketing_prediction_report.html