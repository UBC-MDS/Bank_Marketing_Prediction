# author: Melisa Maidana, Steven Lio, Zheren Xu
# date: 2021-12-02

#This script generates all the documents for the Bank Marketing prediction project

#Usage: make all

all: results/eda_summary_table.csv

# Download Banking dataset:
data/raw/bank-additional/bank-additional-full.csv data/raw/bank-additional/bank-additional.csv	data/raw/bank-additional/bank-additional-names.txt : src/downloader.py 
	python src/downloader.py http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip --path=data/raw

# Run preprocessor script (will create a 80%/20% train/test split):
data/processed/bank-additional-test.csv data/processed/bank-additional-train.csv :  data/raw/bank-additional/bank-additional-full.csv
	python src/data_preprocessing.py data/raw/bank-additional/bank-additional-full.csv data/processed --test_split=0.2

# Generate EDA tables and figures:
results/eda_barchart_by_marital.png results/eda_boxplot_by_age.png results/eda_summary_table.csv : data/processed/bank-additional-train.csv
	python src/eda.py data/processed/bank-additional-train.csv results/eda

# Build model (Dummy Classifier, Random Forest, Logistic Regression):
#python src/Build_Models.py data/processed/bank-additional-test.csv data/processed/bank-additional-train.csv results

clean: 
	rm -rf data/raw/bank-additional/bank-additional-full.csv 
	rm -rf data/raw/bank-additional/bank-additional.csv	
	rm -rf data/raw/bank-additional/bank-additional-names.txt 
	rm -rf data/processed/bank-additional-test.csv 
	rm -rf data/processed/bank-additional-train.csv
	rm -rf results/eda_barchart_by_marital.png 
	rm -rf results/eda_boxplot_by_age.png 
	rm -rf results/eda_summary_table.csv 