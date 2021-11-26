# Bank Marketing Prediction

  - contributors: Melisa Maidana, Steven Lio, Zheren Xu
	
Group data analysis project for DSCI 522 (Data Science Workflows); 
A course in the 2021 Master of Data Science program at the University of 
British Columbia.

## Introduction

For this project we are trying to build a binary classifier to predict if a
banking customer will subscribe (Y) to a new product (bank term deposit) if
they are contacted by the bank with a phone call. This model would help banking
institution to optimize marketing strategies and budget allocation to achieve
a higher new product subscription rate. If choosing an appropriate model for the
binary classifier, we can explore the different attributes of the type of
customers who will likely subscribe to the new product when contacted over the phone.

To answer the predictive question posed above, we plan to first conduct a
series of exploratory data analyses to assess the relationship between subscribing
status against other available attributes in the dataset. We will examine the
variable distributions using summary tables and some data visualization.

For the model building process, the Bank data will be partitioned into training and
testing sets (split 80%:20%). The binary predictive models we will be exploring are
[Logistic Linear Regression](https://en.wikipedia.org/wiki/Logistic_regression) and 
[Random Forest Classifier](https://en.wikipedia.org/wiki/Random_forest). The model will be examined
based on the overall accuracy on the training data and using cross-validation method 
for hyper-parameter tuning. Metrics such as recall rate, precisions and confusion
matrix are assessed. After selecting the final model, we will re-fit the model on the
whole training dataset, as well as evaluate its performance, make predictions and
assess the overperformance of the final model using the testing dataset. Summary
statistics and appropriate visualization as well as the whole model building process
will be provided and included as part of the final report for this project.

## Usage

To replicate the analysis, all relevant scripts will be made available in this
GitHub repository. All necessary dependencies will be provided and commands
required to fetch the relevant data will be provided as follow. Please run
the following commands at the command line/terminal from the root directory of
this project after cloning the GitHub repository to your machine.

#### 1\. Script to download IMDb dataset:

    python src/downloader.py http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip --path=data/raw
	
#### 2\. After file download run the following to process dataset (for a 80%/20% train/test split):

    python src/data_preprocessing.py data/raw/bank-additional/bank-additional-full.csv data/processed --test_split=0.2

#### 3\. Script to build model (Dummy Classifier, Random Forest, Logistics Regression):

    python src/Build_Models.py data/processed/bank-additional-train.csv data/processed/bank-additional-test.csv results

## Dependencies
	
	- Python 3.9.0 and Python packages:
		- docopt==0.6.2
		- ipykernel
		- requests>=2.24.0
		- matplotlib>=3.2.2
		- pandas>=1.3.*
		- scikit-learn>=1.0
		- altair
		- wikipedia
		- seaborn
		- errno
		- zipfile

## License

The data set used in this Banking Marketing Prediction project is released by UCI. 
Detail of this dataset can be found [here](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing). 

The Banking Marketing Prediction materials (excluding original data set) here are licensed
under the MIT License found [here](https://github.com/stevenlio88/IMDB_Rating_Prediction/blob/main/LICENSE).

## References

This dataset is public available for research. The details are described in [Moro et al., 2014]

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
