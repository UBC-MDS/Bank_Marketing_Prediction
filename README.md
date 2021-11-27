# Bank Marketing Prediction

  - Contributors: Melisa Maidana, Steven Lio, Zheren Xu
	
Group data analysis project for DSCI 522 (Data Science Workflows); 
A course in the 2021 Master of Data Science program at the University of 
British Columbia.

## Introduction

Telemarketing campaigns can be very expensive to institutions. 
The possibility to predict the likelihood of customer response can lead to more efficient strategies that reduce implementation costs and maximize the success rate.

The objective of this project is to identify which customers are more likely 
to respond positively to a telemarketing campaign and subscribe to a new product (a long-term deposit). 
To address the predictive question posed above, we plan to conduct an exploratory data analysis 
and build a machine learning model that can predict if a certain customer looks alike the target audience for this product.

## Data

The data set used in this project is related with direct marketing campaigns (phone calls) of a Portuguese banking institution [@moro2014data] can be found [here](http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip).

## Report

The final report can be found [here](https://htmlpreview.github.io/?https://github.com/UBC-MDS/Bank_Marketing_Prediction/blob/main/doc/bank_marketing_prediction_report.html)


## Usage

To replicate the analysis, all relevant data, scripts and necessary dependencies are available in this
GitHub repository. 
Please run the following commands at the command line/terminal from the root directory of
this project after cloning the GitHub repository locally.

#### 1\. Script to download Banking dataset:

    python src/downloader.py http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip --path=data/raw
	
#### 2\. Run preprocessor script (will create a 80%/20% train/test split):

    python src/data_preprocessing.py data/raw/bank-additional/bank-additional-full.csv data/processed --test_split=0.2

#### 3\. Generate EDA tables and figures:

    python src/eda.py data/processed/bank-additional-train.csv results/eda

#### 4\. Build model (Dummy Classifier, Random Forest, Logistic Regression):

    python src/Build_Models.py data/processed/bank-additional-test.csv data/processed/bank-additional-train.csv results

## Dependencies
	
	- Python 3.9.0 and Python packages:
		- docopt==0.6.2
		- ipykernel
		- requests>=2.25.1
		- matplotlib>=3.4.2
		- pandas>=1.3.*
		- scikit-learn>=1.0
		- altair>=4.1.0
		- seaborn>=0.11.2
		- errno
		- zipfile
		- altair_saver>=0.5
		- vega-lite vega-cli canvas
	- R version 3.6.1 and R packages:
		- knitr=1.33
		- tidyverse=1.3.1
		- ggplot2=3.3.5
		- kableExtra=1.3.4

## License

The data set used in this Banking Marketing Prediction project is released by UCI. 
Detail of this dataset can be found [here](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing). 

The Banking Marketing Prediction materials (excluding original data set) here are licensed
under the MIT License found [here](https://github.com/stevenlio88/IMDB_Rating_Prediction/blob/main/LICENSE).

## References

This dataset is public available for research. The details are described in [@moro2014data] 

