# Bank Marketing Prediction

  - Contributors: Melisa Maidana, Steven Lio, Zheren Xu
	
Group data analysis project for DSCI 522 (Data Science Workflows); 
A course in the 2021 Master of Data Science program at the University of 
British Columbia.

## About

Telemarketing campaigns can be very expensive to institutions. The possibility to predict the 
likelihood of customer response to a campaign can lead to more efficient strategies that reduce 
implementation costs and maximize the success rate.

There are two main objectives of this project, first is to develop a predictive model which can 
be used to identify which customers are more likely to respond positively to a telemarketing 
campaign and subscribe to a new product (a long-term deposit) if contacted by the bank over the 
phone.

Second is to explore what can we learn from the predictive model to understand the types of customers 
the bank should prioritize on contact and what are the limitation of the information in the data sets.

Our final classifier was a Logistic Regression model that achieved an F1 score of 59.7% and a recall of 90%. 
The features identified as more important were related to the duration of the call, the month of contact, 
the past reaction of the customer to previous calls, and the Consumer Price Index.
We believe that some additional features, such as the reason for the last contact, could bring additional 
value to the model and help to improve its precision and False Positive rate.

## Data

The data set used in this project is related to direct marketing campaigns (phone calls) of a Portuguese banking institution [@moro2014data] can be found [here](http://archive.ics.uci.edu/ml/machine-learning-databases/00222).
The detail of the dataset is documented [here](http://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

## Report

The final report can be found [here](https://htmlpreview.github.io/?https://github.com/UBC-MDS/Bank_Marketing_Prediction/blob/main/doc/bank_marketing_prediction_report.html)


## Usage

To replicate the analysis, all relevant data, scripts and necessary dependencies are available in this
GitHub repository and you can use either [Docker](https://www.docker.com/get-started) or run the makefile
commands in the command line/terminal at the root of this project directory after cloning this GitHub 
repository locally. 

1. Using Docker

To replicate this analysis, you should download and install [Docker](https://www.docker.com/get-started). Then clone this 
GitHub repository and run the following command in the command prompt/terminal from the root directory of this project:

	docker run --rm -v PATH_ON_YOUR_COMPUTER:/home/bank_marketing_prediction ****/Bank_Marketing_Prediction make -C '/home/bank_marketing_prediction' all

To reset the repo to the original clean state, run the following command in the command prompt/terminal from the root
directory of this project:

	docker run --rm -v PATH_ON_YOUR_COMPUTER:/home/bank_marketing_prediction ****/Bank_Marketing_Prediction make -C '/home/bank_marketing_prediction' clean 

2. Using make command 

You can also replicate the analysis in this project by cloning this GitHub repository and install all necessary dependencies 
listed below, run the following command in the command prompt/terminal from the root directory of this project to replicate 
the full analysis and final report:

    make all

To reset the repo to the original state and delete all results files and report, run the following command at the command
prompt/terminal from the root directory of this project:

    make clean

#### Make file dependency diagram:

![dependency_diagram](Makefile.png)


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
under the MIT License found [here](https://github.com/UBC-MDS/Bank_Marketing_Prediction/blob/main/LICENSE).

## References

This dataset is public available for research. The details are described in [@moro2014data](http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip).

