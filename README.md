# IMDb Rating Prediction

  - contributors: Melisa Maidana, Steven Lio, Zheren Xu
	
Group data analysis project for DSCI 522 (Data Science Workflows); 
A course in the 2021 Master of Data Science program at the University of 
British Columbia.

## Introduction

For this project we are trying to build an average IMDb rating 
prediction model for a given movie. The model can be used to provide
an insights for movie makers and movie marketing decision maker to
estimate the public sentiment towards for a given new movie prior to
release. 

The [IMDb ratings](https://help.imdb.com/article/imdb/track-movies-tv/ratings-faq/G67Y87TFYYP6TWAV#) 
is the weighted average rating given by a IMDb registered user who 
can cast a vote (from 1 to 10) on every released title in the IMDb 
database. IMDb database includes all released movies, TV shows which 
are minimally have been shown in public for at least one time. Users 
can update and overwrite their existing vote as many times as they want.

Some questions we would like to answer with this model is if 
the meta data of a movie title or its film crew can give a prediction 
on the IMDb rating. Also over the years, does newer movies more likely
to recieve a higher or lower ratings and as well as if any given genre
will likely to recieve a higher or lower IMDb ratings as well.

To answer the predictive question posed above, we plan to first conduct a 
series of exploratory data analysis to assess the average IMDb rating 
distribution and compares it againts different infromation available in the 
IMDb listing such as movie genre, run times, specific actor/actress/director 
or award winning members in the film crews. 

For the model building process, the IMDb data may first filter out to include 
English titles only and those released in the North America/UK (Subject to change) 
as our target population. Then the data will be partition into training and testing 
set (split 80%:20%). The predictive model we will be exploring are varies type of 
regression models such as simple linear regression, k-nn regression, random forest 
regression and/or decision tree algorithms. We will use overall accuracy and the 
corresponding model validation metric such as R-square and may conduct variable 
selection analysis to refine the final model. Summary statistics and appropriate 
visualization as well as the whole model building process will be provided and 
included as part of the final report for this project.

After selecting the final model, we will re-fit the model on the whole training 
dataset, as well as evaluate it's performance and make actual prediction on 
upcoming new movies. We will assess the accuracy of the model and discuss potential 
reasons on the model performance as well as what attributes stand out the most
when predicting IMDb rating.

## Usage

To replicate the analysis, all relevant scripts will be made available in this 
GitHub repository. All necessarily dependencies will be provided and commands
required to fetch the relevant data will be provided as follow. Please run 
the following commands at the command line/terminal from the root directory of 
this project after cloning the GitHub repository to your machine.

Script to download IMDb dataset:

    python src/downloader.py https://datasets.imdbws.com/name.basics.tsv.gz data/raw
    python src/downloader.py https://datasets.imdbws.com/title.akas.tsv.gz data/raw
    python src/downloader.py https://datasets.imdbws.com/title.basics.tsv.gz data/raw
    python src/downloader.py https://datasets.imdbws.com/title.crew.tsv.gz data/raw
    python src/downloader.py https://datasets.imdbws.com/title.episode.tsv.gz data/raw
    python src/downloader.py https://datasets.imdbws.com/title.principals.tsv.gz data/raw
    python src/downloader.py https://datasets.imdbws.com/title.ratings.tsv.gz data/raw

## Dependencies
	
	- Python 3.9.0 and Python packages:
		- docopt==0.6.2
		- ipykernel
		- requests>=2.24.0
		- matplotlib>=3.2.2
		- pandas>=1.3.*
		- scikit-learn>=1.0
		- altair
		- pip
		- wikipedia
		- seaborn
		
	- R version 4.1.1 and R packages:
		- knitr==1.26
		- tidyverse==1.2.1

## License

The data set used in this IMDb Rating Prediction project is released by IMDb.com. 
Detail of this dataset can be found [here](https://www.imdb.com/interfaces/). 
The data is made publicly available under the [Non-Commercial Licensing](https://help.imdb.com/article/imdb/general-information/can-i-use-imdb-data-in-my-software/G5JTRESSHJBBHTGX?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=3aefe545-f8d3-4562-976a-e5eb47d1bb18&pf_rd_r=NHTH2EM9XVNMKMK4C9AK&pf_rd_s=center-1&pf_rd_t=60601&pf_rd_i=interfaces&ref_=fea_mn_lk1#) 
and [copyright/license](https://www.imdb.com/conditions?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=3aefe545-f8d3-4562-976a-e5eb47d1bb18&pf_rd_r=NHTH2EM9XVNMKMK4C9AK&pf_rd_s=center-1&pf_rd_t=60601&pf_rd_i=interfaces&ref_=fea_mn_lk2)
by IMDb.com. 

The IMDb Rating Prediction materials (excluding original data set) here are licensed
under the MIT License found [here](https://github.com/stevenlio88/IMDB_Rating_Prediction/blob/main/LICENSE).

## References

To be added later if required.
