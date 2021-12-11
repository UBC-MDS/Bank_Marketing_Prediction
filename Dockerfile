#Dockerfile for Bank_Marketing_Prediction
# DSCI_522 Milestone 4 Group Project 2021

#Authors: Melisa Maidana, Steven Lio, Zheren Xu
#Date: 2021-12-10

# use jupyter/scipy-notebook:latest as the base image
ARG BASE_CONTAINER=jupyter/scipy-notebook:latest
FROM $BASE_CONTAINER

USER root

# Pre-requisites files for R
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    fonts-dejavu && \
    apt-get clean && rm -rf /var/lib/apt/lists/*  

# use mamba to install base R.
RUN mamba install --quiet --yes \
    'r-base' \
    'r-devtools' \
    'r-irkernel' && \
    mamba clean --all -f -y

# Other R packages install through mamba.
RUN set -x && \
    arch=$(uname -m) && \
    if [ "${arch}" == "x86_64" ]; then \
        mamba install --quiet --yes \
            'r-rmarkdown' \
            'r-tidyverse' && \
            mamba clean --all -f -y; \
    fi;
# install kableExtra for R
RUN Rscript -e "install.packages('kableExtra',repos = 'http://cran.us.r-project.org')"

# Install Python packages
RUN mamba install --quiet --yes \
    'ipykernel' \
    'ipython>=7.15' \
    'scikit-learn>=1.0' \
    'docopt' \
    'pandas>=1.3.*'&& \
    mamba clean --all -f -y 

# Install pandoc by conda (or M1 Mac would got an error)
RUN conda install -c conda-forge pandoc

USER ${NB_UID}

WORKDIR "${HOME}"
