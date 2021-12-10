#Dockerfile for Bank_Marketing_Prediction
# DSCI_522 Milestone 4 Group Project 2021

#Authors: Melisa Maidana, Steven Lio, Zheren Xu
#Date: 2021-12-09

# use jupyter/scipy-notebook:latest as the base image
ARG BASE_CONTAINER=jupyter/scipy-notebook:latest
FROM $BASE_CONTAINER

USER root

RUN apt-get update

USER $NB_UID

USER root

# R pre-requisites
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    fonts-dejavu \
    unixodbc \
    unixodbc-dev \
    r-cran-rodbc \
    gfortran \
    gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*  

# R packages including IRKernel which gets installed globally.
RUN mamba install --quiet --yes \
    'r-base' \
    'r-devtools' \
    'r-irkernel' && \
    mamba clean --all -f -y

# These packages are not easy to install under arm
RUN set -x && \
    arch=$(uname -m) && \
    if [ "${arch}" == "x86_64" ]; then \
        mamba install --quiet --yes \
            'r-rmarkdown' \
            'r-tidymodels' \
            'r-tidyverse' && \
            mamba clean --all -f -y; \
    fi;


# # install the kableExtra package using install.packages
RUN Rscript -e "install.packages('kableExtra',repos = 'http://cran.us.r-project.org')"

# install the anaconda distribution of python
# Install Python 3 packages
RUN mamba install --quiet --yes \
    'ipykernel' \
    'ipython>=7.15' \
    'pip' \
    'selenium' \
    'scikit-learn>=1.0' \
    'docopt' \
    'pandas>=1.3.*'&& \
    mamba clean --all -f -y 

RUN apt-get update && apt-get install -y chromium-chromedriver
RUN conda install -c conda-forge altair_saver
RUN npm install -g --force vega-lite vega-cli canvas vega --unsafe-perm=true


USER ${NB_UID}

WORKDIR "${HOME}"

# Example usage:
# docker build --tag v0.1.0 /$(pwd)
# docker run --rm -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v "${PWD}":/home/jovyan/work v0.1.0

# docker build --platform --tag v0.1.0 /$(pwd)












