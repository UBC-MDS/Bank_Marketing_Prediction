#Dockerfile for dockerfile-practice
# DSCI_522 2021

# use jupyter/scipy-notebook:latest as the base image
ARG BASE_CONTAINER=jupyter/scipy-notebook:latest
FROM $BASE_CONTAINER

USER root

RUN apt-get update

USER $NB_UID


# install python 3 packages
RUN pip install \
    "numpy==1.21.*" \
    "pandas==1.3.*" \
    "docopt==0.6.*" \
    "scikit-learn==1.0.*" \
    "seaborn==0.11.*" \
    "altair==4.1.0"