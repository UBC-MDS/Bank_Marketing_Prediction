#Dockerfile for dockerfile-practice
# DSCI_522 2021

# use jupyter/scipy-notebook:latest as the base image
FROM jupyter/scipy-notebook:latest


# install python 3 packages
RUN pip install \
    "numpy==1.21.*" \
    "pandas==1.3.*" \
    "docopt==0.6.*" \
    "scikit-learn==1.0.*" \
