FROM continuumio/miniconda3

MAINTAINER Maurizio Tomasi <maurizio.tomasi@unimi.it>

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get install -y \
    apt-utils \
    g++ \
    gcc \
    gfortran \
    git \
    inkscape \
    make \
    noweb \
    openmpi-bin \
    texlive-latex-base \
    texlive-latex-extra
RUN conda install \
    astropy \
    click \
    matplotlib \
    mpi4py \
    numba \
    numpy \
    pytest \
    scipy
RUN pip install \
    autopep8 \
    healpy \
    quaternionarray
CMD git clone https://github.com/ziotom78/dacapo_calibration && \
    cd dacapo_calibration && \
    make all && \
    make fullcheck
