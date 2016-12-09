FROM continuumio/miniconda3

MAINTAINER Maurizio Tomasi <maurizio.tomasi@unimi.it>

RUN apt-get -y upgrade
RUN apt-get install -y \
    apt-utils \
    g++ \
    gcc \
    git \
    make \
    noweb \
    openmpi-bin \
    texlive-latex-base
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
