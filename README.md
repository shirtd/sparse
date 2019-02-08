# Sparse Filtrations

An implementation of sparse filtrations from [A Geometric Perspective on Sparse Filtrations](https://arxiv.org/abs/1506.03797).

## Build

While the construction of the sparse graph and simplices is self contained the [Dionysus](http://mrzv.org/software/dionysus2/index.html) package is used to construct filtrations and compute persistent (co)homology.
In order to install Dionysus you will need

    * CMake
    * GCC >= 5.4
    * Boost 1.55.

On OSX GCC and CMake and Boost may all be installed with [Homebrew](https://brew.sh/)

    brew install gcc cmake boost

On Ubuntu

    sudo apt install libboost-all-dev cmake

To build the package and install all requirements run the following from the project's root directory

    python setup.py install

Tested on OSX 10.13 and Ubuntu 16.04 with Python 2.7
