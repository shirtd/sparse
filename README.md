# Sparse Filtrations

An implementation of sparse filtrations from [A Geometric Perspective on Sparse Filtrations](https://arxiv.org/abs/1506.03797).

## Build

While the construction of the sparse graph and simplices is self contained the [Dionysus](http://mrzv.org/software/dionysus2/index.html) package is used to construct filtrations and compute persistent (co)homology.
In order to install Dionysus you will need

    - CMake
    - GCC >= 5.4
    - Boost 1.55.

On OSX GCC, CMake, and Boost may all be installed with [Homebrew](https://brew.sh/)

    brew install gcc cmake boost

On Ubuntu

    sudo apt install libboost-all-dev cmake

To build the package and install requirements run the following from the project's root directory

    python setup.py install

Tested on OSX 10.13 and Ubuntu 16.04 with Python 3.7

## Usage

The package includes an example program which compares the rips filtration (constructed with Dionysus) with the sparse rips filtration.
To run the program with progress and timing comparisons

    python main.py -v

Additional command line arguments

    usage: main.py [-h] [--epsilon EPSILON] [--noise NOISE] [--dim DIM]
                   [--prime PRIME] [--function {circle,double}] [--thresh THRESH]
                   [--uniform] [--n N] [--verbose] [--plot] [--cohomology]

    sparse rips (co)homology

    optional arguments:
      -h, --help            show this help message and exit
      --epsilon EPSILON, -e EPSILON
                            sparsity. default: 0.10
      --noise NOISE, -o NOISE
                            noise multiplier. default: 0.2
      --dim DIM, -d DIM     max rips dimension. default: 2
      --prime PRIME, -p PRIME
                            prime (co)homology coefficient. default: 2
      --function {circle,double}, -f {circle,double}
                            shape. default: double
      --thresh THRESH, -t THRESH
                            rips cutoff. default: 2.828
      --uniform, -u         uniformly sample shape. default: False
      --n N, -n N           number of points. default: 200
      --verbose, -v         verbose output. default: False
      --plot                plot diagrams. default: False
      --cohomology, -c      persistent cohomology. default: persistent homology
