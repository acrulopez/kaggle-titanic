# Kaggle-titanic
This repository is done for the THC - Part 2: Coding Challenge of the Astrafy internship.

## Installation
The python version used is 3.6.0. You can install the requirements with `pip install -r requirements.txt` on the root folder
of the repository.

## Running the example
You can run the example with `python src/mlops/main.py --model-dir=""` or `python src/mlops/main.py` on the root folder of the repository.

## Building the package
You can run `python setup.py sdist` on the root directory.

## Considerations
Following the [KISS principle](https://en.wikipedia.org/wiki/KISS_principle) and [avoiding using OOP](https://dpc.pw/the-faster-you-unlearn-oop-the-better-for-you-and-your-software) when it only creates more complexity, I decided not 
to include classes for each of the MLOps steps. This does not mean that on a more complex project that structure
could be convenient.