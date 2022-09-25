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

## Problems training on Vertex AI
I followed the following tutorials on two different regions (eu-west1 and eu-west4):
* https://codelabs.developers.google.com/codelabs/vertex-ai-custom-code-training
* https://codelabs.developers.google.com/vertex_custom_training_prediction

When training, after 2-3 minutes I got the following error
`Training pipeline failed with error message: The following quota metrics exceed quota limits: aiplatform.googleapis.com/custom_model_training_cpus`

Then I ran into [this stackoverflow question](https://stackoverflow.com/questions/73368320/vertax-ai-pipeline-quota) 
which only proposal was to change the region, which turned out not to solve the problem.

After several tries, deploying "Hello world"'s images with Docker to see if that work, using different tutorials and 
being unable to contact customer service, I decided to just adapt the code the way it would be deployed on Vertex AI.

Note: Using the auto-ml tool from google works.