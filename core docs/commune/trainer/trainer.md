# README

This script provides a `Trainer` class that interacts with ray's Hyperparameter search/tuning and commune's Module class. The `Trainer` class facilitates model training with hyperparameter tuning. The script uses ray's hyperparameter tuning package and the commune framework, which is a framework for distributed computation.

## Usage

1. First, import necessary modules.
2. To use, create an instance of the `Trainer` class. The class requires a model, metrics_server, and a custom tuning dictionary as inputs.
3. The `Trainer` class includes methods to set up the model and parameters, and then execute a hyperparameter search on a specified objective function.
4. The `Trainer` class also includes a method `fit` that trains the model using the tuned parameters.
5. If running as a script, a test instance of the `Trainer` class is created and its fit method is used.

## Functionality

* set_model: Ensures availability of the model and sets it in the class.
* set_config: Takes an array of objects and updates the configuration of the model.
* set_tuner: Sets up the hyperparameter tuner with specified or default values.
* hyper2params: Transform hyperparameters into parameters.
* get_hyperopt_tag: Retrieve the hyperparameter optimization tag based on the provided configurations.
* objective: The function to be optimized by the hyperparameter tuner.
* fit: Trains the model based on the tuned hyperparameters and returns the results.

## Dependencies

* torch
* ray
* commune
* typing
* copy

These can be installed with `pip install ray[tune] torch commune`.

## Run

After ensuring that the dependencies are installed, the script can be run using a python environment by calling `python scriptname.py`.

## Note

This script assumes that you have already set up ray and commune properly in your environment. Additionally, this script expects that a predefined model is available for use and training.