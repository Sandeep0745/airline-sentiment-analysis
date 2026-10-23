"""Reusable code for the airline sentiment analysis project.

Modules
-------
config         paths, random seed and label order shared by notebooks and scripts
data           loading, airline-label correction and the train/test split
preprocessing  tweet text cleaning (basic and sentiment-aware)
features       TF-IDF feature builders
modeling       model pipelines, the NB-weighted Logistic Regression and model I/O
evaluation     metrics, per-class reports and repeated-split evaluation
plotting       chart style, colour palette and figure export
train          command-line training of the final model
predict        command-line / programmatic inference with a saved model
"""
