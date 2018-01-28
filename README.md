# Music_Origin_Prediction

Predicting the origin of a song based on audio features

## Problem Overview

Different music styles and genres are known to have unique sound wave characteristics. It is also speculated that different geographic regions have their own unique music style with its own unique characteristics. For this project we want to determine whether we can build a machine learning model that can accurately predict the origin of a song based on different sound wave characteristics retrieved from audio files.

The data used for this study was downloaded [here](http://archive.ics.uci.edu/ml/datasets/geographical+original+of+music). Prior exploration has been performed around this question. The most well-known study can be viewed [here](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7023456).

## Project Overview

The data consists of 1059 songs each having 68 audio features. Each song has a geographic origin in the form of latitude and longitude coordinates. There are 33 unique coordinate sets, which means that this project is a classification problem.

We want to classify each song by geographic location. The project code starts off with data cleaning, scaling the features, and then testing a variety of models and hyperparameter combinations on a range of features. It seems as if not all 33 features are useful for the predictions we are interested in. We use backwards selection and PCA to determine whether we can reduce the number of features used in our model.

Once the redundant features have been excluded, we can compare different models with optimal hyperparameter values to see which model has the highest accuracy score.

The project goes a step further and asks the question whether each of the 33 geographical clusters are unique in terms of music style. The reasoning is used that music style isn't necessarily constrained to borders. Based on this we determine whether we cannot reduce the number of clusters in a sensible manner - it still has to be grouped by geographical region, but if the reduced number of clusters obtain higher accuracy scores, it might be an indication that the original 33 geographical locations aren't unique in terms of music style.

Once we have determined a reduced number of geographical clusters that make sense, we re-run our best models to see whether the accuracy scores increase.

## How to use

The source code for the analysis is in the Jupyter notebook that can be found [here](https://github.com/johannesharmse/Music_Origin_Prediction/blob/master/src/Predicting%20the%20Geographical%20Origin%20of%20Music.ipynb). However, to run the analysis yourself, you need to clone or fork the repository. If you are unfamiliar with this process, you can follow the instructions at this [link](https://help.github.com/articles/fork-a-repo/).

Because the source code is in a Jupyter Notebook, you need to have `Jupyter` installed on your machine. The code has been written in Python, and thus you also need to have the latest version of `Python` installed. If you have these two software dependencies installed, you can open `Jupyter Notebook` and navigate to the location of the notebook.

The notebook explains step-by-step what we are doing and why we are doing it. If you want to re-run the code chunks, make sure that you have the package dependencies installed.

A summary of the analysis can be viewed in the report document.
