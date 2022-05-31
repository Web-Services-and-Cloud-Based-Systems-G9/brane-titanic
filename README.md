# Brane Data Processing Pipeline for the Titanic dataset
The following repository comprises Implementation of a Processing and Visualization packages in brane for Data Science and Machine Learning tasks on a Kaggle competition dataset related to the sinking of the Titanic. This project correspond to Assignment 4.b of the Web Services and Cloud Based Systems at University of Amsterdam (G9).

## Introduction

### The objective
The Titanic was a legendary ship which sank in the North Atlantic Ocean during its first voyage from Southampton, UK to New York City, USA after hitting an iceberg on its way.  

A dataset containing anonymized information about all the passengers such as their age, ticket fare, class inside the ship, etc., was made available as part of a [Kaggle competition](https://www.kaggle.com/competitions/titanic/). [Kaggle](www.kaggle.com) is a website which hosts machine learning competitions between users from all the internet. The objective of this competition, and our objective is to predict if a passenger survived or not to the ship sinking given the information of the passenger.

### Our solution
Using a Decision Tree classifier and the relevant features of the dataset obtained with a previous exploratory data analysis, we will predict for each passenger their survival outcome (1 for survived, 0 for not). In order to do this, we also have to address missing values in the dataset. 

### The tool
We will use Python as programming language for this task. We will use Pandas to manage the dataset, matplotlib and seaborn for visualization purposes and scikit-learn for the Machine Learning task. However, in order for our analysis to be replicable at a high level we will use the [Brane framework](https://wiki.enablingpersonalizedinterventions.nl/user-guide/overview.html). Brane is a framework that addresses the organizational challenges of sharing and replicating parts of a process at a high level. In other words, it lets technical teams to implement their processes as building blocks.

## Brane Packages
We will implement two [Brane packages](https://wiki.enablingpersonalizedinterventions.nl/user-guide/software-engineers/hello-world.html) for our pipeline. One for the processing tasks and another one for the visualization tasks. 

### Processing Package (`titanicprocessing`)

Located at `/brane_packages/processing`, the processing package is comprised of three methods that can be used as building blocks in any [BraneScript](https://wiki.enablingpersonalizedinterventions.nl/user-guide/branescript/introduction.html) pipeline.   
- Drop Unuseful Columns 
- Transform Features
- Train and Predict

For more documentation on each method refer to the README file inside the package.

### Visualization Package (`titanicviz`)

Located at `/brane_packages/visualization`, the visualization package is comprised of three methods that can be used as building blocks in any [BraneScript](https://wiki.enablingpersonalizedinterventions.nl/user-guide/branescript/introduction.html) pipeline.   
- Create Histogram and KDE plot
- Create Stacked Barchart
- Create multi-column Barchart

For more documentation on each method refer to the README file inside the package.

### Publishing (locally)
1. Make sure you have Brane installed locally
2. On the root of the repository run `brane import`

### Testing (locally)
Tests were implemented in [Pytest](https://docs.pytest.org/en/6.2.x/contents.html). There are six tests that needs to pass (3 for the visualization package and 3 for the processing package). Each test checks the correctness YAML output of each method. Some of them check deterministic correctness on the results. Others only check if the output have a correct format.
1. Install pipenv `pip3 install pipenv`
2. Run the tests `pipenv run test`

## Automated Tests
This repository has a GitHub Actions workflow configured that runs automated tests to ensure the methods of the packages work correctly. In addition to this, it makes sure that the package can be deployed successfully by implementing a test which tries to do the deployment process.

## Example Pipeline (BraneScript)
There is an example pipeline in the `pipeline.bs` file which can be run in BraneScript. A Jupyter Notebook with the same pipeline is also available in `pipeline.ipynb` file.