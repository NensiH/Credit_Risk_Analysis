# Credit_Risk_Analysis

## Overview:
Credit risk is very tough to predict. In this project we want to take a look at how all the factors in our loan_stats csv help predict whether someone is low or high risk status. The purpose of this analysis was to create a supervised machine learning model that could accurately predict credit risk. In order to complete this task, we trained and evaluated several models to predict credit risk.

I used 6 different methods, which are:

1. Naive Random Oversampling
2. SMOTE Oversampling
3. Cluster Centroid Undersampling
4. SMOTEENN Sampling
5. Balanced Random Forest Classifying
6. Easy Ensemble Classifying

Through each of these methods, I split my data into training and testing datasets, and compiled accuracy scores, confusion matries, and classification reports as my results.

## Resources:
- **Data Source:** LoanStats_2019Q1.csv
- **Software:** Jupyter Notebook, Anaconda Navigator
- **Environment:** Python 3.7
  - Dependencies
    - Numpy
    - Pandas
    - Pathlib
    - Collections
    - SKLearn



## Results:
In this analysis I used six different algorithms of supervised machine learning. First four algorithms are based on resampling techniques and are designed to deal with class imbalance. After the data is resampled, Logistic Regression is used to predict the outcome. Logistic regression predicts binary outcomes. The last two models are from ensemble learning group. The concept of ensemble learning is the process of combining multiple models, like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model, and therefore increase the overall performance of the model.

Credit card data from LoanStats_2019Q1.csv was cleaned prior to implementing machine learning techniques. Null columns and rows were dropped, interest rates were converted to numerical values, and target (y-axis) columns were converted to low_risk and high_risk based on their values.

Once the data was cleaned, it was split into training and testing categories, which resulted in four sets of data:

- X_train
- X_test
- y_train
- y_test

A random_state of 1 was used across all models to ensure reproducible output.

The balance of low_risk and high_risk is unbalanced, but this was expected as credit risk is an inherently unbalanced classification problem, since good loans easily outnumber risky loans.

### 1. Naive Random Oversampling and Logistic Regression

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. Once the datasets were balanced, the model trained the data, which is where the algorithm analyzes the data and attempts to learn patterns in the data.

Naive random oversampling on this data gave the following scores:

### 2. SMOTE Oversampling and Logistic Regression

The synthetic minority oversampling technique (SMOTE) is another oversampling approach where the minority class is increased. Unlike other oversampling methods, SMOTE interpolated new instances, that is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

Once the data were balanced and trained, SMOTE oversampling gave the following scores:

### 3. Cluster Centroids Undersampling and Logistic Regression

The ClusterCentroid algorithm provides an efficient way to represent the data cluster with a reduced number of samples. A cluster is a group of data points grouped together because of certain similarities. This algorithm does this by performing K-means clustering on the majority class, low_risk, and then creates new data points which are averages of the coordinates of the generated clusters.

Once the data were balanced and trained, ClusterCentroids undersampling gave the following scores:

### 4. SMOTEENN (Combination of Over and Under Sampling) and Logistic Regression
The SMOTEENN algorithm is a combination of SMOTE and Edited Nearest Neighbor (ENN) algorithms. In simple terms, SMOTEENN randomly oversamples the minority class (high_risk) and undersamples the majority class (low_risk)

Once the data were balanced and trained, the SMOTEEN algorithm gave the following scores:


### 5. Balanced Random Forest Classifier

The Balanced Random Forest Classifier is an ensemble method where each tree in the ensemble is built from a sample drawn with replacement (bootstrap sample) from the training set. Instead of using all the features, a random subset of features is selected, which further randomizes the tree. As a result, the bias of the forest increases slightly, but since the less correlated trees are averaged, its variance decreases, which results in an overall better model.

Once the data were balanced and trained, the balanced random forest algorithm gave the following scores:



### 6. Easy Ensemble AdaBoost Classifier

The Easy Ensemble AdaBoost Classifier combine multiple weak or low accuracy models to create a strong, accurate models. This algorithm uses one-level decision trees as weak learners that are added to the ensemble sequentially. This is an iterative process, so each subsequent model attempts to correct predictions made by the previous model in the sequence.

Once the data were balanced and trained, the Easy Ensemble AdaBoost Classifier algorithm gave the following scores:
