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

Initially, Data contained null and string values: 

<img width="996" alt="Screen Shot 2022-02-20 at 10 08 12 AM" src="https://user-images.githubusercontent.com/92277581/154852168-32b40a02-f9c6-416c-933a-e318f1b512de.png">

Credit card data from LoanStats_2019Q1.csv was cleaned prior to implementing machine learning techniques. Null columns and rows were dropped, interest rates were converted to numerical values, and target (y-axis) columns were converted to low_risk and high_risk based on their values.

<img width="980" alt="Screen Shot 2022-02-20 at 10 15 26 AM" src="https://user-images.githubusercontent.com/92277581/154852442-9aae5c1c-2b23-4edf-ac85-f42abac4fc4e.png">

Once the data was cleaned, it was split into training and testing categories, which resulted in four sets of data:

- X_train
- X_test
- y_train
- y_test

A random_state of 1 was used across all models to ensure reproducible output.

The balance of low_risk and high_risk is unbalanced, but this was expected as credit risk is an inherently unbalanced classification problem, since good loans easily outnumber risky loans.

<img width="452" alt="Screen Shot 2022-02-20 at 10 19 31 AM" src="https://user-images.githubusercontent.com/92277581/154852617-1c80328d-8379-4dc5-ac21-a2c6ae810d68.png">

### 1. Naive Random Oversampling and Logistic Regression

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. Once the datasets were balanced, the model trained the data, which is where the algorithm analyzes the data and attempts to learn patterns in the data.

Naive random oversampling on this data gave the following scores:

<img width="712" alt="Screen Shot 2022-02-20 at 10 21 13 AM" src="https://user-images.githubusercontent.com/92277581/154852672-605652c5-6ef6-4008-b87c-b21e72fad5c2.png">


### 2. SMOTE Oversampling and Logistic Regression

The synthetic minority oversampling technique (SMOTE) is another oversampling approach where the minority class is increased. Unlike other oversampling methods, SMOTE interpolated new instances, that is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

Once the data were balanced and trained, SMOTE oversampling gave the following scores:
<img width="727" alt="Screen Shot 2022-02-20 at 10 21 57 AM" src="https://user-images.githubusercontent.com/92277581/154852700-40cb0539-5043-43d4-82e6-af7d8cd7d15a.png">



### 3. Cluster Centroids Undersampling and Logistic Regression

The ClusterCentroid algorithm provides an efficient way to represent the data cluster with a reduced number of samples. A cluster is a group of data points grouped together because of certain similarities. This algorithm does this by performing K-means clustering on the majority class, low_risk, and then creates new data points which are averages of the coordinates of the generated clusters.

Once the data were balanced and trained, ClusterCentroids undersampling gave the following scores:
<img width="743" alt="Screen Shot 2022-02-20 at 10 22 47 AM" src="https://user-images.githubusercontent.com/92277581/154852728-8ece6858-a389-4ba1-840b-6a5f8e73169d.png">


### 4. SMOTEENN (Combination of Over and Under Sampling) and Logistic Regression
The SMOTEENN algorithm is a combination of SMOTE and Edited Nearest Neighbor (ENN) algorithms. In simple terms, SMOTEENN randomly oversamples the minority class (high_risk) and undersamples the majority class (low_risk)

Once the data were balanced and trained, the SMOTEEN algorithm gave the following scores:

<img width="754" alt="Screen Shot 2022-02-20 at 10 23 41 AM" src="https://user-images.githubusercontent.com/92277581/154852765-b75bccca-c34f-4941-b7b4-1369213c0fb2.png">

### 5. Balanced Random Forest Classifier

The Balanced Random Forest Classifier is an ensemble method where each tree in the ensemble is built from a sample drawn with replacement (bootstrap sample) from the training set. Instead of using all the features, a random subset of features is selected, which further randomizes the tree. As a result, the bias of the forest increases slightly, but since the less correlated trees are averaged, its variance decreases, which results in an overall better model.

Once the data were balanced and trained, the balanced random forest algorithm gave the following scores:

<img width="775" alt="Screen Shot 2022-02-20 at 10 25 57 AM" src="https://user-images.githubusercontent.com/92277581/154852891-3838de56-e3e4-400f-8886-573824af99c8.png">


### 6. Easy Ensemble AdaBoost Classifier

The Easy Ensemble AdaBoost Classifier combine multiple weak or low accuracy models to create a strong, accurate models. This algorithm uses one-level decision trees as weak learners that are added to the ensemble sequentially. This is an iterative process, so each subsequent model attempts to correct predictions made by the previous model in the sequence.

Once the data were balanced and trained, the Easy Ensemble AdaBoost Classifier algorithm gave the following scores:

<img width="749" alt="Screen Shot 2022-02-20 at 10 26 33 AM" src="https://user-images.githubusercontent.com/92277581/154852923-3fc717eb-d649-4b9d-a95f-a8ed48ea2e9a.png">

## Summary:
This analysis is trying to find the best model that can detect if a loan is high risk or not. Becasue of that, we need to find a model that lets the least amount of high risk loans pass through undetected. From the results above we can see that first four models don’t do well based off the accuracy scores. Other two models did better. 

Out of the six supervised machine learning algorithms tested, Easy Ensemble AdaBoost CLassifier performed the best overall. It had a balanced accuracy score, along with high precision and recall scores. It also had a high specificity score, which means this algorithm correctly determined actual negatives 91% of the time, and a high F1 score. This means the harmonic mean of precision and recall were 0.97 out of 1.0.
