# CLASSIFICATION OF BREAST CANCER TUMOURS

## Abstract
Background: Breast cancer is the second most severe cancer of all of the cancers that exist. Studies show that 1 in 8 U.S. women develop breast cancer throughout their life. Early detection and diagnosis of breast cancer is crucial to ensure that proper treatment can be received at the right time. When diagnosing breast cancer, a suspicious lump is examined to determine whether it is cancerous or not. This is done by checking certain characteristics of the lump like its area, radius, perimeter, etc. Based on these, the oncologist can determine whether the lump is cancerous and if so, start treatment immediately to ensure that it does not spread. Objective: There is a need to predict whether cells are cancerous or not with high sensitivity. Machine Learning not only offers an alternative approach to the status quo but could also be able to overcome the limitations of the current methods and improve the accuracy and sensitivity of prediction. The overall objective is to develop a model using machine learning algorithms to predict whether a lump or tumor is cancerous or not with high sensitivity. Methods: We have used a Breast Cancer dataset from the University of Wisconsin Hospitals which has been made available on Kaggle. After many preprocessing steps (Splitting data into test and train set, feature construction and feature selection), we trained different models using supervised learning methods like Logistic Regression, Random Forest, Neural Networks, XGBoost and Bagging using the training set. We then evaluated the performance of these models on the testing set using various evaluation metrics like accuracy and sensitivity. Results: overall a high sensitivity model was able to be developed, with the highest sensitivity from a multi-Layered Perceptron model providing a sensitivity of 0.962. The lowest scoring evaluation metric as the specificity, which was concluded to be acceptable as the focus of the study is maximizing the true positive rate.

## Introduction
Breast cancer is the most common type of cancer in women and has the second highest mortality rate. This type of cancer occurs almost entirely in women; however, men can get breast cancer too (American Cancer Society, 2019). Breast cancer cells usually form a tumor that can often be seen on an x-ray or felt as a protruding lump. After a suspicious lump is found, imaging procedures are performed, and the images are analyzed by a radiologist to search for tumors in breast tissue. Once found, tumors are classified as either malignant or benign which determines if the patient is diagnosed with breast cancer or not. We propose the integration of machine learning algorithms to predict whether a lump is cancerous or not based on its shape characteristics. The integration of machine learning can be especially useful as it can provide opportunities to make consistent, accurate results. Models can be developed using supervised learning methods like Logistic Regression, Neural Networks, and Decision Trees. These models can be trained on data which contain the shape features of different tumors and the corresponding , and can then classify new tumors as malignant or benign based on the attributes of the tumor. In this study, the overall objective is as follows:

**Develop a model that is able to correctly predict when tumors are malignant or benign with high sensitivity**.

The reason for demanding a high sensitivity from the model is due to the fact that the repercussions of the model indicating that a malignant (cancerous) tumor is benign (non-cancerous) are more severe than if a benign tumor was classified as malignant. Early detection of breast cancer can be the difference between life and death., so it is very important to have a model that can accurately classify tumors as malignant with very low error rates. In this project, we are going to predict whether a person has cancer or not given the characteristics of the tissue being examined.

## Data
### Dataset
The features of the dataset are listed as follows:

- mean radius: The mean of distances from the center to points on the perimeter. (Continuous)
- mean_texture: The standard deviation of grayscale values of the images. (Continuous)
- mean_perimeter: The perimeter of the core of the tumor. (Continuous)
- mean_area: The mean area of the tumor. (Continuous)
- mean_smoothness The mean of local variations in radius lengths. (Continuous)
- diagnosis: Diagnosis of the tissue being examined. (Discrete, 0 if cancerous and 1 otherwise)

They were computed from the digitized image of a Fine Needle Aspiration (biopsy) of a breast mass. The Breast Cancer Prediction Dataset: v.2 provided by [Kaggle](https://www.kaggle.com/merishnasuwal/breast-cancer-prediction-dataset) (2018). From these images, the shapes and other characteristics of the tissue under observation are recorded. The dataset contains 596 rows and 6 columns (5 describing the characteristics of the tissue under observation and one column corresponding to the diagnosis).

### Data Preprocessing
Since certain models cannot handle missing values internally and need to explicitly be told what to do with them, we first checked for the presence of missing or N.A values. We then checked for the presence of duplicate rows. Due to the nature of our dataset, duplicate rows could be a cause for concern as it is highly unlikely that two tumor cells are exactly the same with respect to each characteristic feature. The presence of duplicates may indicate a repetition of the same data instance and might need to be removed. Following that, the next phase was to encode the categorical data.

Lastly, we split the data into a train set, for training the model and a test set which we used to evaluate the model. Due to the small portion of data, we did not consider validation. Instead, we used train set for this purpose.

### Feature Construction
In order to better train our model, we constructed two additional geometry-based features based on the correlated features. These are **fractal dimension** and the **compactness** of the growths.

### Response Variable
Understanding the response variable is vital in understanding what kind of problem we are dealing with. The response variable is binary class with 0 (benign) and 1 (malignant).

We then looked at the relationships among the features in our dataset to find the most positively or negatively correlated features using a correlation matrix.

### Feature Selection
In this part, we studied the different features to find the ones that are most useful and the ones that did not add value to our model. We chose to use backward selection with cross validation using accuracy and precision as performance metrics, to identify which features are important and which are not.

### Feature Transformation
Next, we studied the impact of transformed data on our model(s). In this case, we created pipelines to standardize the data, and then transform it into high-order terms. After that, we employed cross-validation to estimate the accuracy of each model with transformed data.

## Methods
### Predictive Models
In this part, 6 predictive models were trained using the training set.

- Logistic Regression
- Decision Tree
- Bootstrap Aggregating or Bagging
- Random Forest
- Extreme Gradient Boosting (XGBoost)
- Multi-Layer Perceptron

## Model Evaluation
In evaluating our model, we turned to the confusion matrix to calculate the accuracy, recall, precision, sensitivity, and specificity metrics, and the ROC curve, especially the AUROC to evaluate the ability of the model to separate between the target labels.
