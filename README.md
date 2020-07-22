# AQI-prediction

Project Title: Air Quality Index Prediction of Delhi

Description: Since the Air pollution is increasing day by day,And there are various gases which constitute in the AQI of a region, So, we used machine learning to predict the AQI of a region.

Dataset: The dataset is fetched from kaggle. The dataset includes AQI of different cities of India collected for a period of time. The dataset has levels of different gases ,like NH3,NO,NO2,CO,SO2,O3 etc. , and the AQI along with it.

Project Detail: Since the dataset was containing Nan values, so we replaced all Nan values with median of the correspoding columns.Since the data was from many Indian cities so we filtered the data and took only data corresponding to Delhi for this project. Then we splited the dataset into training and testing sets with a ratio of 80%-20%. Then we used different machine learning models to predict the AQI.Since it was a Regression task so the models used were Linear Regression, Random Forest ,Artificial Neural Network. We implemented Linear Regression from scratch and fitted on data.

Outcome: We predicted the AQI of Delhi with taking the levels of different gases present in atmosphere.
