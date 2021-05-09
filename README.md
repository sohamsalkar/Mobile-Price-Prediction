# Mobile-Price-Prediction


### Table of Content
1. [Abstract](#abstract)
2. [Implementation](#implementation)
3. [Analysis](#analysis)
4. [Instructions](#instructions)
5. [Screenshots](#screenshots)


## Abstract
In this Modern Era, Smartphones are an integral part of the lives of human beings. When a smartphone is purchased, many factors like the Display, Processor, Memory, Camera, Thickness, Battery, Connectivity, and others are taken into account. One factor that people do not consider is whether the product is worth the cost. To solve the problem we will develop a model that will predict the approximate price of the new smartphone on the basis of given data. Battery power,   clock_speed, Front Camera, Mobile Depth, Number of cores, Resolution Height, Resolution Width, RAM, Screen Height, Screen Width, Bluetooth, 4G/3G Supported, Touch Screen,  etc

## IMPLEMENTATION
Multiple regression techniques were used in the data mining process for predicting the price of a mobile phone. This approach was used because it can provide a broader look and understanding of the all parameters All data mining implementation and processing in this study was done using jupyter Notebook.
1. Logistic Regression
2. K-Nearest Neighbor
3. Random Forest Regression
4. Support Vector Clustering
5. Decision Tree

## Analysis
As a final analysis, it was obviously noticed that some algorithms worked better with the dataset than others, in detail, *Logistic Regression Algorithm had the best accuracy of 97.01%*, which was significantly more than the expected (default model) accuracy, *SVC was next with 94.07%* respectively, and the least accurate was *KNN with 64.80%*.

## Instructions
to run the front end project, in terminal simply run <br>
`python MobilePricePrediction.py`

## Screenshots
Same price predicted by all the three algorithms.
![SamePrediction](https://github.com/sohamsalkar/Mobile-Price-Prediction/blob/main/_Screenshots/sameres.png)
Different price predicted by DT but same by KNN and LR from which LR gives the maximum accuracy.
![DifferentPrediction](https://github.com/sohamsalkar/Mobile-Price-Prediction/blob/main/_Screenshots/diffres.png)
