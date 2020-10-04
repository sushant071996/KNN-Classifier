## Data Analysis and Visualisation to predict Purchased or not (Use of KNN Classifier)

In this project I'm trying to analyze and visualize the Output variable i.e Purchased. There are 3 input variable i.e Gender, Age and Estimated Salary depending on these input variable we can predict whether a specific user purchased or not Purchased from a shop 

## Data Description

    Gender - male or female
    
    Age - Age of User visiting shop.
    
    Estimated Salary - Salary of a user visiting shop.
    
    Purchased - 0=User does not purchased 1=User did purchased
   
## Process

    1)Read the data set
    
    2) EDA
    
    3)check multicollinearity between input feature
    
    4)check relation between continous input feature and categorical output feature (Using Anova Test)
    
    5)check relation bewteen categorical input feature and categorical output feature (Using Chi-square Test)
    
    6)Build the Model
    
    7)Test the Model
    
    8)Evaluate the model
    
## Softwares and Libraries Used:

       - Anaconda Distribution
	- Jupyter Notebook
	
	- Numpy
	- Pandas
	- Matplotlib
	- Seaborn
    - sklearn 
    - statsmodels
    - warnings

## Importing the Modules:

    import pandas as pd 
    import os as os
    import numpy as np

    #Visuals
    import matplotlib.pyplot as plt
    import seaborn as sns
  
    # To split data
    from sklearn.model_selection import train_test_split

    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    # Evalution 
    from sklearn.metrics import confusion_matrix
    
## Analysis:

1)Univariant Analysis

![](Figures/histbox.png)

2)Bivariant Analysis

![](Figures/scatter.png)

3)Correlation Using HeatMap

![](Figures/heatmap.png)

5)Anova Test

![](Figures/anova_2.png)

![](Figures/anova_1.png)

  We will use One Way Anova to check asscocaition between Age and Purchased, Then between Estimated Salary and Purchased.

6)Box Plot

![](Figures/boxplot.png)

![](Figures/boxplot_1.png)

  Thus both box plot also shows tha mean is different for both age and estimated salary thus both feature are important and should be consider while model building
  
7)Chi-Sqaure Test : 
  To check Relation between two categorical feature
  
8)Scatterplot with approx boundry line:  

![](Figures/scatterwithboundary_1.png)

9)KNN Analysis:

![](Figures/KNN.png)

10)Build Knn Model and Evaluate:

![](Figures/confusion_knn.png)

11)Boundary via KNN classifier:

![](Figures/knnfigure.png)

## Conclusing Statement

  We can conclude that Age and Estimated Salary after standardizing gives an accuracy of 95% for train and 88% for test

## Note

  You can also view the code in HTML format
