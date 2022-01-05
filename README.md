# STUDY PROJECT REPORT : Drug Consumption 💊📊

[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](http://forthebadge.com)

## Overview 

This is our final Project for the "Python for Data Analysis" course. The goal for this project was to understand the implications of manipulating a data set, from data processing and data visualization to machine learning. With **Chloé Coursimault** we worked on the Drug Consumption Data Set from the UCI Machine Learning repository
[Drug Consumption Data Set from the UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29). 


<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/92330168/147699118-4588c0b4-1830-44f8-9e42-748ce32a66c0.jpg">
</p>

There are 1885 responders in the database. Twelve characteristics are known for each respondent: NEO-FFI-R (neuroticism, extraversion, openness to experience, agreeableness, and conscientiousness), BIS-11 (impulsivity), and ImpSS (sensation seeking), as well as education, age, gender, country of residence, and ethnicity, are used to assess personality.

---

Participants were also asked about their use of 18 legal and illegal drugs (alcohol, amphetamines, amyl nitrite, benzodiazepines, cannabis, chocolate, cocaine, caffeine, crack, ecstasy, heroin, ketamine, legal highs, LSD, methadone, mushrooms, nicotine, and volatile substance abuse) as well as one fictitious drug (Semeron) that was used to identify over-claimers.

They had to choose between 'Never Used,' 'Over a Decade Ago,' 'Last Decade,' 'Last Year,' 'Last Month,' 'Last Week,' and 'Last Day' for each drug.



## What do you need ? 🎒

This project uses :

<img title="Python" alt="python" width="40px" src="https://img.icons8.com/color/32/000000/python--v1.png">|<img title="Colab" alt="Colab" width="40px" src="https://colab.research.google.com/img/colab_favicon_256px.png">|
|--|--|

To download the dataset, click on this [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00373/).

To have a more detailed description of the dataset, click on this [link](https://github.com/heloise-de-castelnau/Drug_Consumption_Project/blob/main/AttributesInfo.md).

For this project we will mainly use the following libraries :
```python
#Import Packages
# essential libraries

# storing and anaysis
import numpy as np
import pandas as pd
import random 
import math

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from pywaffle import Waffle


#Machine learning cleaning and boosting
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#models 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
#metrics
from sklearn.metrics import f1_score #compromis entre rappel et précision
from sklearn.metrics import recall_score #taux de vrais +
from sklearn.metrics import precision_score #proportion de prédictions correctes parmi les points que l’on a prédits positifs.
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

```

## Key Topics 🔍
  
To carry out the study of our dataset, we splitted our project into different points : 

  

* Dataset exploration & analysis
  * Drugs : proportions & representations
  * Demographic data 
  * Personality traits
* Data pre-processing
  * Biased variables
  * Binary outputs
  * Data cleaning
* Machine Learning prediction
  * Logistic Regression
  * Linear SVC
  * Gaussian NB
  * Random Forest
  * Decision Tree
  * KNN
  * XGBOOST
  * SVM
  * Neural Network
* Conclusion
  * ML algorithms comparisons
  * Wich one is the best model ? 🥇


---

Throughout this study we wanted to reply to this question : **" Has the individual with the characteristics X used illegal drugs lately (up to one year)? "**.
In order to reply to this question, we splitted the drug consumption in two categories, if the drog was used up to last year, the individual is considered as an user, otherwise it's a non-user. We performed multiples machine learning and found at that the most performant on were : *XGBOOST, Linear SVC & logistic regression*


## Django 🦍
<p>
This project is also available in an Application form through Django ! Don't hesitate to have a look at our presentation video for a little overview 🎥 :)  
  
**Please keep in mind that the goal of our system is to enable specialists such as psychologists and others to detect current or future users, so the metrics used for the search through the API should be filled by profesionnal and therefore as written from the original dataset.**
  
(Click on the picture ⤵️ ) 
</p>
  
[![Watch the video](https://images.assetsdelivery.com/compings_v2/giamportone/giamportone1904/giamportone190400156.jpg)](https://www.youtube.com/watch?v=2gHpapC8ZMw)

<p>
  
If you want to open the Api on your computer follow thoses request :
  
* Go to the directory where the manage.py file is stocked
* Check that python is installed as a environment variable
* Check that the module pandas, scikitlearn & xgboost are installed
* Check that Django is installed
  
Then for the Command Prompt : 
 
```python
cd...
py -m pip install pandas
py -m pip install scikit
py -m pip install xgboost  
py manage.py runserver  
```  

  
</p>



## About the project 🤝

This project was realized with **Chloé Coursimault**, a Data & AI Student at ESILV engineering school.
<p align="left">
</p>

## Helpful Links

* [UCI Dataset source](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)

