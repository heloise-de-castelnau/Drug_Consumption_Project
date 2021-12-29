# STUDY PROJECT REPORT : Drug Consumption 💊📊

[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](http://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](http://forthebadge.com)

## Overview 

This is our final Project for the "Python for Data Analysis" course. The goal for this project was to understand the implications of manipulating a data set, from data processing and data visualization to machine learning. With **Chloé Coursimault** we worked on the Drug Consumption Data Set from the UCI Machine Learning repository. 


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
<p>
To have a more detailes description of the dataset, click on this [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00373/).

For this project we will mainly use the following libraries :
```python
#Import Packages
# essential libraries
import json
import random
from urllib.request import urlopen

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

# converter
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()   
from sklearn.preprocessing import LabelEncoder

# hide warnings
import warnings
warnings.filterwarnings('ignore')

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
  * Biaised variables
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

Throughout this study we wanted to reply to this question : *" Has the individual with the characteristics X used illegal drugs lately (up to one year)? "*.
In order to reply to this question, we splitted the drug consumption in two categories, if the drog was used up to last year, the individual is considered as an user, otherwise it's a non-user. Then we got rid of biaised variables such as *Country* and *Ethinicity* because they were composed on unevenly represented categories (80% of white from UK).
We performed multiples machine learning and found at that the most performant was : *Random forest & logistic regression*

<p>
This project is also available in an Application form through Django ! Don't hesitate to have a look :) 
</p>

<p>

</p>



## About the project 🤝

This project was realized with **Chloé Coursimault**, a Data & AI Student at ESILV engineering school.
<p align="left">
</p>

## Helpful Links

* [UCI Dataset source](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)

