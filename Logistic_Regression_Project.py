

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


ad_data=pd.read_csv('advertising.csv')

ad_data.head()



ad_data.info()


ad_data.describe()


sns.histplot(ad_data, x='Age', bins=30)


sns.jointplot(x='Age', y='Area Income', data=ad_data)


sns.jointplot(x='Age', y='Area Income', data=ad_data, kind='kde')


sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data, color="green")


sns.pairplot(data=ad_data, hue='Clicked on Ad')



X_train, X_test, y_train, y_test=train_test_split(ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage', 'Male']], ad_data['Clicked on Ad'], test_size=0.3, random_state=101)


ad_data.columns


logm=LogisticRegression()




logm.fit(X_train, y_train)



predictions=logm.predict(X_test)




from sklearn.metrics import classification_report



print(classification_report(y_test,predictions))

