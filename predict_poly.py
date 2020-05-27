# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:48:47 2020

@author: rashmibh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import timedelta

from sklearn.linear_model import LinearRegression
# Importing the INDIA dataset
dataset_india = pd.read_csv('covid_19_india.csv', parse_dates=[0],date_parser=lambda x: datetime.strptime(x, '%d/%m/%y'),usecols=[1,8])
#dataset['id'] = [random.randint(0,1000) for x in range(dataset.shape[0])]
gk_india = dataset_india.groupby("Date", as_index=False)["Confirmed"].sum() 

# Importing the US dataset
dataset_us = pd.read_csv('us-counties.csv', parse_dates=[0],date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'),usecols=[0,4])
#dataset['id'] = [random.randint(0,1000) for x in range(dataset.shape[0])]
gk_us = dataset_us.groupby("date", as_index=False)["cases"].sum() 


# Importing the India dataset
dataset_kar = pd.read_csv('covid_19_india.csv', parse_dates=[0],date_parser=lambda x: datetime.strptime(x, '%d/%m/%y'),usecols=[1,3,8])
#dataset['id'] = [random.randint(0,1000) for x in range(dataset.shape[0])]
gk_kar = dataset_kar.groupby(["Date","State/UnionTerritory"], as_index=False)["Confirmed"].sum() 
gk_kar= gk_kar.loc[gk_kar['State/UnionTerritory'] == 'Karnataka']
#gk_kar.to_csv('stateResult.csv') 
gk_kar= gk_kar.iloc[:,[0,2]]
gk_kar=gk_kar.reset_index(drop=True)

country="India"
dataset=dataset_india
gk=gk_india
case_row=2 #one for country

gk.insert(0, 'New_ID', range(1, 1 + len(gk)))

X = gk.iloc[:,0].values.reshape(-1,1)
y = gk.iloc[:,case_row].values.reshape(-1,1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/4, random_state = 0)


#-----------polynomial
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

original_len = gk.shape[0]
in_val=8

from sklearn.preprocessing import StandardScaler

#We can try different degrees
for j in range(in_val,9):
    poly_reg = PolynomialFeatures(degree = j)
    X_poly = poly_reg.fit_transform(X_train)
    poly_reg.fit(X_poly, y_train)
    
    scalerX = StandardScaler().fit(X_poly)
    X_scaled = scalerX.transform(X_poly)

    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_scaled, y_train)
    y_pred = lin_reg_2.predict(scalerX.transform(poly_reg.fit_transform(X_test)))
       
    #Save the prediction graph with no of days
    
    X_grid = np.arange(min(X_test), max(X_test), 0.1)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(X_train, y_train, color = 'green')
    plt.plot(X_grid, lin_reg_2.predict(scalerX.transform(poly_reg.fit_transform(X_grid))), color = 'purple')
    plt.scatter(X_test, y_test, color = 'red')
    plt.scatter(X_test, lin_reg_2.predict(scalerX.transform(poly_reg.fit_transform(X_test))), color = 'blue')
   
    plt.title('Total Cases vs Date (Training set)')
    plt.xlabel('Date')
    plt.ylabel('Total no of Cases')
    figure = plt.gcf()  # get current figure
    figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
    plt.savefig(country+"_Poly_"+str(j)+'_Predict.png', bbox_inches='tight') # bbox_inches removes extra white spaces
    plt.clf()
  
    
    count_last=0
    for i in range(1,original_len):
        predicted_val = (lin_reg_2.predict(scalerX.transform(poly_reg.fit_transform([[i]]))))
        gk.loc[i-1,"P_"+str(j)]=int(predicted_val)
    
    str_last=str(gk['Date'][original_len-1].date())
    count_last=original_len-1
    #row_df = pd.DataFrame(columns=["New_ID","Date","Confirmed","Predicted"])
    for i in range(1,15):
        predicted_val = (lin_reg_2.predict(scalerX.transform(poly_reg.fit_transform([[count_last+i]]))))
        #row_df.loc[0]=[count_last+i, datetime.strptime(str_last, '%Y-%m-%d')+timedelta(days=i),-1,predicted_val]
        #gk = pd.concat([gk, row_df], ignore_index=True)
        if( j == in_val ):
            gk.loc[count_last+i-1]=[count_last+i, datetime.strptime(str_last, '%Y-%m-%d')+timedelta(days=i-1),-1,-1]
            gk.loc[count_last+i-1,"P_"+str(j)]=int(predicted_val)
        else:
            gk.loc[count_last+i-1,"P_"+str(j)]=int(predicted_val)
 
#Save the prediction result in excel
gk.to_csv(country+'_Poly_Prediction.csv') 


#Save the prediction graph with date
X_date = gk.iloc[:,1].values.reshape(-1,1)
y = gk.iloc[:,2].values.reshape(-1,1)
y_pred = gk.iloc[:,3].values.reshape(-1,1)

axes = plt.gca()

date_form = mdates.DateFormatter("%d-%m-%Y")
axes.xaxis.set_major_formatter(date_form)

locator = mdates.DayLocator(interval=3)
axes.xaxis.set_major_locator(locator)

X_date_h=[]
X_date_p=[]
y_h=[]
y_p=[]

for i in range (0,len(y)):
    if( y[i]!=-1):
        X_date_h.append(X_date[i])
        X_date_p.append(X_date[i])
        y_h.append(y[i])
        y_p.append(y_pred[i])
    else:
        X_date_p.append(X_date[i])
        y_p.append(y_pred[i])
        
      
plt.plot(X_date_p, y_p, color = 'red', label='Predicted Values')
plt.scatter(X_date_h, y_h, color = 'purple', label='Data Points')

plt.gcf().autofmt_xdate()

plt.title('Total Covid-19 Confirmed Cases vs Date in '+str(country)+' --- Polynomial Regression',fontsize=20, fontweight='bold')
plt.xlabel('Date',fontsize=16, fontweight='bold')
plt.ylabel('Total Covid-19 Confirmed Cases in '+str(country),fontsize=16, fontweight='bold')
plt.legend()

figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
plt.savefig(country+"_Poly_"+str(in_val)+'_Date.png', bbox_inches='tight') # bbox_inches removes extra white spaces
plt.clf()







