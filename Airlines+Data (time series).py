# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:45:31 2022

@author: ashwi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statsmodels.api as smf


df = pd.read_excel("Airlines+Data.xlsx")
df
df.shape
df.dtypes
df.info()

# finding the null values
df.isnull()
df.isnull().sum()
# there is no null values

df.describe()

df["Passengers"].plot()
# Trend,seasonal,residual,observed graphs 
import statsmodels.api as smf
from statsmodels.tsa.seasonal import seasonal_decompose
seasonal_ts_add = smf.tsa.seasonal_decompose(df["Passengers"],extrapolate_trend='freq',period=10)
seasonal_ts_add.plot()

# Extracting the month names from the Month
df
df["Month"] = pd.to_datetime(df["Month"])
df["Months"] = df["Month"].dt.strftime("%b")
df.head()

# extracting the dummies from the column Month
month_dummies = pd.DataFrame(pd.get_dummies(df["Months"]))

# concating the data 
df1 = pd.concat([df,month_dummies],axis=1)
df1.head()

#
df1['t'] =np.arange(1,97)
df1['t_squared'] = df1['t']*df1['t']
df1['log_Passengers'] = np.log(df1['Passengers'])
df1.head()


# Data spliting like Train and Test
Train = df1.head(75)
Test = df1.tail(25)

import statsmodels.formula.api as smf

# Linear Model

linear_model = smf.ols("Passengers~t",data=Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
print("RMSE Linear:",rmse_linear)


# Exponential Model

exp_model = smf.ols("log_Passengers~t",data=Train).fit()
pred_exp = pd.Series(exp_model.predict(pd.DataFrame(Test['t'])))
rmse_exp =np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_exp)))**2))
print("RMSE Exponential: ",rmse_exp)

# Quadratic Model

quad_model = smf.ols("Passengers~t+t_squared",data=Train).fit()
pred_quad = pd.Series(quad_model.predict(Test[['t','t_squared']]))
rmse_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_quad))**2))
print("RMSE Quadratic: ",rmse_quad)

# Additive Seasonality

add_sea = smf.ols("Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec",data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))  
print("RMSE Additive Seasonality: ",rmse_add_sea)

# Additive seasonality with Quadratic Trend

add_sea_quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_ass_sea_quad = pd.Series(add_sea_quad.predict(Test))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_ass_sea_quad))**2))
print('RMSE Additive Seasonality Quadratic: ',rmse_add_sea_quad)

# Multiplicative Seasonality 

mul_sea = smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_mul_sea = pd.Series(mul_sea.predict(Test))
rmse_mul_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_mul_sea)))**2))
print("RMSE Multiplicative Seasonality: ",rmse_mul_sea)


# Multiplicative Aditive Seasonality

mul_add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=Train).fit()
pred_mul_add_sea = pd.Series(mul_add_sea.predict(Test))
rmse_mul_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_mul_add_sea)))**2))
print("RMSE MUltiplicative Additive Seasonality: ",rmse_mul_add_sea)

# Creating table of RMSE values
data1 = {"MODEL":pd.Series(["RMSE Linear","RMSE Exponential","RMSE Quadratic","RMSE Additive Seasonality","RMSE Additive Seasonality Quadratic","RMSE Multiplicative Seasonality","RMSE MUltiplicative Additive Seasonality"]),
         "RMSE Value":pd.Series([rmse_linear,rmse_exp,rmse_quad,rmse_add_sea,rmse_add_sea_quad,rmse_mul_sea,rmse_mul_add_sea])}

table_rmse =pd.DataFrame(data1)
table_rmse

# Creating the new data to predict the future prediction
data = [['2003-01-01','Jan'],['2003-02-01','Feb'],['2003-03-01','Mar'],['2003-04-01','Apr'],['2003-05-01','May'],['2003-06-01','Jun'],['2003-07-01','Jul'],['2003-08-01','Aug'],['2003-09-01','Sep'],['2003-10-01','Oct'],['2003-11-01','Nov'],['2003-12-01','Dec']]

forecast = pd.DataFrame(data,columns=['Date','Months'])
forecast

# creating dummies, t and t-squared values

dummies = pd.DataFrame(pd.get_dummies(forecast['Months']))
forecast1 = pd.concat([forecast,dummies],axis=1)
forecast1['t'] = np.arange(1,13)
forecast1['t_squared'] = forecast1['t']*forecast1['t']
print("\nAfter Dummy, T and T-Square\n\n",forecast1.head())

# forcasting using Multiplicative Aditive Seasonality
model_full = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec',data=df1).fit()
pred_new = pd.Series(model_full.predict(forecast1))
forecast1['Forecasted_log'] = pd.Series(pred_new)
forecast1['Forecasted_Passengers'] = np.exp(forecast1['Forecasted_log'])

# Final prediction of next 12 months
final_predict = forecast1.loc[:,['Date','Forecasted_Passengers']]
final_predict











