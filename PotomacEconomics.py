#!/usr/bin/env python
# coding: utf-8

# In[88]:


#### Potomac Economics Data Analytics
## Anil Niraula

#!pip install matplotlib.pyplot
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy

#### Pull all electricity data 
##(Custom FUNTION: pullAnnualData())


def pullAnnualData(year):
 link = 'https://marketplace.spp.org/file-browser-api/download/generation-mix-historical?path=%2FGenMix_'+str(year)+'.csv'
 req =   requests.get(link)
#req = requests.get('https://marketplace.spp.org/file-browser-api/download/generation-mix-historical?path=%2FGenMix_2020.csv', verify = False)
 req_content = req.content

 csv = open('C:\\temp\\data.csv', 'wb')
 csv.write(req_content)
 data = pd.read_csv('C:\\temp\\data.csv')
 return(data)
# Get data from https://marketplace.spp.org/pages/generation-mix-historical

#### Pull data for year 2020

data = pullAnnualData(year = 2020)
#Print all column names
#print(list(data))

#### Rename columns
#cols = ['first_interview', 'second_interview']
#inter ['total_2'] = inter[cols].sum(axis=1)

data.rename({'GMT MKT Interval': "DatetimeEST"}, axis=1, inplace=True)
data.rename({' Load': "Load"}, axis=1, inplace=True)

#### Convert DatetimeESt to datetime format

data['Date'] = pd.to_datetime(data['DatetimeEST'])#removed '.dt.date'

#### Combine multiple sources into single columns
data['Coal'] = data[' Coal Market'] + data[' Coal Self']#combine coal sources
data['Diesel'] = data[' Diesel Fuel Oil Market'] + data[' Diesel Fuel Oil Self']#combine
data['Hydro'] = data[' Hydro Market'] + data[' Hydro Self']#combine
data['Gas'] = data[' Natural Gas Market'] + data[' Gas Self']#combine
data['Nuclear'] = data[' Nuclear Market'] + data[' Nuclear Self']#combine
data['Waste_Disposal'] = data[' Waste Disposal Services Market'] + data[' Waste Disposal Services Self']#combine
data['Waste_Heat'] = data[' Waste Heat Market'] + data[' Waste Heat Self']#combine
data['Solar'] = data[' Solar Market'] + data[' Solar Self']#combine
data['Wind'] = data[' Wind Market'] + data[' Wind Self']#combine
data['Other'] = data[' Other Market'] + data[' Other Self']#combine

#### Remove original columns
data = data.drop([
' Coal Market', ' Coal Self',
' Diesel Fuel Oil Market', ' Diesel Fuel Oil Self', 
' Hydro Market', ' Hydro Self', 
' Natural Gas Market', ' Gas Self', 
' Nuclear Market', ' Nuclear Self', 
' Solar Market', ' Solar Self', 
' Waste Disposal Services Market', ' Waste Disposal Services Self',
' Wind Market', ' Wind Self', 
' Waste Heat Market', ' Waste Heat Self', 
' Other Market', ' Other Self','DatetimeEST'], axis=1)

#### Filter out January 2021

data = data.loc[(data['Date'] < '1/1/2021')]

#### Group 5-minute data to 1-hour

data = data.groupby(pd.Grouper(key='Date', axis=0, freq='H')).mean()

#print(hourly)

#### Find Daily supply & demand values based on Max Load hour each day
#### 1-hour to 1-day

max_load = data.loc[data.groupby(pd.Grouper(freq='D')).idxmax().iloc[:, 0]]
#max_load = max_load3.sort_values(by = "Date")
#max_load_percent[''] = df.groupby('Date').transform('sum') - df.groupby('Date')

cols = ['Coal','Diesel','Hydro','Gas','Nuclear','Waste_Disposal','Waste_Heat','Solar','Wind','Other']
max_load['total'] = max_load[cols].sum(axis=1)
max_load = pd.DataFrame(max_load)

max_load2 = max_load.loc[:,cols].div(max_load["total"], axis=0)
max_load2['Load'] = max_load['Load']
max_load = max_load2

#df = pd.DataFrame(max_load)
#df.to_csv("Potomac_max_load.csv")

#print(data.max_load2(10))

#### Visualize

data_viz = max_load.drop('Load', axis=1)
plt.figure(figsize=(10, 6))
plot = plt.plot(data_viz)

plt.legend(cols,
           bbox_to_anchor = (1, 1))
plt.xlabel('Hourly Data', fontsize=12)
plt.ylabel('% of Total Generation', fontsize=12)
plt.title('Share of Electric Supply By Source (During 2020 Peak Load Hours)',fontsize=14)
#plt.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))


plt.show()


#### Save Max Load results to csv

#max_load['total'] = max_load[cols].sum(axis=1)
#df = pd.DataFrame(max_load)
#df.to_csv("Potomac_max_load.csv")

####
#### Minimumn Load Hours
####

min_load = data.loc[data.groupby(pd.Grouper(freq='D')).idxmin().iloc[:, 0]]

cols = ['Coal','Diesel','Hydro','Gas','Nuclear','Waste_Disposal','Waste_Heat','Solar','Wind','Other']
min_load['total'] = min_load[cols].sum(axis=1)
min_load = pd.DataFrame(min_load)

min_load2 = min_load.loc[:,cols].div(min_load["total"], axis=0)
min_load2['Load'] = min_load['Load']
min_load = min_load2

#print(data.max_load2(10))

#### Visualize

data_viz = min_load.drop('Load', axis=1)
plt.figure(figsize=(10, 6))
plot = plt.plot(data_viz)

plt.legend(cols,
           bbox_to_anchor = (1, 1))
plt.xlabel('Hourly Data', fontsize=12)
plt.ylabel('% of Total Generation', fontsize=12)
plt.title('Share of Electric Supply By Source (During 2020 Lowest Load Hours)',fontsize=14)
#plt.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))


plt.show()

#### Save Min Load results to csv

#min_load['total'] = min_load[cols].sum(axis=1)
#df = pd.DataFrame(min_load)
#df.to_csv("Potomac_min_load.csv")


# In[ ]:


Anil

