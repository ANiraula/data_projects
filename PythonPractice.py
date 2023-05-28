#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Anil Niraula


"""

1. Transfrom to pd.DataFrame & data.columns
2. Look a missing values w/ data.isnull().all()
3. Look at unique IDs w/ data2['DistrictID'].unique()
4. Look at data.describe() numeric columns
5. Plotly px.box() + facet_col
6. Time series to datatetime + plot w/ px.scatter()
7. Transform* categorical to dummy (np.where(data["col"]>2, 1,0))

#### Load credit data
#https://medium.com/towards-entrepreneurship/importing-a-csv-file-from-github-in-a-jupyter-notebook-e2c28e7e74a5

link = 'https://raw.githubusercontent.com/ANiraula/data_projects/main/credit/data/Credit_Data.csv'
download = requests.get(link).content
data = pd.read_csv(io.StringIO(download.decode('utf-8')))
#print (data.head())
## Look at the data

###### Drop NAs ######
## Potentially
## - Review missing values (share, columns)
## - Replace w/ median (of ts other cluster) or non-missing values from another column?
data = data.dropna().copy()

print(data.head())



txt = "welcome to the jungle"

x = txt.split()

print(x[1]+' '+x[3])

#>> Use map() to apply function multiple time on a list
people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']
 
def split_title_and_name(person):
   title = person.split()[0]
   lastname = person.split()[-1]
   return '{} {}'.format(title, lastname)
 
list(map(split_title_and_name, people))
 
#>> lambda ----------
people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']
def split_title_and_name(person):
   return person.split()[0] + ' ' + person.split()[-1]
list(map(lambda person: person.split()[0] + ' ' + person.split()[-1], people))

#>> For loop —----
people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']
def split_title_and_name(person):
   return person.split()[0] + ' ' + person.split()[-1]
for person in people:
   print(split_title_and_name(person))

#List comprehension —--------
def times_tables():
   lst = []
   for i in range(10):
       for j in range (10):
           lst.append(i*j)
   return lst
 
times_tables() == [j*i for i in range(10) for j in range(10)]

#!pip install matplotlib.pyplot
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, std
from statsmodels.stats.outliers_influence import variance_inflation_factor 
#import statsmodels.api as sm
import seaborn as sn #corr table
import io #pull csv from GitHub
"""

####### Functions #######

#def add(x, y):
#   return x+y

#add(10,20)

#def add(x,y,z=None):
#    if (z==None):
#          return x+y
#    else: 
#          return x+y+z

#print(add(10,20, 1))
"""
def add(x,y,z = None, k = None):
    if z == None and k == None:
         return x+y
    if z == None:
        return x+y+k
    if k == None:
        return x + y + z
    else:
        return x+y+z+k
print(add(10,20, 0,5))
"""
############################

### Type & edit

#x = (1,'x',2,'y',3,'z')
#type(x)
x = [1,'x',2,'y',3,'z']
type(x)
x.append(5)
print(x)

### Loop

for ID in x:
    print(ID)

### 
i = 0
while ( i != len(x)):
    print(x[i])
    i = i +1
    
###
[2,2] + [1,5]    

###
[2] * 5

### Function w/ some description (ise ? or ??)
"""
x = [4,7,1,4]

def find(value, list):

    
    Finds value in a list
    --------
    Returns: Yes/No
    
   
    if value in list:
        print('Yes!')
    else:
        print('No')

#find(4, x)
"""
#np.*load*?


# In[ ]:





# In[5]:


### Strings split
firstname = "Christopher Arthur Hansen Brooks".split(' ')[0]
lastname = "Christopher Arthur Hansen Brooks".split(' ')[-1]

print(firstname)
print(lastname)


# In[2]:


a = 'foo'


# In[ ]:


a.isnumeric


# In[7]:


###### Libraries & edit ######

dict = {
    "a" : "Apple",
    "b" : "Banana",
}

ds = ['a', 'b', 'c', 'd']
x = ['has_{} 1'.format(d) for d in ds]

print(x)

#['has_a 1', 'has_b 1', 'has_c 1', 'has_d 1']


# In[9]:


s = "That I ever did see. Dusty as the handle on the door"

index = s.find("Dusty")
print(index)


# In[13]:


if "Dusty" in s:
    print("Query found")
else:
    print("Query not found")


# In[1]:



############################## Create & load list of functions
#### Save functions as "py" -> load to Jupiter notebook
#### Load the "py" files w/ "%run -i name.py"


x = [4,7,1,4, 35, 7]
## Load functions from py file
get_ipython().run_line_magic('run', '-i Python_find_function.py')
find(35, x)
#!pip install latexify_py
import latexify
#find

@latexify.with_latex
def solve(a,b,c):
    return(-b + math.sqrt(b**2-4*a*c))/(2*a)

solve

### Parallel functions
#https://pythonalgos.com/run-multiple-functions-in-parallel-in-python3/

#### Magic commands w/ %
#%run
#%pwd
#%%timeit


# In[2]:


add(1,2,3,10, flag = True)


# In[1]:


import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
#to set up integration for Jupyter
import matplotlib.pyplot as plt
plt.plot(np.random.randn(60))


# In[ ]:





# In[3]:


import pandas as pd#
#%pwd
#%cd
data = pd.read_csv("Alaska_DEED_2005-2021_v1.csv", low_memory=False)
data.head()


# In[ ]:





# In[15]:


12//5


# In[4]:


y = round(data.isna().sum()/len(data)*100,0)
y = y.apply( lambda x : str(x) + '%')
y


# In[6]:


# Use the subset parameter to search for duplicate values only in the Name column of the DataFrame
import pandas as pd

### Creatting "dups" Python function to view duplicates ###

#function parameters: type = 'All', 'One', 'Multi'
def dups(dataset = data,drop = False, type = 'All', cols = None):
    dataset = pd.DataFrame(data)
    if drop == False:
     if type == 'All':
        return dataset[dataset.duplicated()]
     if type == "Single":
        return dataset[dataset.duplicated([cols])].sort_values(by = cols)
     if type == "Multi":
        return dataset[dataset.duplicated(subset=cols, keep=False)].sort_values(by = cols)

    if drop == True:
     if type == 'All':
        return dataset.drop_duplicates(keep = 'first')
     if type == "Single":
        return dataset.drop_duplicates(subset = cols, keep = 'first')
     if type == "Multi":
         return dataset.drop_duplicates(subset = cols, keep = 'first')

#Examples---

##### Just view duplicates (drop = False) ####

## 'All' -- At least 1 duplicate in 1 of the columns
dups(dataset = data,drop = False, type = "All")

## 'Single' -- duplicate per 1 in column
dups(dataset = data,drop = False, type = "Single", cols = 'SchoolID')

## 'Multi' -- duplicate per multiple columns
cols = ['SchoolID', 'LastName']
dups(dataset = data, drop = False, type = "Multi", cols = cols)

##### Drop duplicates (drop = True) ####
## drop 'All' columns with duplicates
dups(dataset = data,drop = True, type = "All")

## drop 'Single'-- drop rows w/ duplicates in a single column
dups(dataset = data, drop = True, type = "Single", cols = 'SchoolID')

## 'Multi' -- droprows w/ duplicates in multiple columns
cols = ['SchoolID', 'LastName']
dups(dataset = data, drop = True, type = "Multi", cols = cols)


# In[89]:


## Load functions from py file
get_ipython().run_line_magic('run', '-i dups.py')
cols = ['SchoolID', 'LastName']
dups(dataset = data, drop = True, type = "Multi", cols = cols)


# In[94]:


cols = ['SchoolID', 'LastName']
x = dups(dataset = data, drop = False, type = "Multi", cols = cols)


# In[ ]:





# In[131]:


data1 = dups(dataset = data, drop = False, type = "Single", cols = 'SchoolID')

data1 == data1.shift()


# In[20]:


#https://plotly.com/python/linear-fits/
#!pip install plotly.express
import plotly.express as px
fig = px.box(data, 
             y = "YearsExperience", 
             color = 'Gender',
             facet_col="Race",
             title = "AK Teachers: Experience Distributions (2005-2021)")

fig.show()


# In[13]:


data2020 = data[data["Year"] == 2020]
#data2020.loc[(data2020.Gender == "M M") & (data2020.Gender != "m") & (data2020.Gender != "f")]
fig2 = px.scatter(data2020, x="YearsExperience", y='Salary', 
                  facet_col="Gender", color="HighestDegree", 
                  trendline="ols",
                  title = 'AK Teachers: Experice-to-Salary (2020)')


fig2.show()


# In[15]:


data2015 = data[data["Year"] >2015]
#data2020.loc[(data2020.Gender == "M M") & (data2020.Gender != "m") & (data2020.Gender != "f")]
fig3 = px.scatter(data2015, x="Year", y="Salary", color="HighestDegree", 
                  trendline="ols",
                  title = 'AK Teachers: Experice-to-Salary')


fig3.show()


# In[16]:


#data = data.rename(mapper = str.strip, axis = 'columns')
#data['FYE']
list(data)
data.columns


# In[51]:


## Zip
y = [1,22, 17]
category = ['Person1','Person2','Person3']

combo = zip(category, y)
ic(list(combo))


# In[50]:


#pip install icecream
from icecream import ic

ic(add)
ic(add(1,2,5,4))

ic(data.columns)

data2 = data.set_index(['Year'])
#data2.loc['Year']
data2['DistrictID'].unique()


# In[5]:


#https://www.bls.gov/developers/api_python.htm
#https://towardsdatascience.com/acquire-and-visualize-us-inflation-data-with-the-bls-api-python-and-tableau-409a2dca1537
import requests
import json
import prettytable
headers = {'Content-type': 'application/json'}
data = json.dumps({"seriesid": ['CUSR0000SA0','CUSR0000SETB01', 'CUSR0000SAF1','CUSR0000SETA02'],"startyear":"2020", "endyear":"2021"})
p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
json_data = json.loads(p.text)
for series in json_data['Results']['series']:
    x=prettytable.PrettyTable(["CPI type","year","period","value","footnotes"])
    seriesId = series['seriesID']
    for item in series['data']:
        year = item['year']
        period = item['period']
        value = item['value']
        footnotes=""
        for footnote in item['footnotes']:
            if footnote:
                footnotes = footnotes + footnote['text'] + ','
        if 'M01' <= period <= 'M12':
            x.add_row([seriesId,year,period,value,footnotes[0:-1]])
    output = open(seriesId + '.txt','w')
    output.write (x.get_string())
    output.close()
    

print(x)


# In[ ]:





# In[ ]:





# In[2]:


x.update({'seriesid: 'value1', 'key2': 'value2'})
print(x)


# In[7]:


### Darts
#https://unit8co.github.io/darts/
#%pip install darts

###########################
def dups(dataset,drop = False, type = 'All', cols = None):
    dataset = pd.DataFrame(dataset)
    if drop == False:
     if type == 'All':
        return dataset[dataset.duplicated()]
     if type == "Single":
        return dataset[dataset.duplicated([cols])].sort_values(by = cols)
     if type == "Multi":
        return dataset[dataset.duplicated(subset=cols, keep=False)].sort_values(by = cols)

    if drop == True:
     if type == 'All':
        return dataset.drop_duplicates(keep = 'first')
     if type == "Single":
        return dataset.drop_duplicates(subset = cols, keep = 'first')
     if type == "Multi":
         return dataset.drop_duplicates(subset = cols, keep = 'first')
    
###########################


# In[46]:


import pandas as pd
from darts import TimeSeries

# Read a pandas DataFrame
data = pd.read_csv("Alaska_DEED_2005-2021_v1.csv", low_memory=False)
data = pd.DataFrame(data)
cols = ["Year", "Salary"]
data = dups(dataset = data,drop = True, type = "Single", cols = "Year")
data
# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(data, "Year", "Salary")

# Set aside the last 36 months as a validation series
train, val = series.split_before(0.6)


# In[ ]:


#########################################


# In[50]:


#### Clustering
## Import Darts data set
from darts.datasets import ETTh2Dataset

series = ETTh2Dataset().load()[:10000][["MUFL", "LULL"]]
train, val = series.split_before(0.6)


# In[51]:


#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
import numpy as np
from darts.ad import KMeansScorer
from sklearn.cluster import KMeans
#%pip install numpy==1.21.4 
import numpy as np

scorer = KMeansScorer(k=2, window=5)
scorer.fit(train)
anom_score = scorer.score(val)


# In[52]:


from darts.ad import QuantileDetector

detector = QuantileDetector(high_quantile=0.995)
detector.fit(scorer.score(train))
binary_anom = detector.detect(anom_score)


# In[54]:


import matplotlib.pyplot as plt

series.plot()
(anom_score / 2. - 100).plot(label="computed anomaly score", c="orangered", lw=3)
(binary_anom * 45 - 150).plot(label="detected binary anomaly", lw=3)


# In[15]:


import matplotlib.pyplot as plt

series.plot()
prediction.plot(label="forecast", low_quantile=0.05, high_quantile=0.95)
plt.legend()


# In[56]:


from darts.models import ExponentialSmoothing

model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples=1000)


# In[58]:


import plotly.express as px


# In[59]:


import pandas as pd
from darts import TimeSeries

# Read a pandas DataFrame
data = pd.read_csv("Alaska_DEED_2005-2021_v1.csv", low_memory=False)
data = pd.DataFrame(data)
cols = ["Year", "Salary"]
data = dups(dataset = data,drop = True, type = "Single", cols = "Year")
data
# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(data, "Year", "Salary")


# In[61]:


px.line(data, x = "Year", y = "Salary")


# In[68]:


data.groupby(['Year'])['Salary'].median()


# In[124]:


#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
import pandas as pd
from darts import TimeSeries

# Read a pandas DataFrame
data = pd.read_csv("Alaska_DEED_2005-2021_v1.csv", low_memory=False)
data = pd.DataFrame(data)
cols = ["Year", "Salary"]
data = dups(dataset = data,drop = True, type = "Single", cols = "Year")
data
# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(data, "Year", "Salary")

# Set aside the last 36 months as a validation series
train, val = series.split_before(0.6)


# In[164]:


data = pd.read_csv("Alaska_DEED_2005-2021_v1.csv", low_memory=False)
data = pd.DataFrame(data)


# In[166]:


arr = data['Salary']
X = arr.values.reshape(-1,1)


# In[167]:


#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=3, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)

data['Cluster'] = km.fit_predict(X)


# In[132]:


import plotly.express as pd
px.scatter(data, 
           x = "Year", 
           y = "Salary", 
           color = 'Cluster')


# In[133]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

X = np.concatenate([cluster_1, cluster_2, outliers])
y = np.concatenate(
    [np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)


# In[199]:


from sklearn.ensemble import IsolationForest

anomaly_inputs = ['Salary', 'Year']
data_x = pd.DataFrame(data)

model_IF = IsolationForest(contamination=float(0.05),random_state=42)

model_IF.fit(data_x[anomaly_inputs])


# In[172]:


data_x.head()


# In[200]:


data_x['anomaly_scores'] = model_IF.decision_function(data_x[anomaly_inputs])
data_x['anomaly'] = model_IF.predict(data_x[anomaly_inputs])


# In[202]:


data_x.loc[:, [ 'Salary', 'Year','anomaly_scores','anomaly'] ]


# In[194]:



def outlier_plot(data, outlier_method_name, x_var, y_var, 
                 xaxis_limits=[0,100000], yaxis_limits=[2004,2022]):
    
    print(f'Outlier Method: {outlier_method_name}')
    
    # Create a dynamic title based on the method
    method = f'{outlier_method_name}_anomaly'
    
    # Print out key statistics
    print(f"Number of anomalous values {len(data[data['anomaly']==-1])}")
    print(f"Number of non anomalous values  {len(data[data['anomaly']== 1])}")
    print(f'Total Number of Values: {len(data)}')
    
    # Create the chart using seaborn
    g = sns.FacetGrid(data, col='anomaly', height=4, hue='anomaly', hue_order=[1,-1])
    g.map(sns.scatterplot, x_var, y_var)
    g.fig.suptitle(f'Outlier Method: {outlier_method_name}', y=1.10, fontweight='bold')
    g.set(xlim=xaxis_limits, ylim=yaxis_limits)
    axes = g.axes.flatten()
    axes[0].set_title(f"Outliers\n{len(data[data['anomaly']== -1])} points")
    axes[1].set_title(f"Inliers\n {len(data[data['anomaly']==  1])} points")
    return g
#view raw med_outlier_plot.py hosted with ❤ by GitHub


# In[203]:


import seaborn as sns
outlier_plot(data_x, 'Isolation Forest',  'Year', 'Salary', [2004, 2022], [1000, 120000]);


# In[ ]:


#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
from sklearn.ensemble import IsolationForest

