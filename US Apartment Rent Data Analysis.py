#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install xlrd


# In[3]:


import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import seaborn as sns
import matplotlib.pyplot as plt


# In[12]:


# This is an exploratory data analysis of US apartment rent from Jamuary 2017 to September 2020.
# It includes rent from different sized apartments in hundreds of US cities and most US states. 
# I will look to answer different questions such as:
#
# - How has rent changed over time?
# - How does rent vary by state? 
# - How does the price of living alone compare to living with roommates?
#
# Data taken from https://www.apartmentlist.com/research/category/data-rent-estimates
#
# Mark Aguila


# In[105]:


rent = pd.read_csv('Apartment_List_Rent_Data_-_City_2020-9.csv')


# In[80]:


# The data set has thousands of rows and over one hundred thousand data points.
# It is far to large to display here in its entirety.

print('Number of unique locations:', len(rent['Location'].unique()))
print('Number of rows:', len(rent))
print('Number of data  points:', rent.size)


# In[207]:


# This is a small preview into the data set.
# This selects a random sample of 10 rows from the data.
# Re-run this cell using "SHIFT + ENTER" to select a new random sample. 

rent.sample(10)


# In[87]:


# Data Cleaning
# Notice entries listed as NaN.
# These NOT A NUMBER (NaN) entries mean
# the data set is missing entries.

rent.loc[:,['Location','Price_2017_01']]


# In[106]:


# Here is a breakdown of how many rent values are 
# missing for each month from 01/2017 to 09/2020.

rent.isna().sum()[4:]


# In[107]:


# I will remove NaN entries to work with complete rent information.

rent = rent.dropna(how='all', thresh = 20)#.sample(10)
len(rent['Location'].unique())
#rent[rent['Location'] == 'Brookhaven, GA']


# In[188]:


# I'll start by plotting average rent over time to see 
# what trends have emerged over the last three years.
avg_rent = rent.mean()
avg_rent.head()


# In[186]:


# Labels for months
years = [str(year) for year in range(2017, 2021)]
months = [str(month) for month in range(1, 13)]
dates = [month+'/'+year for year in years for month in months]

# Average rent graph
dims = (20, 12)
fig, ax = plt.subplots(figsize=dims)
chart = sns.lineplot(ax=ax, data=rent.mean())
chart.set_xticklabels(labels = dates, rotation=45, )
chart.set_title('Apartment Rent Fluctuation (01/2017-09/2020)')
ax.set_xlabel("Month")
ax.set_ylabel("Dollars")


# In[ ]:


# Rents have seen short terms increases and decreases with a long term rise in prices.
# However, this includes the full rent of all apartment sizes. Many people do not pay
# for a several bedroom apartment alone. 


# In[189]:


# The following adjusts for a more accurate look of what a renter might pay
# per month. It divides the average rent of an apartment by how many rooms it has.

studio = rent[rent['Bedroom_Size'] == 'Studio'].mean() 
single = rent[rent['Bedroom_Size'] == '1br'].mean()
double = rent[rent['Bedroom_Size'] == '2br'].mean() / 2
triple = rent[rent['Bedroom_Size'] == '3br'].mean() / 3
quad = rent[rent['Bedroom_Size'] == '4br'].mean() / 4

per_person = pd.concat([studio, single, double, triple, quad], axis=1)
per_person.rename(axis = 1, mapper={0:'Studio', 1:'Single', 2:'Double', 3:'Triple', 4:'Quad'}, inplace = True)
per_person.head()


# In[192]:


# The following charts changes in rent payments that a renter
# sharing an apartment might pay. It follows the same trend as
# total rent but notice it ranges from around 800 to 900 dollars
# rather than 1450 to 1600 dollars.

dims = (20, 12)
fig, ax = plt.subplots(figsize=dims)
chart = sns.lineplot(ax=ax, data=per_person.transpose().mean())
chart.set_xticklabels(labels = dates, rotation=45, )
chart.set_title('Adjusted Rent Fluctuation (01/2017-09/2020)')
ax.set_xlabel("Month")
ax.set_ylabel("Dollars")


# In[190]:


# This graph separates the trends by apartment size. It shows
# adjusted rent to show comparable payments per room since 
# total rent payments for larger apartments would obviously be higher.

dims = (20, 12)
fig, ax = plt.subplots(figsize=dims)
chart = sns.lineplot(ax=ax, data=per_person)#.set_title('Rent Fluctuation by Apartment Size 01/2017-09/2020')
chart.set_xticklabels(labels = dates, rotation=45, )
chart.set_title('Adjusted Rent Fluctuation by Apartment Size (01/2017-09/2020)')
ax.set_xlabel("Month")
ax.set_ylabel("Dollars")


# In[193]:


# Renters with more roommates pay less rents by splitting rent
# among more people. However, it seems an added benefit is that
# larger apartments have seen more stable rent over time.


# In[ ]:





# In[ ]:





# In[203]:


t = rent.groupby('Location').max().loc[:,'Price_2017_01':] - rent.groupby('Location').min().loc[:,'Price_2017_01':]  
#- rent.groupby('State').min()


# In[ ]:





# In[ ]:





# In[ ]:





# In[217]:


state = pd.read_csv('Apartment_List_Rent_Data_-_State_2020-9.csv')
us = pd.read_csv('Apartment_List_Rent_Data_-_National_2020-9.csv')
#g  = pd.read_excel('h01ar.xlsx')


# In[218]:


g.head()


# In[ ]:





# In[306]:


# A look at rent fluctuation in a sample of 10 different states. Most states have maintained relatively stable rates.
# From 2017 to 2020, rent rose or fell a few hundred dollars within each state.
states = ['CA', 'MA', 'TX', 'OH', 'MI', 'CO', 'MO', 'FL', 'IA', 'KS']
states_selected = rent[rent['State'].isin(states)]
state_rents = states_selected.loc[:,:'Price_2020_09'].groupby('State').mean().transpose().dropna()


# In[307]:


dims = (2*11.7, 2*8.27)
fig, ax = plt.subplots(figsize=dims)
chart = sns.lineplot(ax=ax, data=state_rents)
chart.set_xticklabels(avg.index, rotation=45)
chart = sns.lineplot(ax=ax, data=state_rents)


# In[205]:


state_rent = pd.DataFrame(rent.groupby("State").mean()['Price_2017_01'].sort_values(ascending = False))
fig = state_rent.plot.bar(figsize=(25, 10))
#fig.suptitle('test title', fontsize=20)
plt.xlabel('xlabel', fontsize=18)
plt.ylabel('Rent per Month', fontsize=16)


# In[ ]:


start_date = 'Price_2017_01'
end_date = 'Price_2020_01'
latest_date = 'Price_2020_09'


# In[ ]:





# In[153]:


# Average rent for 1br apartments (2020)

avg2020 = avg.loc['Price_2020_01':,:]
dims = (2*11.7, 2*8.27)
fig, ax = plt.subplots(figsize=dims)
chart = sns.scatterplot(ax=ax, data=avg2020)
chart.set_xticklabels(avg2020.index, rotation=45)
chart = sns.lineplot(ax=ax, data=avg2020)


# In[163]:


# Add percent changes in rent from date X to Y.

june2019 = rent['Price_2019_06'] 
end = june2019

def delta_rent(x, y):
    return 100 * (y - x) / x 

m = rent['Price_2019_05']
rent['Month'] = delta_rent(m, end)

q = rent['Price_2019_03']
rent['Quarter'] = delta_rent(q, end)

h = rent['Price_2018_12']
rent['Half'] = delta_rent(h, end)

y1 = rent['Price_2018_06']
rent['Year1'] = delta_rent(y1, end)

y2 = rent['Price_2017_06']
rent['Year2'] = delta_rent(y2, end)


# In[182]:


# DELTA includes June 2019 rents and percent changes over time.
delta = rent[['City', 'State', 'Bedroom_Size', 'Price_2019_06', 'Month', 'Quarter', 'Half', 'Year1', 'Year2']]


# In[210]:


# Plot monthly rent (06/2020) against percent change from 1 month earlier
dims = (1.5*11.7, 1.5*8.27)
fig, ax = plt.subplots(figsize=dims)
chart.set_xticklabels(delta.index, rotation=45)
chart = sns.scatterplot(ax=ax, data = delta, x='Price_2019_06', y='Month')
sns.scatterplot(ax=ax, x = np.arange(0, 5000), y = [rent['Month'].mean()]*5000, palette = 'red')


# In[209]:


# Plot monthly rent (06/2020) against percent change from 1 quarter earlier
dims = (1.5*11.7, 1.5*8.27)
fig, ax = plt.subplots(figsize=dims)
chart.set_xticklabels(delta.index, rotation=45)
chart = sns.scatterplot(ax=ax, data = delta, x='Price_2019_06', y='Quarter')
sns.scatterplot(ax=ax, x = np.arange(0, 5000), y = [rent['Quarter'].mean()]*5000, palette = 'red')


# In[208]:


# Plot monthly rent (06/2020) against percent change from half a year earlier
dims = (1.5*11.7, 1.5*8.27)
fig, ax = plt.subplots(figsize=dims)
chart.set_xticklabels(delta.index, rotation=45)
chart = sns.scatterplot(ax=ax, data = delta, x='Price_2019_06', y='Half')
sns.scatterplot(ax=ax, x = np.arange(0, 5000), y = [rent['Half'].mean()]*5000, palette = 'red')


# In[207]:


# Plot monthly rent (06/2020) against percent change from 1 year earlier
dims = (1.5*11.7, 1.5*8.27)
fig, ax = plt.subplots(figsize=dims)
chart.set_xticklabels(delta.index, rotation=45)
chart = sns.scatterplot(ax=ax, data = delta, x='Price_2019_06', y='Year1')
sns.scatterplot(ax=ax, x = np.arange(0, 5000), y = [rent['Year1'].mean()]*5000, palette = 'red')


# In[206]:


# Plot monthly rent (06/2020) against percent change from 2 years earlier
dims = (1.5*11.7, 1.5*8.27)
fig, ax = plt.subplots(figsize=dims)
chart.set_xticklabels(delta.index, rotation=45)
chart = sns.scatterplot(ax=ax, data = delta, x='Price_2019_06', y='Year2')
sns.scatterplot(ax=ax, x = np.arange(0, 5000), y = [rent['Year2'].mean()]*5000, palette = 'red')


# In[258]:


# Compare solo vs roommate living

single = rent.groupby('Bedroom_Size').mean().loc['1br',:'Price_2020_09']
double = rent.groupby('Bedroom_Size').mean().loc['2br',:'Price_2020_09'] / 2
triple = rent.groupby('Bedroom_Size').mean().loc['3br',:'Price_2020_09'] / 3
quad = rent.groupby('Bedroom_Size').mean().loc['4br',:'Price_2020_09'] / 4
studio = rent.groupby('Bedroom_Size').mean().loc['Studio',:'Price_2020_09']

payments = pd.concat([single, double, triple, quad, studio], axis=1).reset_index()
dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=dims)
fig = payments.plot(ax=ax)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




