#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import Library yang dibutuhkan
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import pandas as pd
import numpy as np


# In[3]:


automobile = pd.read_csv('Automobile_data.csv')
automobile.head()


# In[4]:


automobile.dtypes


# In[56]:





# In[61]:


automobile['bore'] = automobile['bore'].str.replace(',', '.').astype(float)
automobile['stroke'] = automobile['stroke'].str.replace(',', '.').astype(float)
automobile['price'] = automobile['price'].str.replace(',', '').astype(float)
automobile['horsepower'] = automobile['horsepower'].str.replace(',', '').astype(int)
automobile['peak-rpm'] = automobile['peak-rpm'].str.replace(',', '').astype(float)


# In[62]:


automobile.dtypes


# In[5]:


automobile.isnull().sum()


# In[6]:


# Find out number of records having '?' value for normalized losses
automobile['normalized-losses'].loc[automobile['normalized-losses'] == '?'].count()


# In[7]:


# Setting the missing value to mean of normalized losses and conver the datatype to integer
nl = automobile['normalized-losses'].loc[automobile['normalized-losses'] != '?']
nlmean = nl.astype(str).astype(int).mean()
automobile['normalized-losses'] = automobile['normalized-losses'].replace('?',nlmean).astype(int)
automobile['normalized-losses'].head()


# In[9]:


# Find out the number of values which are not numeric
automobile['price'].str.isnumeric().value_counts()


# In[10]:


# List out the values which are not numeric
automobile['price'].loc[automobile['price'].str.isnumeric() == False]


# In[11]:


#Setting the missing value to mean of price and convert the datatype to integer
price = automobile['price'].loc[automobile['price'] != '?']
pmean = price.astype(str).astype(int).mean()
automobile['price'] = automobile['price'].replace('?',pmean).astype(int)
automobile['price'].head()


# In[12]:


# Checking the numberic and replacing with mean value and conver the datatype to integer
automobile['horsepower'].str.isnumeric().value_counts()
horsepower = automobile['horsepower'].loc[automobile['horsepower'] != '?']
hpmean = horsepower.astype(str).astype(int).mean()
automobile['horsepower'] = automobile['horsepower'].replace('?',pmean).astype(int)


# In[13]:


#Checking the outlier of horsepower
automobile.loc[automobile['horsepower'] > 10000]


# In[14]:


#Excluding the outlier data for horsepower
automobile[np.abs(automobile.horsepower-automobile.horsepower.mean())<=(3*automobile.horsepower.std())]


# In[15]:


#Cleaning Bore


# In[16]:


# Find out the number of invalid value
automobile['bore'].loc[automobile['bore'] == '?']


# In[63]:


# Replace the non-numeric value to null and conver the datatype
automobile['bore'] = pd.to_numeric(automobile['bore'],errors='coerce')
automobile.dtypes


# In[18]:


#Cleaning the stroke


# In[64]:


automobile['stroke'] = pd.to_numeric(automobile['stroke'],errors='coerce')
automobile.dtypes


# In[20]:


#Cleaning the peak rpm data


# In[65]:


# Convert the non-numeric data to null and convert the datatype
automobile['peak-rpm'] = pd.to_numeric(automobile['peak-rpm'],errors='coerce')
automobile.dtypes


# In[22]:


# remove the records which are having the value '?'
automobile['num-of-doors'].loc[automobile['num-of-doors'] == '?']
automobile = automobile[automobile['num-of-doors'] != '?']
automobile['num-of-doors'].loc[automobile['num-of-doors'] == '?']


# In[23]:


#Univariate Analysis


# In[90]:


automobile.make.value_counts().nlargest(9).plot(kind='bar', figsize=(15,5))
plt.title("Number of vehicles by make")
plt.ylabel('Number of vehicles')
plt.xlabel('Make');


# In[27]:


#Insurance risk ratings Histogram


# In[67]:


automobile.symboling.hist(bins=6,color='green');
plt.title("Insurance risk ratings of vehicles")
plt.ylabel('Number of vehicles')
plt.xlabel('Risk rating');


# In[29]:


#Normalized losses histogram


# In[68]:


automobile['normalized-losses'].hist(bins=5,color='orange');
plt.title("Normalized losses of vehicles")
plt.ylabel('Number of vehicles')
plt.xlabel('Normalized losses');


# In[69]:


#Fuel type bar chart


# In[70]:


automobile['fuel-type'].value_counts().plot(kind='bar',color='purple')
plt.title("Fuel type frequence diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Fuel type');


# In[71]:


#Fuel type pie diagram


# In[72]:


automobile['aspiration'].value_counts().plot.pie(figsize=(6, 6), autopct='%.2f')
plt.title("Fuel type pie diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Fuel type');


# In[73]:


automobile.horsepower[np.abs(automobile.horsepower-automobile.horsepower.mean())<=(3*automobile.horsepower.std())].hist(bins=5,color='red');
plt.title("Horse power histogram")
plt.ylabel('Number of vehicles')
plt.xlabel('Horse power');


# In[74]:


automobile['curb-weight'].hist(bins=5,color='brown');
plt.title("Curb weight histogram")
plt.ylabel('Number of vehicles')
plt.xlabel('Curb weight');


# In[75]:


automobile['drive-wheels'].value_counts().plot(kind='bar',color='grey')
plt.title("Drive wheels diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Drive wheels');


# In[76]:


automobile['num-of-doors'].value_counts().plot(kind='bar',color='purple')
plt.title("Number of doors frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('Number of doors');


# In[77]:


import seaborn as sns
corr = automobile.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13,7))
a = sns.heatmap(corr, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)


# In[78]:


plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="make", y="price", data=automobile)


# In[79]:


g = sns.lmplot('price',"engine-size", automobile);


# In[80]:


g = sns.lmplot('normalized-losses',"symboling", automobile);


# In[81]:


plt.scatter(automobile['engine-size'],automobile['peak-rpm'])
plt.xlabel('Engine size')
plt.ylabel('Peak RPM');


# In[82]:


g = sns.lmplot('city-mpg',"curb-weight", automobile, hue="make", fit_reg=False);


# In[83]:


g = sns.lmplot('highway-mpg',"curb-weight", automobile, hue="make",fit_reg=False);


# In[84]:


automobile.groupby('drive-wheels')['city-mpg'].mean().plot(kind='bar', color = 'peru');
plt.title("Drive wheels City MPG")
plt.ylabel('City MPG')
plt.xlabel('Drive wheels');


# In[85]:


automobile.groupby('drive-wheels')['highway-mpg'].mean().plot(kind='bar', color = 'peru');
plt.title("Drive wheels Highway MPG")
plt.ylabel('Highway MPG')
plt.xlabel('Drive wheels');


# In[86]:


plt.rcParams['figure.figsize']=(10,5)
ax = sns.boxplot(x="drive-wheels", y="price", data=automobile)


# In[87]:


pd.pivot_table(automobile,index=['body-style','num-of-doors'], values='normalized-losses').plot(kind='bar',color='purple')
plt.title("Normalized losses based on body style and no. of doors")
plt.ylabel('Normalized losses')
plt.xlabel('Body style and No. of doors');


# In[89]:


automobile.to_excel('automobile2.xlsx', index=False)


# In[91]:


# Menghitung frekuensi setiap jenis mobil
make_counts = automobile['make'].value_counts()

# Menampilkan jenis mobil terbanyak
most_common_make = make_counts.idxmax()
most_common_count = make_counts.max()

print(f"Jenis mobil terbanyak: {most_common_make} dengan jumlah: {most_common_count}")

# Menampilkan semua frekuensi jenis mobil
print("\nFrekuensi setiap jenis mobil:")
print(make_counts)


# In[92]:


# Menghitung frekuensi setiap jenis mobil
fuel_type_counts = automobile['fuel-type'].value_counts()

# Menampilkan jenis mobil terbanyak
most_common_fuel_type = fuel_type_counts.idxmax()
most_common_count = fuel_type_counts.max()

print(f"Jenis mobil terbanyak: {most_common_fuel_type} dengan jumlah: {most_common_count}")

# Menampilkan semua frekuensi jenis mobil
print("\nFrekuensi setiap jenis mobil:")
print(fuel_type_counts)


# In[93]:


# Mengurutkan DataFrame berdasarkan kolom 'horsepower' dalam urutan menurun
sorted_automobile = automobile.sort_values(by='horsepower', ascending=False)

# Mengambil 5 jenis mobil dengan horsepower terbesar
top_5_horsepower = sorted_automobile.head(5)


# In[94]:


top_5_horsepower


# In[96]:


# Memfilter mobil dengan bahan bakar gas dan bertipe sedan
gas_sedan_cars = automobile[(automobile['fuel-type'] == 'gas') & (automobile['body-style'] == 'sedan')]

# Memilih hanya kolom 'make', 'fuel-type', dan 'body-style'
result = gas_sedan_cars[['make', 'fuel-type', 'body-style']]

print("Daftar mobil dengan bahan bakar gas dan bertipe sedan (hanya kolom make, fuel-type, dan body-style):")
print(result)


# In[ ]:




