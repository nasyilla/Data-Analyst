#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('Data olah covid - Tugas Mengolah dan Menganalisis Data Covid-19 di United Kingdom.csv')
df.head()


# In[4]:


#mencari lokasi baris pada United Kingdom
for i in range(len(df)):
    if df.loc[i, 'location'] == "United Kingdom":
        print(f"'United Kingdom' ditemukan pada baris ke-{i}")


# In[5]:


#Cleansing pada data yang memuat negara United Kingdom
df_uk=df.loc[df['location'] == 'United Kingdom']
df_uk


# In[6]:


df_uk=df.loc[df['location'] == 'United Kingdom']
df_uk


# In[7]:


#mengisi nilai dengan angka 0 
df_uk=df_uk.fillna(0)
df_uk


# In[8]:


df_uk=df_uk.reset_index(drop=True)


# In[8]:


#Cleansing pada semua kolom di df_uk, kecuali date dan total_cases
df_uk = df_uk[['date', 'total_cases']]
df_uk.head()


# In[9]:


df_uk['date'] = pd.to_datetime(df_uk['date'])
df_uk


# In[11]:


# Membuat Kurva Plot dengan Pustaka Matplotlib
plt.plot(df_uk['date'], df_uk['total_cases'], label='Total Kasus')

plt.title('Grafik Total Kasus di United Kingdom')
plt.xlabel('Hari ke-')
plt.ylabel('Jumlah kasus perhari')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[12]:


puncak=8
x=np.linspace(-8,8,100)
y=puncak/(1+np.exp(-x))
plt.plot(x,y, label = 'Signoid')
plt.grid()
plt.show()


# In[13]:


#fungsi Sigmoid untuk memprediksi kasus Covid-19 di United Kingdom
def kurva_sigmoid(t,a,t0,c):
    return c/(1+np.exp(-(t-t0)/(a)))


# In[14]:


x = list(df_uk.index)
y = list(df_uk['total_cases'])


# In[15]:


#Import package curve_fit dan fslove dari scipy.optimize
from scipy.optimize import curve_fit, fsolve


# In[16]:


#Menggunakan metode Levenberg–Marquardt (lm) pada curve_fit(... ,method=’lm’)
fit= curve_fit(kurva_sigmoid, x, y, method='lm')
varA,varB = fit


# In[17]:


print(varA)


# In[18]:


print(varB)


# In[19]:


std_er=np.zeros(len(varA))

for i in range(len(varA)):
    std_er[i]=np.sqrt(varB[i][i])


# In[20]:


a = varA[0] + std_er[0]
t0 = varA[1] + std_er[1]
c = varA[2] + std_er[2]


# In[21]:


#Menentukan pada hari keberapa jumlah kasus puncak di United Kingdom
def puncak(x):
    return kurva_sigmoid(x,a,t0,c)-int(c)


# In[22]:


n_puncak=int(fsolve(puncak,t0))


# In[23]:


n_puncak


# In[24]:


#menghitung jumlah kasus puncak di United Kingdom
n_0=max(x)+1
pred_x=list(range(n_0,n_puncak))


# In[25]:


pred_y=np.zeros(len(x+pred_x))
for i in range(n_puncak):
    pred_y[i]=kurva_sigmoid(i,a,t0,c)


# In[26]:



print("Prediksi jumlah puncak {} orang".format(int(pred_y[-1])))


# In[27]:


#Kurva Plot pada Matplotlib
plt.plot(x+pred_x,pred_y,linewidth=2.0,label='Prediksi', color='red')
plt.scatter(x,y, label='Data Asli', s=10, color='green')
plt.xlabel('Hari ke')
plt.ylabel('Jumlah kumulatif kasus positif')
plt.grid()
plt.legend(loc='best')
plt.show()


# In[28]:


tgl_puncak = df_uk['date'].min() + pd.to_timedelta(t0, unit='D')
ftm = '%Y-%m-%d'
h_puncak=tgl_puncak.strftime(ftm)
print(h_puncak)


# In[29]:


#Menghitung Akurasi dengan R2 Score
def akurasi_r2(y_asli, y_prediksi, x):
    atas=sum((y_asli-y_prediksi[0:len(x)] )**2)
    bawah=sum(((y_asli - np.mean(y)))**2)
    r=1-(atas/bawah)
    return r


# In[30]:


akurasi=akurasi_r2(y,pred_y,x)
print(akurasi*100)


# In[ ]:





# In[ ]:





# In[31]:


# visualisasi menggunakan bar chart untuk membandingkan jumlah kasus asli dan prediksi pada hari ke-365 


# In[32]:


df_uk['total_pred']=pred_y[0:len(x)].astype(int)
df_uk['selisih']=abs(df_uk['total_pred']-df_uk['total_cases'])
df_uk=df_uk[['date','total_cases','total_pred','selisih']]


# In[33]:


df_uk.index = range(1, len(df_uk) + 1)
df_uk.head(20)


# In[34]:


#bar chart untuk membandingkan jumlah kasus asli dan 
#prediksi pada hari ke-365
labels = ['Real', 'Prediksi']
day_365 = 365
if day_365 <= len(df_uk):
    actual_cases_365 = df_uk.loc[day_365, 'total_cases'] 
    predicted_cases_365 = df_uk.loc[day_365, 'total_pred'] 
    values = [actual_cases_365, predicted_cases_365]

    plt.bar(labels, values, color='maroon')
    plt.show()


# In[ ]:





# In[ ]:




