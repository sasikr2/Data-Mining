#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Electricity Consumption by Consuming Sector_D_20211110_094023.csv", skiprows = [0], header= 1)

sector  = df['ConsumingSector'].unique()
year = df['YearValue'].unique()

df_l4 = df[((df['YearValue'] >= 2012) & (df['YearValue'] <= 2019))]

data_sectorwise = []
for se in sector:
    data_sectorwise.append(df_l4[(df_l4['ConsumingSector'] == se) ])

fig = plt.figure(figsize = (15,10))
ax = plt.subplot(111)
for i in range(8):
    ax.plot(data_sectorwise[i]['YearValue'].values ,data_sectorwise[i]['Consumption_GWh'].values, '-',  label = sector[i])
    
plt.xlabel("Years", fontsize = 13)
plt.ylabel("Generation(GHw)", fontsize = 13)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("sectorwise_consumption.png")
# plt.show()

