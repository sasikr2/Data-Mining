# -*- coding: utf-8 -*-
"""gas and oil averages.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15sVc0x97vvhpaW6e1R1AY0ORoXCl6nuc
"""

import pandas as pd
import numpy as np
from collections import defaultdict as dd
from matplotlib import pyplot as plt

data = pd.read_csv("Gas Consumption - CNG Sales Quantity_D_20211115_120440.csv", usecols = [0,1,2,3], error_bad_lines=False, header = None, skiprows=[0,1,2])
data = pd.DataFrame(data)
d = dd(list)
ind = [0, 1, 3]
a, b, c = data[0], data[1], data[3]
for i in range(len(a)):
    d[b[i]].append(c[i])
out_ = dd(list)
for i in d:
    val = sum(d[i])/len(d[i])
    out_['state'].append(i)
    out_['average gas consumption'].append(val)
#print(out_)
out_ = pd.DataFrame(out_)
out_.to_csv('avg_gas_consumption_state.csv')

plt.figure(figsize=(10, 6))
plt.barh(out_['state'], out_['average gas consumption'], align = 'center', height = 0.8)
plt.title('Average gas consumption every state')
plt.ylabel('State')
plt.xlabel('Quantity(tonne)')
plt.savefig('Average gas consumption every state')
#plt.show()

l = []
for i in range(len(out_['state'])):
    l.append([out_['state'][i], out_['average gas consumption'][i]])
l = sorted(l, key = lambda x: x[1])[::-1]
l = l[:5]
x, y = [], []
for i in l:
    x.append(i[0])
    y.append(i[1])

plt.figure(figsize=(8, 6))
plt.barh(x, y, align = 'center', height = 0.8)
plt.title('Average gas consumption for top 5 states')
plt.ylabel('State')
plt.xlabel('Quantity(tonne)')
plt.savefig('Average gas consumption for top 5 states')
#plt.show()

d = dd(list)
ind = [0, 1, 2, 3]
a, b, c = data[0], data[2], data[3]
for i in range(len(a)):
    d[b[i]].append(c[i])
out_ = dd(list)
for i in d:
    val = sum(d[i])/len(d[i])
    out_['company'].append(i)
    out_['average gas consumption'].append(val)
#print(out_)
out_ = pd.DataFrame(out_)
out_.to_csv('avg_gas_consumption_company.csv')

plt.figure(figsize=(10, 6))
plt.barh(out_['company'], out_['average gas consumption'], align = 'center', height = 0.8)
plt.title('Average gas consumption every company')
plt.ylabel('Company')
plt.xlabel('Quantity(tonne)')
plt.savefig('Average gas consumption every company')
#plt.show()

l = []
for i in range(len(out_['company'])):
    l.append([out_['company'][i], out_['average gas consumption'][i]])
l = sorted(l, key = lambda x: x[1])[::-1]
l = l[:5]
x, y = [], []
for i in l:
    x.append(i[0])
    y.append(i[1])

plt.figure(figsize=(8, 6))
plt.barh(x, y, align = 'center', height = 0.8)
plt.title('Average gas consumption for top 5 companies')
plt.ylabel('Company')
plt.xlabel('Quantity(tonne)')
plt.savefig('Average gas consumption for top 5 companies')
#plt.show()

d = dd(list)
ind = [0, 1, 3]
a, b, c = data[0], data[1], data[3]
for i in range(len(a)):
    d[a[i]].append(c[i])
x = [2014+i for i in range(7)]
y = [0]*7
for i in d:
    val = sum(d[i])/len(d[i])
    y[i-2014]=val
plt.plot(x,y)
plt.title('Average gas consumption every year')
plt.xlabel('Year')
plt.ylabel('Quantity(tonne)')
plt.savefig('Average gas consumption every year')
#plt.show()

data = pd.read_csv("Gas Supply - Import Countrywise_D_20211115_121312.csv", usecols = [0,1,3,4], error_bad_lines=False, header = None, skiprows=[0,1,2])
data = pd.DataFrame(data)
d = dd(list)
di = dd(list)
a, b, c, e = data[0], data[1], data[3], data[4]
for i in range(len(a)):
    d[b[i]].append(c[i])
    di[b[i]].append(e[i])
out_ = dd(list)
for i in d:
    val = sum(d[i])/len(d[i])
    x = 0
    for j in range(len(d[i])):
        x += di[i][j]*d[i][j]
    x /= sum(d[i])
    out_['state'].append(i)
    out_['average gas quantity import'].append(val)
    out_['average price'].append(x)
#print(out_)
out_ = pd.DataFrame(out_)
out_.to_csv('avg_gas_import.csv')

plt.figure(figsize=(16, 10))
plt.barh(out_['state'], out_['average gas quantity import'], align = 'center', height = 0.8)
plt.title('Average gas import quantity every country')
plt.ylabel('Country')
plt.xlabel('Quantity(tonne)')
plt.savefig('Average gas import quantity every country')
#plt.show()

plt.figure(figsize=(16, 10))
plt.barh(out_['state'], out_['average price'], align = 'center', height = 0.8)
plt.title('Average gas import price every country')
plt.ylabel('Country')
plt.xlabel('Price')
plt.savefig('Average gas import price every country')
#plt.show()

d = dd(list)
ind = [0, 1, 3]
a, b, c = data[0], data[1], data[3]
for i in range(len(a)):
    d[a[i]].append(c[i])
x = [2005+i for i in range(16)]
y = [0]*16
for i in d:
    val = sum(d[i])/len(d[i])
    y[i-2005]=val
plt.plot(x,y)
plt.title('Average gas imports every year')
plt.xlabel('Year')
plt.ylabel('Quantity(tonne)')
plt.savefig('Average gas import every year')
#plt.show()

data = pd.read_csv("Oil Consumption by State_D_20211115_122143.csv", usecols = [0,1,2,3], error_bad_lines=False, header = None, skiprows=[0,1,2])
data = pd.DataFrame(data)
d = dd(list)
di = dd(list)
ind = [0, 1, 3]
a, b, c, e = data[0], data[1], data[2], data[3]
for i in range(len(a)):
    d[b[i]].append(e[i])
    di[c[i]].append(e[i])
out_ = dd(list)
for i in d:
    val = sum(d[i])/len(d[i])
    out_['region'].append(i)
    out_['average oil consumption'].append(val)
#print(out_)
out_ = pd.DataFrame(out_)
out_.to_csv('avg_oil_consumption_region.csv')

plt.figure(figsize=(10, 6))
plt.barh(out_['region'], out_['average oil consumption'], align = 'center', height = 0.8)
plt.title('Average oil consumption every region')
plt.ylabel('Region')
plt.xlabel('Quantity(tonne)')
plt.savefig('Average oil consumption every region')
#plt.show()

out_ = dd(list)
for i in di:
    val = sum(di[i])/len(di[i])
    out_['region'].append(i)
    out_['average oil consumption'].append(val)
print(out_)
out_ = pd.DataFrame(out_)
out_.to_csv('avg_oil_consumption_state.csv')

plt.figure(figsize=(16, 10))
plt.barh(out_['region'], out_['average oil consumption'], align = 'center', height = 0.8)
plt.title('Average oil consumption every state')
plt.ylabel('State')
plt.xlabel('Quantity(tonne)')
plt.savefig('Average oil consumption every state')
#plt.show()

l = []
for i in range(len(out_['region'])):
    l.append([out_['region'][i], out_['average oil consumption'][i]])
l = sorted(l, key = lambda x: x[1])[::-1]
l = l[:5]
x, y = [], []
for i in l:
    x.append(i[0])
    y.append(i[1])

plt.figure(figsize=(8, 6))
plt.barh(x, y, align = 'center', height = 0.8)
plt.title('Average oil consumption for top 5 states')
plt.ylabel('State')
plt.xlabel('Quantity(tonne)')
plt.savefig('Average oil consumption for top 5 states')
#plt.show()


d = dd(list)
ind = [0, 1, 3]
a, b, c = data[0], data[1], data[3]
for i in range(len(a)):
    d[a[i]].append(c[i])
x = [2011+i for i in range(10)]
y = [0]*10
for i in d:
    val = sum(d[i])/len(d[i])
    y[i-2011]=val
plt.plot(x,y)
plt.title('Average oil consumption every year')
plt.xlabel('Year')
plt.ylabel('Quantity(tonne)')
plt.savefig('Average oil consumption every year')
#plt.show()