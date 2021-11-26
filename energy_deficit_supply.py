
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("Electricity Energy Demand & Deficit by State.csv", skiprows = [0], header = 1)

df1.sort_values(['State', "YearValue"], inplace = True)

df1['difference' ]  = df1['EnergyRequirement_MU'] - df1['EnergyAvailability_MU']


states = df1['State'].unique()
dct1 =[]
for state in states:
    data  = df1[df1['State'] == state]
    x = np.array(data['difference'].ewm(com = 0.3).mean())[-1] #  com of 0.3
    #print(state, x)
    dct1.append((x, state))

dct1.sort()

fig = plt.figure(figsize = (15,10) )
for val, state in dct1[2:9]:
    if state in ['DADRA & NAGAR HAVELI','PUDUCHERRY', "SIKKIM", 'GOA']:
        continue
    #print(state)
    data = df1[df1['State'] == state]
    plt.plot(data['YearValue'],data['difference'].ewm(com = 0.3).mean(),'-', label = state)

for val, state in dct1[-4:]:
    #print(state)
    data = df1[df1['State'] == state]
    plt.plot(data['YearValue'],data['difference'].ewm(com = 0.3).mean(),'--', label = state)

plt.legend()
plt.xlabel("Years", fontsize = 12)
plt.ylabel("Diff of Demand and Supply", fontsize = 12)
plt.savefig("energy_deficit_supply.png")
# plt.show()