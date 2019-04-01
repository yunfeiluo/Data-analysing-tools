import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import k_means as km
import lin_reg as lg

'''
x_data = np.random.normal(size=50)
y_data = np.random.normal(size=50)
z_data = np.random.normal(size=50)
color = ['k']*len(x_data)
closest = [-1]*len(x_data)

df = pd.DataFrame({
    'x1': x_data,
    'x2': y_data,
    'x3': z_data,
    'color': color,
    'closest': closest
})
'''
df = pd.read_csv("./###.csv", encoding = 'utf8')

#看attribute两两的correlation，一般x1或者x2啥的就是每个数据的attribute，比如name啊age啊之类的
print(df.corr())

ndf = df.filter([str(df.columns[2]), str(df.columns[3]), 
                 str(df.columns[10]), str(df.columns[4]), str(df.columns[8]), str(df.columns[9]), str(df.columns[10])])

#ndf = ndf[df['###'] == "###"]

#lg.plot2var(ndf)
#lg.plot3var(ndf)
df1 = ndf[df['###'] == "###"]
df2 = ndf[df['###'] == "###"]
lg.colorplot(df1, df2)

'''
k = 3
centroids = {}
for i in range(0, k):
    centroids[i] = [np.random.uniform(min(df.iloc[:,0]), max(df.iloc[:,0])), np.random.uniform(min(df.iloc[:,1]), max(df.iloc[:,1]))]

for i in range(0, k):
    centroids[i] = [np.random.randint(min(df.iloc[:,0]), max(df.iloc[:,0]+1)), 
    np.random.randint(min(df.iloc[:,1]), max(df.iloc[:,1]+1)), 
    np.random.randint(min(df.iloc[:,2]), max(df.iloc[:,2]+1))]


km.k_means(df, centroids)
#km.k_means3d(df, centroids)
'''
