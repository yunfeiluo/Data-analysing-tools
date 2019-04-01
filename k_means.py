import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import copy

def plot(centroids, colors, df, end):
    plt.scatter(df.iloc[:,0], df.iloc[:,1], color=df['color'], alpha=0.3, edgecolor='k')
    for i in centroids:
        plt.scatter(*[centroids[i][0], centroids[i][1]], color = colors[i])
        #plt.scatter(*centroids[i], color = colors[i])
    plt.xlim(min(df.iloc[:,0]) - 1, max(df.iloc[:,0]) + 1)
    plt.ylim(min(df.iloc[:,1]) - 1, max(df.iloc[:,1]) + 1)
    if (end):
        plt.show()
    else:
        plt.show(block=False)
        plt.pause(0.2)
        plt.close()

def assign(centroids, colors, df):
    color = ['k']*len(df.iloc[:,0])
    closest = [-1]*len(df.iloc[:,0])
    for i in range(0, len(df.iloc[:,0])):
        closest[i] = 0
        cld = math.sqrt(((df.iloc[:,0][i]-centroids[0][0])**2) + ((df.iloc[:,1][i]-centroids[0][1])**2))
        for j in range(0, len(centroids)):
            dist = math.sqrt(((df.iloc[:,0][i]-centroids[j][0])**2) + ((df.iloc[:,1][i]-centroids[j][1])**2))
            if (dist < cld):
                closest[i] = j
                cld = dist
        color[i] = colors[closest[i]]
    df['color'] = color
    df['closest'] = closest

def update(centroids, df):
    for i in centroids:
        centroids[i][0] = np.mean(df[df['closest'] == i].iloc[:,0])
        centroids[i][1] = np.mean(df[df['closest'] == i].iloc[:,1])

def isSame(lastC, centroids):
    for i in lastC:
        if (lastC[i] != centroids[i]):
            return False
    return True

def contains(colors, color):
    for i in colors:
        if (colors[i] == color):
            return True
    return False

def randColor(colors):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    rand = np.random.randint(0, 8)
    while (contains(colors, color[rand])):
        rand = np.random.randint(0, 8)
    return color[rand]

def k_means(df, centroids):
    colors = {}
    for i in range(len(centroids)):
        colors[i] = randColor(colors)

    lastC = copy.deepcopy(centroids)
    while(True):
        assign(centroids, colors, df)
        update(centroids, df)
        assign(centroids, colors, df)
        if (isSame(lastC, centroids)):
            print("Done.")
            plot(centroids, colors, df, True)
            break
        plot(centroids, colors, df, False)
        lastC = copy.deepcopy(centroids)

###############
#K-means for 3 random variables

def assign3d(centroids, colors, df):
    color = ['k']*len(df.iloc[:,0])
    closest = [-1]*len(df.iloc[:,0])
    for i in range(0, len(df.iloc[:,0])):
        closest[i] = 0
        cld = math.sqrt(((df.iloc[:,0][i]-centroids[0][0])**2) + ((df.iloc[:,1][i]-centroids[0][1])**2) + ((df.iloc[:,2][i]-centroids[0][1])**2))
        for j in range(0, len(centroids)):
            dist = math.sqrt(((df.iloc[:,0][i]-centroids[j][0])**2) + ((df.iloc[:,1][i]-centroids[j][1])**2) + ((df.iloc[:,2][i]-centroids[0][1])**2))
            if (dist < cld):
                closest[i] = j
                cld = dist
        color[i] = colors[closest[i]]
    df['color'] = color
    df['closest'] = closest

def update3d(centroids, df):
    for i in centroids:
        centroids[i][0] = np.mean(df[df['closest'] == i].iloc[:,0])
        centroids[i][1] = np.mean(df[df['closest'] == i].iloc[:,1])
        centroids[i][2] = np.mean(df[df['closest'] == i].iloc[:,2])

def plot3d(centroids, colors, df, end):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df.iloc[:,0], df.iloc[:,1], df.iloc[:,2], c=df['color'], alpha=0.3, marker='o')
    for i in centroids:
        ax.scatter(*[centroids[i][0], centroids[i][1], centroids[i][2]], c=colors[i])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    #plt.xlim(min(df['x1']) - 1, max(df['x1']) + 1)
    #plt.ylim(min(df['x2']) - 1, max(df['x2']) + 1)
    if (end):
        plt.show()
    else:
        #plt.show(block=False)
        #plt.pause(0.5)
        plt.close()

def k_means3d(df, centroids):
    colors = {}
    for i in range(len(centroids)):
        colors[i] = randColor(colors)

    lastC = copy.deepcopy(centroids)
    while(True):
        assign3d(centroids, colors, df)
        update3d(centroids, df)
        assign3d(centroids, colors, df)
        if (isSame(lastC, centroids)):
            print("Done.")
            plot3d(centroids, colors, df, True)
            break
        plot3d(centroids, colors, df, False)
        lastC = copy.deepcopy(centroids)