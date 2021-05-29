#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 12:47:53 2021

@author: will
"""
import numpy as np
#import seaborn
import matplotlib.pyplot as plt


def local_ICE(model,X,index,local=5,n_points=10,alpha=0.5,linewidth=0.5):
    n,d = X.shape
    low,high = np.quantile(X[:,index], [0.25,0.75])
    half_len = (high - low)/local/2
    intervals = np.linspace(-half_len,half_len,n_points)
    
    X = np.repeat(X[...,None],n_points,2)
    X[:,index] = X[:,index] + intervals
    X = X.swapaxes(1,2).reshape(-1,d)
    y = model(X)
    x=X[:,index].reshape(-1,n_points)
    y=y.reshape(-1,n_points)
    
    #seaborn.scatterplot(x=X[:,index],y=y,alpha=alpha,s=s)
    for i in range(n):
        plt.plot(x[i],y[i],color='blue', linestyle='-.',alpha=alpha,linewidth=linewidth)