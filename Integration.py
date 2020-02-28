# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 09:38:04 2020
@author: Kenneth
Integration Methods:
    - Left Rieman
    - Midpoint
    - Gauss Legendre
"""

import math
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt


Nodes = 5

def Left_Rieman(pdf,interval,Nodes):
    left = interval[0]
    right = interval[1]
    step = (right-left)/Nodes
    Integral = 0
    for i in range(Nodes):
        Integral += pdf(left)*step
        left += step
    return(Integral)

def Midpoint(pdf,interval,Nodes):
    left = interval[0]
    right = interval[1]
    step = (right-left)/Nodes
    Integral = 0
    for i in range(Nodes):
        Integral += pdf(left+step/2)*step
        left += step
    return(Integral)

def Gauss_Legendre(pdf,interval,Nodes):
    left = interval[0]
    right = interval[1]
    points, weights = np.polynomial.legendre.leggauss(Nodes)
    def new_pdf(x):
        return(pdf( (right-left)*(x+1)/2 + left ))
    Integral = (right-left)*sum(weights*new_pdf(points))/2
    return Integral
