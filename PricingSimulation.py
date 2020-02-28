# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 12:01:55 2019
@author: Kenneth
PricingSimulation.py
Methods involved in option pricing simulations, including:
    - class Random_Number(): to generate random numbers
    
    - class Sto_Process(s_0 , k, r, sigma , tau): to generate stochastic process paths in subclasses
        - subclass: Brownian_Motion
        - subclass: Bachelier_Model
        - subclass: CEV_Process
        - functions: BS_Call, BS_Put, BS_delta
    
    - class Payoff(): to calculate various options' payoff on specified path
        - functions: Euro_Call, Euro_Put
        - functions: Lookback_Call, Lookback_Put
        
    - function calculate_implied_volatility(): takes a Sto_Process object to calculate implied volatility on Black-Scholes Model 
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

#Nt = 100

class Random_Number(): 
    """
    This class is for generating list of random numbers of different types.
    """
    def __init__(self,n):
        self.n = n
    
    def Normal(self, miu=0, sigma=1):
        """
        This function returns a list of normal random numbers.
        :miu: the mean of the Gaussian distribution
        :sigma: the standard deviation of the Gaussian distribution
        """
        l = np.random.randn(self.n)
        l = miu + sigma*l
        return l
        '''
        def Poisson(self,lamda=1):
            return 0
        '''
        
class Payoff():
    """
    The payoff object takes an argument of Sto_Process object to construct.
    It calculates the payoff of different options.
    """
    def __init__(self,sto):
        self.s_0 = sto.s_0
        self.s_t = sto.S_T()
        self.k = sto.k
        self.r = sto.r
        self.sigma = sto.sigma
        self.tau = sto.tau
        self.s = sto.s    
    """
    Euro_Call and Euro_Put calculate the payoff of European options.
    """
    def Euro_Call(self):
        if self.s_t > self.k:
            return ( self.s_t - self.k )*(math.exp(-self.r * self.tau))
        else:
            return 0
    def Euro_Put(self):
        if self.s_t < self.k:
            return ( - self.s_t + self.k )*(math.exp(-self.r * self.tau))
        else:
            return 0
    """
    Lookback_Call and Lookback_Put calculate the payoff of European options.
    """
    def Lookback_Call(self):
        max_price = max(self.s['price'])
        if max_price > self.k:
            return ( max_price - self.k )*(math.exp(-self.r * self.tau))
        else:
            return 0       
    def Lookback_Put(self):
        min_price = min(self.s['price'])
        if min_price < self.k:
            return ( - min_price + self.k )*(math.exp(-self.r * self.tau))
        else:
            return 0
        
class Sto_Process():
    """
    Encapsulates model parameters and generate a path with random numbers.
    """
    def __init__(self, s_0 , k, r, sigma , tau):
        self.s_0 = s_0
        self.k = k
        self.r = r
        self.sigma = sigma
        self.tau = tau
        # A blank DataFrame is created here which is for price sample paths
        self.s = pd.DataFrame(data=[self.s_0],index=[0],columns=['price'])
        self.Nt = 100
        #print('Create Sto')
    # Describe() and plot() allows to analyze the sanple path 
    def describe(self):
        print(self.s.describe())
        return 0
    def plot(self):
        return self.s.plot(figsize=(12,5))
    # To get access to the terminal price of the process
    def S_T(self):
        return self.s['price'][self.tau]
    # Set the time sections of each time unit
    def get_Nt(self):
        return self.Nt
    def set_Nt(self,k):
        self.Nt = k
    # Set the sigma of the option
    def get_sigma(self):
        return self.sigma
    def set_sigma(self,sigma):
        self.sigma = sigma
    """
    BS_Call and BS_Put calculate the price of options on parameters by formula.
    """
    def BS_Call(self):
        sigmaRtT = ( self.sigma * math . sqrt ( self.tau ))
        rSigTerm = (self.r + self.sigma * self.sigma /2.0) * self.tau
        d1 = ( math . log ( self.s_0 /self.k) + rSigTerm ) / sigmaRtT
        d2 = d1 - sigmaRtT
        term1 = self.s_0 * norm.cdf(d1)
        term2 = self.k * math . exp (-self.r * self.tau ) * norm.cdf (d2)
        return term1 - term2
    
    def BS_Put(self):
        sigmaRtT = ( self.sigma * math . sqrt ( self.tau ))
        rSigTerm = (self.r + self.sigma * self.sigma /2.0) * self.tau
        d1 = ( math . log ( self.s_0 /self.k) + rSigTerm ) / sigmaRtT
        d2 = d1 - sigmaRtT
        term1 = self.k * math . exp (-self.r * self.tau ) * norm.cdf(-d2)
        term2 = self.s_0 * norm.cdf (-d1)
        return term1 - term2
    
    def BS_delta(self):
        sigmaRtT = ( self.sigma * math . sqrt ( self.tau ))
        rSigTerm = (self.r + self.sigma * self.sigma /2.0) * self.tau
        d1 = ( math . log ( self.s_0 /self.k) + rSigTerm ) / sigmaRtT
        return (norm.cdf(d1))
        
class Brownian_Motion(Sto_Process):
    """
    Subclass of Sto_Process: Brownian Motion type
    """ 
    def __init__(self, s_0 , k, r, sigma , tau):
        Sto_Process.__init__(self, s_0 , k, r, sigma , tau)
        dt = self.tau/self.Nt
        # Create a Random_Number object to acquire random numbers in a list
        a = Random_Number(int(self.tau/dt)).Normal(0,math.sqrt(dt))
        step = 0
        s_t = self.s_0
        while step < self.tau/dt:
            s_t += (self.r*s_t*dt + self.sigma*s_t*a[step])
            step += 1
            df_s_t = pd.DataFrame(data=[s_t],index=[step*dt],columns=['price'])
            self.s = pd.concat([self.s,df_s_t],axis=0)
            
class Bachelier_Model(Sto_Process):  
    """
    Subclass of Sto_Process: Bachelier model type
    """
    def __init__(self, s_0 , k, r, sigma , tau):
        #print('P1')
        Sto_Process.__init__(self, s_0 , k, r, sigma , tau)
        dt = self.tau/self.Nt
        # Create a Random_Number object to acquire random numbers in a list
        a = Random_Number(int(self.tau/dt)).Normal(0,math.sqrt(dt))
        step = 0
        s_t = self.s_0
        while step < self.tau/dt:
            s_t += (self.r*dt + self.sigma*a[step])
            step += 1
            df_s_t = pd.DataFrame(data=[s_t],index=[step*dt],columns=['price'])
            self.s = pd.concat([self.s,df_s_t],axis=0)
        #print('P2')
        
class CEV_Process(Sto_Process):
    """
    Subclass of Sto_Process: CEV process type
    """ 
    def __init__(self, s_0 , k, r, sigma , tau, beta):
        Sto_Process.__init__(self, s_0 , k, r, sigma , tau)
        self.beta = beta
        dt = self.tau/self.Nt
        # Create a Random_Number object to acquire random numbers in a list
        a = Random_Number(int(self.tau/dt)).Normal(0,math.sqrt(dt))
        step = 0
        s_t = self.s_0
        while step < self.tau/dt:
            s_t += (self.r*s_t*dt + self.sigma*pow(s_t,self.beta)*a[step])
            step += 1
            df_s_t = pd.DataFrame(data=[s_t],index=[step*dt],columns=['price'])
            self.s = pd.concat([self.s,df_s_t],axis=0)
            

#implied volatility by Black-Scholes
def calculate_implied_volatility(option, option_type ,value):
    ''' calculate the implied volatility of an observed option. '''
    
    accuracy = 0.001
    
    upper = 0.50 # initial value of the upper bound
    lower = 0.00 # initial value of the lower bound
     
    option.sigma = upper

    if option_type == "Call": 
        while abs(option.BS_Call() - value) > accuracy:
    
            if option.BS_Call() > value:
    
                upper = option.sigma
                option.sigma = (upper + lower) / 2
            else:
    
                lower = option.sigma
                option.sigma = (upper + lower) / 2
        
        return option.sigma
    elif option_type == "Put":
        while abs(option.BS_Put() - value) > accuracy:
    
            if option.BS_Put() > value:
    
                upper = option.sigma
                option.sigma = (upper + lower) / 2
            else:
    
                lower = option.sigma
                option.sigma = (upper + lower) / 2
        
        return option.sigma
    else:
        return "Type Undefined"


        