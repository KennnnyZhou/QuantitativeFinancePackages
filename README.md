# QuantitativeFinancePackages
Option Pricing

Integration.py
Integration Methods:
    - Left Rieman
    - Midpoint
    - Gauss Legendre

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
    
    FFT.py
    Fast fourier transformation for asset pricing, mainly in Heston model.
