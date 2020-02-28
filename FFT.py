"""
Modified by Kenneth Zhou
FFT Method


"""
import math
import numpy as np
import pandas as pd
import cmath
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
#from scipy.optimize import brentq as root
from scipy.optimize import root
from scipy import interpolate

class BSOption:
    """ encapsulates the data required to do Black-Scholes option pricing formula.
    """
    def __init__(self,s,x,t,sigma,rf,div):
        self.s = s
        self.x = x
        self.t = t
        self.sigma = sigma
        self.rf = rf
        self.div = div
        
    def __repr__(self):
        """ returns a string representation of the BSOption object
        """
        s = 's = $%.2f, x = $%.2f, t = %.2f(years), sigma = %.3f, rf = %.3f, div = %.2f' % (self.s, self.x, self.t, self.sigma,self.rf, self.div)
        return s
    
    def d1(self):
        """ calculates d1 of the option
        """
        numerator = math.log(self.s/self.x) + (self.rf-self.div+self.sigma**2*0.5)*self.t      # Numerator of d1
        denominator = self.sigma * self.t**0.5                                                 # Denominator of d1
        
        return numerator/denominator
        
    def d2(self):
        """ calculates d2 of the option
        """
        return self.d1() - self.sigma*self.t**0.5
    
    def nd1(self):
        """ calculates N(d1) of the option
        """
        return norm.cdf(self.d1())
    
    def nd2(self):
        """ calculates N(d2) of the option
        """
        return norm.cdf(self.d2())
    
class BSEuroCallOption(BSOption):
    def __repr__(self):
        """ returns a string representation of the BSEuroCallOption object
        """
        s = 'BSEuroCallOption, value = $%.2f, \n' % self.value()
        s += 'parameters = (' + BSOption.__repr__(self) + ')'
        
        return s
    
    def value(self):
        """ calculates value for the option
        """ 
        c = self.nd1() * self.s * math.exp(-self.div * self.t)
        c -= self.nd2() * self.x * math.exp(-self.rf * self.t)
        
        return c
    
    def delta(self):
        """ calculates delta for the option
        """
        return self.nd1()

class FTT:
    def __init__(self,sigma,eta0,kappa,rho,theta,S0,r,T):
        self.sigma = sigma
        self.eta0 = eta0
        self.kappa = kappa
        self.rho = rho
        self.theta = theta
        self.S0 = S0
        self.r = r
        self.T = T
        
    def Heston_fft(self,alpha,n,B,K):
        """ Define a function that performs fft on Heston process
        """
        #bt = time.time()
        r = self.r
        T = self.T
        S0 = self.S0
        N = 2**n
        Eta = B / N
        Lambda_Eta = 2 * math.pi / N
        Lambda = Lambda_Eta / Eta
        
        J = np.arange(1,N+1,dtype = complex)
        vj = (J-1) * Eta
        m = np.arange(1,N+1,dtype = complex)
        Beta = np.log(S0) - Lambda * N / 2
        km = Beta + (m-1) * Lambda
        
        ii = complex(0,1)
        
        Psi_vj = np.zeros(len(J),dtype = complex)
        
        for zz in range(0,N):
            u = vj[zz] - (alpha + 1) * ii
            numer = self.Heston_cf(u)
            denom = (alpha + vj[zz] * ii) * (alpha + 1 + vj[zz] * ii)
            
            Psi_vj [zz] = numer / denom
            
        # Compute FTT
        xx = (Eta/2) * Psi_vj * np.exp(-ii * Beta * vj) * (2 - self.dirac(J-1))
        zz = np.fft.fft(xx)
        
        # Option price
        Mul = np.exp(-alpha * np.array(km)) / np.pi
        zz2 = Mul * np.array(zz).real
        k_List = list(Beta + (np.cumsum(np.ones((N, 1))) - 1) * Lambda)
        Kt = np.exp(np.array(k_List))
       
        Kz = []
        Z = []
        for i in range(len(Kt)):
            if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( zz2[i] != float("inf"))&(zz2[i] != float("-inf")) & (zz2[i] is not  float("nan")):
                Kz += [Kt[i]]
                Z += [zz2[i]]
        tck = interpolate.splrep(Kz , np.real(Z))
        price =  np.exp(-r*T)*interpolate.splev(K, tck).real
        #et = time.time()
        
        #runt = et-bt
        runt = 0
        
        return(price,runt)
    
    def fft_array(self,alpha,n,B,K):
        """ Define a function that performs fft on Heston process
        """
        r = self.r
        T = self.T
        S0 = self.S0
        N = 2**n
        Eta = B / N
        Lambda_Eta = 2 * math.pi / N
        Lambda = Lambda_Eta / Eta
        
        J = np.arange(1,N+1,dtype = complex)
        vj = (J-1) * Eta
        m = np.arange(1,N+1,dtype = complex)
        Beta = np.log(S0) - Lambda * N / 2
        km = Beta + (m-1) * Lambda
        
        ii = complex(0,1)
        
        Psi_vj = np.zeros(len(J),dtype = complex)
        
        for zz in range(0,N):
            u = vj[zz] - (alpha + 1) * ii
            numer = self.Heston_cf(u)
            denom = (alpha + vj[zz] * ii) * (alpha + 1 + vj[zz] * ii)
            
            Psi_vj [zz] = numer / denom
            
        # Compute FTT
        xx = (Eta/2) * Psi_vj * np.exp(-ii * Beta * vj) * (2 - self.dirac(J-1))
        zz = np.fft.fft(xx)
        
        # Option price
        Mul = np.exp(-alpha * np.array(km)) / np.pi
        zz2 = Mul * np.array(zz).real
        k_List = list(Beta + (np.cumsum(np.ones((N, 1))) - 1) * Lambda)
        Kt = np.exp(np.array(k_List))
       
        Kz = []
        Z = []
        for i in range(len(Kt)):
            if( Kt[i]>1e-16 )&(Kt[i] < 1e16)& ( Kt[i] != float("inf"))&( Kt[i] != float("-inf")) &( zz2[i] != float("inf"))&(zz2[i] != float("-inf")) & (zz2[i] is not  float("nan")):
                Kz += [Kt[i]]
                Z += [zz2[i]]
        tck = interpolate.splrep(Kz , np.real(Z))
        
        price = []
        for i in range(len(K)):
            price +=  [np.exp(-r*T)*interpolate.splev(K[i], tck).real]
        price = pd.Series(price)
        
        return(price)
    
    
    def dirac(self,n):
        """ Define a dirac delta function
        """
        y = np.zeros(len(n),dtype = complex)
        y[n==0] = 1
        return y
        
    def Heston_cf(self,u):
        """ Define a function that computes the characteristic function for variance gamma
        """
        sigma = self.sigma
        eta0 = self.eta0
        kappa = self.kappa
        rho = self.rho
        theta = self.theta
        S0 = self.S0
        r = self.r
        T = self.T
        
        ii = complex(0,1)
        
        l = cmath.sqrt(sigma**2*(u**2+ii*u)+(kappa-ii*rho*sigma*u)**2)
        w = np.exp(ii*u*np.log(S0)+ii*u*(r-0)*T+kappa*theta*T*(kappa-ii*rho*sigma*u)/sigma**2)/(cmath.cosh(l*T/2)+(kappa-ii*rho*sigma*u)/l*cmath.sinh(l*T/2))**(2*kappa*theta/sigma**2)
        y = w*np.exp(-(u**2+ii*u)*eta0/(l/cmath.tanh(l*T/2)+kappa-ii*rho*sigma*u))
        
        return y
    
    def alpha_plot(self,lst,n,B,K):
        yy = np.array([self.Heston_fft(a,n,B,K)[0] for a in lst])
        
        plt.plot(lst,yy)
        plt.title("Fig. FFT European Call Option Price vs Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("FFT European Call Option Price")
        plt.show()
        
    def NB_plot(self,n_list,B_list,K):
        zz = np.zeros((len(n_list),len(B_list)))
        ee = np.zeros((len(n_list),len(B_list)))
        cc = []
        xx, yy = np.meshgrid(n_list, B_list)
        for i in range(len(n_list)):
            for j in range(len(B_list)):
                temp = self.Heston_fft(1,n_list[i],B_list[j],K)
                zz[i][j] = temp[0]
                ee[i][j] = 1/((temp[0]-21.27)**2*temp[1])
                cc += [(ee[i][j],n_list[i],B_list[j])]
                
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, zz.T, rstride=1, cstride=1, cmap='rainbow')
        plt.title("Fig. FFT European Call Option Price vs N & B")
        ax.set_xlabel("N")
        ax.set_ylabel("B")
        ax.set_zlabel("FFT European Call Option Price")
        plt.show()
        
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(xx, yy, ee.T, rstride=1, cstride=1, cmap='rainbow')
        plt.title("Fig. FFT Efficiency vs N & B")
        ax.set_xlabel("N")
        ax.set_ylabel("B")
        ax.set_zlabel("FFT Efficiency")
        plt.show()
        
        print(max(cc))
        
        
        
def plot_vol_K(price_list,K_list):
    vol = []
    
    for i in range(len(K_list)):
        result = root(lambda x: BSEuroCallOption(150, K_list[i], 0.25, x, 0.025, 0.00).value()-price_list[i],0.3)
        vol += [result.x]
        
    vol = np.array(vol)
    plt.plot(K_list,vol)
    plt.title("Fig. Implied Volatility vs Strike K")
    plt.xlabel("Strike K")
    plt.ylabel("Implied Volatility")
    plt.show()
    
def plot_vol_T(price_list,t_list):
    vol = []
    
    for i in range(len(t_list)):
        result = root(lambda x: BSEuroCallOption(150, 150, t_list[i], x, 0.025, 0.00).value()-price_list[i],0.3)
        vol += [result.x]
        
    vol = np.array(vol)
    plt.plot(t_list,vol)
    plt.title("Fig. Implied Volatility vs Expiry T")
    plt.xlabel("Expiry T")
    plt.ylabel("Implied Volatility")
    plt.show()
        
'''
if __name__ == '__main__':
    # a
    alpha = 1.5
    sigma = 0.2
    eta0 = 0.08
    kappa = 0.7
    rho = -0.4
    theta = 0.1
    S0 = 250
    K = 250
    r = 0.02
    expT = 0.5
    
    n = 11
    B = 250 * 2.7
    a = FTT(sigma,eta0,kappa,rho,theta,S0,r,expT)
    a.Heston_fft(alpha,n,B,K)[0]
    
    # i
    alphas = np.linspace(0.1,35,num = 200)
    a.alpha_plot(alphas,n,B,K)
    
    b=[]
    for i in [0.01, 0.02, 0.25, 0.5, 0.8, 1 ,1.05, 1.5, 1.75,10,30,40]:
        b += [a.Heston_fft(i,n,B,K)[0]]
    print(b)
    
    # ii
    a.Heston_fft(alpha,n,B,K)
    Bs = np.linspace(250*2.5,250*2.7,100)
    ns = np.array([7,8,9,10,11,12,13,14])
    a.NB_plot(ns,Bs,K)
    #a.Heston_fft(alpha,11,260,260)
    #a.Heston_fft(alpha,11,250,260)
'''
    

    
    

    

    
    
    
