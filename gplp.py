from scipy import special
from scipy.special import jv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import fmin_l_bfgs_b as fmin  # minimizar
import time  # medir tiempo
from numba import jit  
from scipy.fftpack import fft
from statsmodels.stats.correlation_tools import cov_nearest
import seaborn as sb
import scipy.signal as sgnl
from scipy import signal
import pandas as pd
import pylab as plot
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import copy

class GPLP:
    
    # Class Attribute none yet

    # Initializer / Instance Attributes
    
    def __init__(self, space_input, space_output, window, kernel  = "rbf", windowshape = "square", shift = False, grid_num = 5000):
        #Raw data and important values
        
        #Time domain
        self.offset = np.median(space_input)
        self.x = space_input - self.offset #Instantes de muestreo
        self.y = space_output #Valores de las muestras


        self.Nx = len(self.x)  #Numero de muestras
        self.time = np.linspace(np.min(self.x), np.max(self.x), grid_num) #Grilla mas fina donde haremos la interpolacion
        self.T = np.abs(np.max(self.x) - np.min(self.x))  # Tiempo total de muestreo
        self.grid_num = grid_num # Guardamos la resolucion de la grilla
        
        #Freq Domain 
        self.spectra_domain =np.linspace(0.0, self.Nx/(2.0*self.T), int(self.Nx/2) ) #Dominio de las frecuencias relevantes (Segun Nyquist)
        self.spectra = 2.0/self.Nx * np.abs(fft(self.y)[:int(self.Nx/2)]) #Transformada de Fourier
        self.shift = 0
        
        #Parameters 
        self.window = window   #Ventana
        self.sigma = 3*np.std(self.y)  #Primer candidato a variable de escala
        self.gamma = 1/2/((np.max(self.x)-np.min(self.x))/self.Nx)**2 #Primer candidato a variable de senisbilidad (o suavidad)
        self.noise = np.std(self.y)/10 # Ruido intrinseco
        
        #Kernel
        self.kernel_name= kernel #Definimos el kernel que vamos a ocupar, ojo kernel es un string
        self.windowshape = windowshape  #La forma de la ventana: Circular, Triangular, Cuadrada, Hahn

        self.shift = shift #low-pass is default, band-pass if True: Muy importante! Permite acceder a otras frecuencias
        [func, conv] = assign_kernel(kernel, windowshape, shift) #Asignamos el kernel del proceso GPLP segun el nombre y la ventana que nos pasan
        self.kernel = func #Asignacion del kernel base
        self.conv_kernel = conv #Asignacion del kernel asociado al GPLP
                
        #Post training
        
        self.error_bar = None
        self.K = None #Matriz de gram
        self.invK = None #Inversa de la matriz de gram
        self.filtered = None #La señal filtrada en el tiempo
        self.filt_spect_dom =np.linspace(0.0, self.grid_num/(2.0*self.T), int(self.grid_num/2) )
        self.filt_spect = None #Freq Domain 
                                                  
        
        #Numerical bullshit
        
        self.min_noise = 0.005 #Esto pa que no se caiga el sinc
        
  #  def __add__(self,other):
   #     return MyNum(self.num+other.num)
        
    def __rmul__(self, other): #Podemos multiplicar señales
        another_signal = copy.deepcopy(self)     
        
        if another_signal.error_bar is None:
            pass
        else:
            another_signal.error_bar *= other
            
        if another_signal.K is None:
            pass
        else:
            another_signal.K *= other
            another_signal.invK *= other

        another_signal.filtered *= other #Time Domain
        another_signal.filt_spect *= other #Freq Domain 
        
        return another_signal
        
    def like_SE(self,theta): #POR HACER: DEFINIR UNA PRIORI SUPER DIFUSA
        
        sigma_noise, gamma_1, sig_1 = np.exp(theta) #Pasamos los parametros
        
#         if self.kernel == "square_exp":
#             Gram = K_SE(self.x, self.x, gamma=gamma_1,
#             sigma=sig_1) + sigma_noise**2 * np.identity(self.Nx) + self.min_noise*np.identity(self.Nx)
#         elif self.kernel == "mattern1":
#             Gram = Mattern1_SE(self.x, self.x, gamma=gamma_1,
#             sigma=sig_1) + sigma_noise**2 * np.identity(self.Nx) + self.min_noise*np.identity(self.Nx)
            
        Gram = self.kernel(self.x, self.x, gamma = gamma_1, sigma = sig_1) #Calculamos la gram
        
        Gram += sigma_noise**2 * np.identity(self.Nx) + self.min_noise*np.identity(self.Nx) #Regularizamos un poquillo
        # inverse with cholesky
        cGg = np.linalg.cholesky(Gram)
        invGram = np.linalg.inv(cGg.T) @ np.linalg.inv(cGg)
        # nll
        nll = 2 * np.log(np.diag(cGg)).sum() + self.y.T @ (invGram @ self.y) #Primer termino penaliza parametros, el segundo termino penaliza el error
        return 0.5 * nll + 0.5 * len(self.y) * np.log(2 * np.pi)

        
    def train(self):
        # Here we perform Kernel Regression with the priori kernel with parameters that minimize the -log(likelihood)
        # Likelihood = 2 * log (diag(Cholesky (Gram))).sum + y.T* invGram * y
        
        # fixed args of function
        args = (self.y, self.x)
        
        
        # initial point
        params0 = np.asarray([self.noise, self.gamma, self.sigma])
        X0 = np.log(params0)

        print('Condicion inicial optimizador: ', params0)

        #Minimizamos la neg_log likelihood
        time_GP = time.time()
        X_opt, f_GP, data = fmin(
            self.like_SE,
            X0,
            None, 
            approx_grad=True)
        #   disp=1,
         #   factr=0.00000001 / (2.22E-12),
          #  maxiter=1000)
        time_GP = time.time() - time_GP
        print("Tiempo entrenamiento {:.4f} (s)".format(time_GP))

        sigma_n_GP_opt, gamma_opt, sigma_opt = np.exp(X_opt)
        print('Hiperparametros encontrados: [noise, gamma, sigma] ', np.exp(X_opt), 'NLL: ', f_GP)

        self.sigma =np.exp(X_opt)[2]
        self.gamma= np.exp(X_opt)[1]
        self.noise = np.exp(X_opt)[0] + self.min_noise
        
        self.K = self.kernel(self.x, self.x, self.gamma, self.sigma) + self.noise*np.eye(self.Nx) 
        self.invK = np.linalg.inv(self.K)
 
    #Acá calculamos la posterior
    def filt(self, eval_postcov = True, smoothing = False):
        noise_var = 1 #conditioning propblems
        m = len(self.y)
   
        K_ry = self.conv_kernel(self.sigma, self.gamma,self.window, self.time, self.x, self.shift, flag=2)   
        K_ry = K_ry.T
        
        self.filtered= np.matmul(K_ry,self.invK).dot(self.y)

        #Aca calculamos la varianza posterior si nos da la gana
        if eval_postcov == True:

            M = K_ry@np.linalg.solve(self.K+noise_var*np.eye(m), K_ry.T) #Here we use noise_var as there may be conditioning problems
            
            K_rr = self.conv_kernel(self.sigma, self.gamma,self.window, self.time, self.x, self.shift, flag=1)
            # print(np.linalg.eigvals(K_rr).min())
            if smoothing == True:
                K_rr_parche = cov_nearest(K_rr) +1e-8*np.eye(len(self.time))
                self.Covariance = (K_rr_parche - M)
            else:
                self.Covariance = K_rr - M
            self.error_bar =2*np.sqrt(np.diag(self.Covariance))   
        else:
            pass
        
        self.filt_spect = 2.0/self.grid_num * np.abs(fft(self.filtered)[:int(self.grid_num/2)])      
        
    #Este par de funciones igual nunca las ocupo
    def plot_spectra(self):
        plt.plot(self.spectra_domain, self.spectra,'g', label = 'Ground truth spectrum')
        plt.plot(self.filt_spect_dom, self.filt_spect,'r', label = 'GPLP')
        
    def plot_signal(self):
        if self.filtered is None:
            plt.plot(self.x, self.y)
        else:
            plt.plot(self.x, self.y)
            plt.plot(self.time, self.filtered)

        
#################### Different Kernel Priors #############################################################################################        
        
def K_SE(a, b, gamma=1. / 2, sigma=1):
    """
    Squared Exponential kernel
    Returns the gram matrix given by the kernel
    k(a,b) = sigma**2*exp(-gamma*(a-b)**2)
    Note that: gamma = 1 /(2*lengthscale**2)
    
    Inputs:
    a:(numpy array)   Array length n_a with first input
    b:(numpy array)   Array length n_b with second input
    gamma:(float)     Kernel parameter
    sigma:(float)     Kernel parameter, signal variance

    Returns:
    (numpy array) n_a X n_b gram matrix where element
    [i,j] = k(a[i], b[j])
    """
    # transform to array if a single point
    if np.ndim(a) == 0: a = np.array([a])
    if np.ndim(b) == 0: b = np.array([b])
    # create matrix
    gram = np.zeros((len(a), len(b)))
    # compute
    gram = sigma*np.exp(-gamma * (np.subtract.outer(a,b))**2)
    # condition if a single point
    if (len(a) == 1) or (len(b) == 1):
        return gram.reshape(-1)
    else:
        return gram
    
def Mattern1_SE(a,b, gamma= 1./2, sigma=1):
    """
    Mattern v=1/2 kernel
    Returns the gram matrix given by the kernel
    k(a,b) =
    Note that: gamma = 1 /(2*lengthscale**2)
    
    Inputs:
    a:(numpy array)   Array length n_a with first input
    b:(numpy array)   Array length n_b with second input
    gamma:(float)     Kernel parameter
    sigma:(float)     Kernel parameter, signal variance

    Returns:
    (numpy array) n_a X n_b gram matrix where element
    [i,j] = k(a[i], b[j])
    """
    # transform to array if a single point
    if np.ndim(a) == 0: a = np.array([a])
    if np.ndim(b) == 0: b = np.array([b])
    # create matrix
    gram = np.zeros((len(a), len(b)))
    # compute
    gram = sigma*np.exp(-gamma *np.absolute(np.subtract.outer(a,b))*np.sqrt(3))*(1+np.sqrt(3)*gamma*np.absolute(np.subtract.outer(a,b)))
    # condition if a single point
    if (len(a) == 1) or (len(b) == 1):
        return gram.reshape(-1)
    else:
        return gram
    
    
############################### FUnctions to make synthetic data ######################################################################    

def synth_data(low_freqs, high_freqs, coefs_low=None, coefs_high= None, L= 40, n=1000, sample = .25, noise_var = 0.01, random_sampling = False, W = None, W_coeffs = None, sampling_scheme = "full_random"): 
    
    #low_freqs and high_freqs are arrays containing the frequencies (in HZ) for each sine_wave
    #coefs_high and coefs_low are arrays containing the coefficients that go with each frequency (len(coeffs_i) = len(i_freqs))
    #(-L,L) is the time domain of the signal
    #n is the resolution of the grid (The domain is divided in n+1 segments)
    #sample is the fraction to be sampled from this grid 
    #noise_var corresponds to the variance of the white noise added in each observation
    #random_sampling determines if the sample is taken unifromly or randomized 
    #W is an array of coefficients inside each sinc component
    
    m=int(n*sample) #Puntos a samplear

    delta = int(n/m)


    x = np.linspace(-L,L,n)
    
    # Zona de line spectra
    if coefs_low is None:
        coefs_low = np.ones(len(low_freqs))
    if coefs_high is None:
        coefs_high = np.ones(len(high_freqs))
    f_baja = np.zeros(n)
    f_alta = np.zeros(n)
    for i in range(len(low_freqs)):
        f_baja += coefs_low[i]*np.cos(low_freqs[i]*2*np.pi*x)
    for i in range(len(high_freqs)):
        f_alta += coefs_high[i]*np.cos(high_freqs[i]*2*np.pi*x)
    f = np.zeros(n)
    f = f_alta + f_baja
    f += np.sqrt(noise_var)*np.random.randn(n)#Señal suma de una frecuancia baja mas una alta 
    noise = f - f_alta -f_baja
    
    # Zona de square spectra
    if W is None:
        pass
    else:
        if W_coeffs is None:
            W_coeffs = np.ones(len(W))
        for i in range(len(W)):
            f += W_coeffs[i]*np.sinc(W[i]*x)
    
    #Sampleamos m puntos
    if random_sampling == False:
        y = np.zeros(m)
        positions = np.zeros(m)
        for i in range(m):
            y[i] = f[(delta)*i]
            positions[i] = x[(delta)*i]
    else:
        if sampling_scheme == "full_random":
            positions_init = np.arange(n)
            positions_2 = np.random.choice(positions_init, m, replace =False)
            positions_2.sort()
            positions = np.array([x[i] for i in positions_2])
            y = np.array([f[i] for i in positions_2])
        elif sampling_scheme == "additive":
            positions = []
            position = 0
            while True:
                position = np.random.exponential(scale = 2*L/((np.floor(n*sample))) ) + position
                if position > 2*L:
                    break
                positions.append(position)
            positions = np.array(positions)
            positions += -L
            indexes = np.zeros(len(positions))
            for i in range(len(positions)):
                index = 0
                for j in range(len(x)):
                    if np.abs(positions[i] -x[j])< np.abs(positions[i]-x[index]):
                        index = j
                indexes[i] = index
            y = np.array([f[i] for i in indexes.astype(np.int64)])  
                
        elif sampling_scheme == "jitter":
            positions = np.linspace(-L,L, np.floor(n*sample).astype(np.int64)+1)
            positions = positions[:-1] + L//(n*sample) 
            jitters = np.random.uniform(low = (-L/np.floor(n*sample)) *0.9 , high = (L/np.floor(n*sample))*0.9, size = np.floor(n*sample).astype(np.int64))
            positions = positions + jitters
            indexes = np.zeros(len(positions))
            for i in range(len(positions)):
                index = 0
                for j in range(len(x)):
                    if np.abs(positions[i] -x[j])< np.abs(positions[i]-x[index]):
                        index = j
                indexes[i] = index
            y = np.array([f[i] for i in indexes.astype(np.int64)])
            
       
    P = np.subtract.outer(x, positions)

    nyq = (m/(2*L))
    
    return (y, positions, P, f, f_alta, f_baja, x,nyq, noise)


#######################################################################################################################################

#Kernel Functions convolved with diferent windows

#Esta se calcula inmediatamente!
def conv_rbf_square(sigma, gamma,window, time, x, shift, flag):
    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x,time)
        n = t.shape
    resp = np.zeros(shape=n)
    C = sigma*0.5*np.exp(-(t**2)*gamma)
    indexes =  np.abs(t)  <26*np.sqrt(1/gamma)

    s1 = (1/(np.sqrt(1/gamma)))*(np.pi*window*(1/gamma) - t[indexes]*1j) 
    s2 = s1.conjugate()
    s3 = C[indexes]* (special.erf(s1) + special.erf(s2))
    resp[indexes] = s3.real

    return resp

def conv_rbf_square_shift(sigma, gamma, window, time, x, shift, flag):
    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x, time)
        n = t.shape
    resp = np.zeros(shape=n)
    C = sigma*(0.25)*np.exp(-(t**2)*gamma)
    indexes =  np.abs(t)  <26*np.sqrt(1/gamma)

    A = (window * np.pi)/(np.sqrt(gamma))
    B = (shift * np.pi/ (np.sqrt(gamma)) )
    D = t[indexes]*1j*np.sqrt(gamma)

    s1 = A + B + D
    s2 = A - B + D
    s3 = A + B - D
    s4 = A - B - D


    s5 = 2*C[indexes]* (special.erf(s1) + special.erf(s2)+special.erf(s3) + special.erf(s4))
    resp[indexes] = s5.real

    return resp

#De acá se calcula ocupando la transformada discreta de FOurier basicamente
def conv_rbf_circle(sigma, gamma, window, time, x, shift,flag):

    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x, time)
        n = t.shape
    resp = np.zeros(shape=n)

    indexes =  np.abs(t)  <26*np.sqrt(1/gamma) #Esto era importante, y una molestia asociada al Sinc.

    b= window
    K = sigma/np.sqrt(gamma)

    f = lambda x: (np.pi)*special.j1(2*np.pi*b*x)/(2*x*np.pi) #Este es el semicirculo en tiempo

    exes, weights = np.polynomial.hermite.hermgauss(50) #Se ocupa Hermite Gauss porque queremos integrar f respecto a el RBF, que es una Gaussiana
    #El factor K esta para ajustar la gaussiana
    
    g = lambda T: (f((1/np.sqrt(gamma))* exes + T)*weights).sum()
    

    for i in range(len(t[:,0])):
        for j in range(len(t[0,:])):
            if indexes[i,j]==0:
                pass
            else:
                resp[i,j] = K*g(t[i,j])#gp.predict(t[i,j], return_std=False)[0]

    return resp

def conv_rbf_circle_shift(sigma, gamma, window, time, x, shift,flag):

    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x, time)
        n = t.shape
    resp = np.zeros(shape=n)

    indexes =  np.abs(t)  <26*np.sqrt(1/gamma)

    b= window
    K = sigma/np.sqrt(gamma)
   
    f = lambda x: np.pi*(jv(1,2*np.pi*x*b)/(np.pi*x))*np.cos(2*np.pi*shift*x)

    exes, weights = np.polynomial.hermite.hermgauss(20)
    
    g = lambda T: (f((1/np.sqrt(gamma))* exes + T)*weights).sum()

    for i in range(len(t[:,0])):
        for j in range(len(t[0,:])):
            if indexes[i,j]==0:
                pass
            else:
                resp[i,j] = K*g(t[i,j])
    return resp

def conv_rbf_triangle(sigma, gamma, window, time, x, shift,flag):

    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x, time)
        n = t.shape
    resp = np.zeros(shape=n)

    indexes =  np.abs(t)  <26*np.sqrt(1/gamma)

    b= window
    K = sigma/np.sqrt(gamma)
    
    f = lambda x: (b) * (np.sinc(b*x)**2)

    exes, weights = np.polynomial.hermite.hermgauss(100)
    
    g = lambda T: (f((1/np.sqrt(gamma))* exes + T)*weights).sum()
    
    for i in range(len(t[:,0])):
        for j in range(len(t[0,:])):
            if indexes[i,j]==0:
                pass
            else:
                resp[i,j] = K*g(t[i,j])

    return resp

def conv_rbf_triangle_shift(sigma, gamma, window, time, x, shift,flag):

    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x, time)
        n = t.shape
    resp = np.zeros(shape=n)

    indexes =  np.abs(t)  <26*np.sqrt(1/gamma)

    b= window
    K = sigma/np.sqrt(gamma)
    
    f = lambda x:  (2*b)*(np.sinc(b*x))**2 * np.cos(2*np.pi*shift*x)

    exes, weights = np.polynomial.hermite.hermgauss(100)
    
    g = lambda T: (f( (1/np.sqrt(gamma)) * exes + T)*weights).sum()
    
    for i in range(len(t[:,0])):
        for j in range(len(t[0,:])):
            if indexes[i,j]==0:
                pass
            else:
                resp[i,j] = K*g(t[i,j])

    return resp

def conv_mattern_square(sigma, gamma , window, time, x, shift, flag):
    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x, time)
        n = t.shape
    resp = np.zeros(shape=n)

    
    Nn= 4000
    delta = window/(2*Nn)
    
    Tt = np.linspace(shift + delta ,shift + window - delta ,Nn)
    
    S = lambda x: sigma*(24*gamma**3)/((3*gamma**2 + (2*np.pi)**2 * (x) )**2)# Priori kernel PSD
    
    basis = lambda x, t: np.cos(2*np.pi * x*t)
    
    
    for i in range(len(t[:,0])):
        for j in range(len(t[0,:])):
            resp[i,j] = 4*(S(Tt). dot(basis(Tt, t[i,j])) )*(np.sinc(2*delta*t[i,j])*(delta))

    return resp

def conv_mattern_circle(sigma, gamma , window, time, x, shift, flag):
    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x, time)
        n = t.shape
    resp = np.zeros(shape=n)

    
    Nn= 4000
    delta = window/(2*Nn)
    
    Tt = np.linspace(delta ,window - delta ,Nn)

    #Escribimos el PSD del kernel objetivo (en este caso Mattern)
    S = lambda x: sigma*(24*gamma**3)/((3*gamma**2 + (2*np.pi)**2 * (x) )**2)*(np.sqrt(window**2 - x**2)/window)# Priori kernel PSD

    #Nos definimos distintos shift de coseno que van a empujar unos rectangulos (sincs en tiempo) para poder
    #calcular la integral

    basis = lambda x, t: np.cos(2*np.pi * x*t)
    

    #Es una integral de Riemann entera rasca nada mas (En frecuencia, porque en tiempo parece mas complicao) 
    for i in range(len(t[:,0])):
        for j in range(len(t[0,:])):
            resp[i,j] = 4*(S(Tt). dot(basis(Tt, t[i,j])) )*(np.sinc(2*delta*t[i,j])*(delta))

    return resp

def prueba_kernel(sigma, gamma , window, time, x, shift, flag):
    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x, time)
        n = t.shape
    resp = np.zeros(shape=n)

    #indexes =  np.abs(t)  <26*np.sqrt(1/gamma)
    
    b= window
    K = sigma/(gamma*np.sqrt(3))
    
    f = lambda x:  (2*b)*(np.sinc(2*b*x)) * np.cos(2*np.pi*shift*x)

    exes1, weights1 = np.polynomial.laguerre.laggauss(150)
    
    exes2, weights2 = special.roots_genlaguerre(150, 1)
    
    
    g1= lambda T: (f( (1/(gamma*np.sqrt(3) )) * exes1 + T)*weights1).sum() + (f( (-1/(gamma*np.sqrt(3))) * exes1 + T)*weights1).sum()
    
    g2= lambda T: (f( (1/(gamma*np.sqrt(3))) * exes2 + T)*weights2).sum() + (f( (-1/(gamma*np.sqrt(3))) * exes2 + T)*weights2).sum()

    
    for i in range(len(t[:,0])):
        for j in range(len(t[0,:])):
            resp[i,j] = K*( g1(t[i,j]) + g2(t[i,j]) )

    return resp
    
#     Nn= 3000
#     delta = window/(2*Nn)
    
#     Tt = np.linspace(0 + delta ,window - delta ,Nn)
    
#     S = lambda x: sigma*np.sqrt(np.pi/gamma)*np.exp(-((np.pi**2)/gamma)*x**2) # Priori kernel PSD
    
#     basis = lambda x, t: np.cos(2*np.pi * x*t)
    
    
#     for i in range(len(t[:,0])):
#         for j in range(len(t[0,:])):
#                 resp[i,j] = 4*(S(Tt). dot(basis(Tt, t[i,j])))*(np.sinc(2*delta*t[i,j])*(delta))

#     return resp

    
    
    

    

# def conv_mattern_square(sigma, gamma, window, time, x, shift, flag):
#      if flag==0:
#         t= np.abs(np.subtract.outer(x, x))
#         n = t.shape
#     elif flag == 1:
#         t= np.abs(np.subtract.outer(time, time)) 
#         n= t.shape
#     elif flag == 2:
#         t = np.subtract.outer(x, time)
#         n = t.shape
#     resp = np.zeros(shape=n)

#     indexes =  np.abs(t)  <26*np.sqrt(1/gamma)

#     b= window
#     alpha= 0.5*gamma**2
#     K = sigma
    
#     f = lambda x: (2*b) * (np.sinc(b*x))

#     exes, weights = np.polynomial.laguerre.laggauss(100)

    
#     g = lambda T: (f((1/alpha)* exes + T)*weights).sum()
    
#     for i in range(len(t[:,0])):
#         for j in range(len(t[0,:])):
#             if indexes[i,j]==0:
#                 pass
#             else:
#                 resp[i,j] = K*g(t[i,j])

#     return resp
    
#     pass

def conv_mattern_square_shift(sigma, gamma, window, time, x, shift, flag):
    pass

def sinc_kernel(sigma, gamma,window, time, x, shift, flag):
    if flag==0:
        t= np.abs(np.subtract.outer(x, x))
        n = t.shape
    elif flag == 1:
        t= np.abs(np.subtract.outer(time, time)) 
        n= t.shape
    elif flag == 2:
        t = np.subtract.outer(x, time)
        n = t.shape
    resp = sigma*np.sinc(window*t)
    return resp
    
## From here we use numerical integration  ##



## Assign a function to the object ##

def assign_kernel(kernel, windowshape, shift):
    if kernel == "rbf":
        func = K_SE
        if windowshape == "square" and shift == False:
            conv = conv_rbf_square
        elif windowshape == "square" and shift == True:
            conv = conv_rbf_square_shift
        elif windowshape == "circle" and shift == False:
            conv = conv_rbf_circle
        elif windowshape == "circle" and shift == True:
            conv = conv_rbf_circle_shift
        elif windowshape == "triangle" and shift == False:
            conv = conv_rbf_triangle 
        elif windowshape == "triangle" and shift == True:
            conv = conv_rbf_triangle_shift
        else:
            raise ValueError('Invalid windowshape or shift value')

            
    elif kernel == "mattern1":
        func = Mattern1_SE
        if windowshape == "square" and shift == False:
            conv = conv_mattern_square
        elif windowshape == "square" and shift == True:
            conv = conv_mattern_square
        elif windowshape == "circle" and shift == False:
            conv = conv_mattern_circle
        elif windowshape == "circle" and shift == True:
            conv = conv_mattern_circle
        elif windowshape == "triangle" and shift == False:
            conv = conv_mattern_triangle 
        elif windowshape == "triangle" and shift == True:
            conv = conv_mattern_triangle
        else:
            raise ValueError('Invalid windowshape or shift value')
    elif kernel =="prueba":
        func = K_SE
        conv =prueba_kernel
    elif kernel =="sinc":
        func = K_SE
        conv = sinc_kernel
    else:
        raise ValueError('Invalid kernel')        
    return [func, conv]