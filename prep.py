#Isaac Fernandez Hernandez

import pandas     as pd
import numpy      as np

# Load data time series
def load_tseries(n):
  dinp = []
  param  = load_cnf_prep()
  print("Cargando la data de las clases y generacion de caracteristicas")
  param[2] = 4
  for i in range(1,n+1):
    st = np.genfromtxt('Data/Clase%d.csv'%i,delimiter=',')
    dinp.append(get_features(st,param[0],np.shape(st)[0],param[1],param[2]))
  print("Creando labels binarias")
  dinp, dout = binary_label(dinp)
  return dinp, dout

# Create Features
# data: Series de tiempo de la clase c
# ns:   Numero de series a utilizar por clase/data

def get_features(data,lss,ns,fs,l):
  m = ns if ns < len(data) else len(data)
  for i in range(0,m):
    features = hankel_features(data[i,:],lss,fs,l) if i == 0 else np.vstack((features,hankel_features(data[i,:],lss,fs,l)))
  return features

# Hankel's features
def hankel_features(serie,lss,fs,l):
  l_serie = np.shape(serie)[0]
  nss = int(l_serie/lss)
  sub_series = np.reshape(serie,(nss,lss))
  for i in range(0,np.shape(sub_series)[0]):
    c,ce = cfourier(sub_series[i],fs,l)
    u, c_sv, v = np.linalg.svd(c,False)
    ce = np.hstack((ce,c_sv)).ravel()
    ce = np.hstack((ce,spectral_entropy(sub_series[i]))).ravel()
    f = ce if i == 0 else np.vstack((f,ce))
  return f

def cfourier(serie,fs,l):
  n = len(serie)
  idx = np.round((np.arange(1,l+1) * (fs/(l*2))) * n/fs)
  j = 0
  for i in idx:
    serie,c,ce = fpbi(serie,n,i)
    if j == 0:
      cp = c
      cet = ce
      j = 1
    else:
      cp = np.vstack((cp,c))
      cet = np.hstack((cet,ce))
  return cp,cet

def fpbi(serie,n,idx):
  F = np.fft.fft(serie)
  F[int(idx+1):int(n/2)]=0
  F[int(n/2+1):int(n-idx)]=0
  c = np.fft.ifft(F).real
  x = serie - c
  return x,c,spectral_entropy(c)

def hanma (serie,k,m):
  h = np.zeros((k,m))
  for i in range(0,m):
    h[:,i] = serie[i:k+i]
  return h

# spectral entropy
def spectral_entropy(x):
  n = len(x) 
  fhat = np.fft.fft(x)
  fhat = fhat[0:int(n/2)]

  A = (np.sqrt(fhat * np.conj(fhat)).real)**2
  
  p = A/sum(A)
  p=p[p>0]
  return -1/np.log2(n)*sum(p*np.log2(p))

# Binary Label
def binary_label(dinp):
  n_class = np.shape(dinp)[0]
  n_features = np.shape(dinp)[2]
  for i in range(0,n_class):
    n_class_data = np.shape(dinp[i])[0]
    A = np.zeros((n_class_data,n_class),int)
    A[:,i] = 1
    A = np.hstack((dinp[i],A))
    B = A if i == 0 else np.vstack((B,A))
  np.random.shuffle(B)
  dinp,dout = np.hsplit(B,[n_features])
  return dinp, dout

# Data norm 
def data_norm(data):    
  a = 0.01 
  b = 0.99 
  data_norm = ((data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0)))*(b-a)+a
  return data_norm

# Save Data based on Hankel's features
def save_data(Dinp,Dout):
  print("Nuevas caracteristicas generadas en el archivo dinp.csv")
  np.savetxt("dinp.csv", Dinp, delimiter=",", fmt="%.6f")
  print("Label binarias generadas en el archivo dout.csv")
  np.savetxt("dout.csv", Dout, delimiter=",", fmt="%i")
  return

def load_cnf_prep():
  par_sof=[]
  par = np.genfromtxt('cnf_prep.csv',delimiter=',')
  par_sof.append(np.int16(par[0])) # Largo de las sub-series de tiempo
  par_sof.append(np.int16(par[1])) # Frecuencia de Muestreo
  par_sof.append(np.int16(par[2])) # Sub-banda Fourier.
  return par_sof

def main():
	print("Iniciando el pre-procesamiento")
	#n = int(input("Ingrese el numero de clases:\n"))
	n = 8
	Dinp,Dout = load_tseries(n)	
	Dinp      = data_norm(Dinp)
	save_data(Dinp,Dout)
	print("Pre-procesamiento terminado")
	return pd.DataFrame(Dinp),pd.DataFrame(Dout)

if __name__ == '__main__':   
	 main()


