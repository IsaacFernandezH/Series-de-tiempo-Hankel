#Isaac Fernandez Hernandez

import pandas     as pd
import numpy      as np

def load_cnf_softmax():
  par_sof=[]
  print("Cargando el archivo de configuracion")
  par = np.genfromtxt('cnf_softmax.csv',delimiter=',')
  par_sof.append(np.float64(par[0])) # % of training data
  par_sof.append(np.int16(par[1])) # Iteration number
  par_sof.append(np.float64(par[2])) # Learning rate (mu)
  return par_sof

# Load data 
def load_data_trn(ptrain):
  print("Cargando la data Dinp y Dout")
  dinp = np.genfromtxt('dinp.csv',delimiter=',')
  dout = np.genfromtxt('dout.csv',delimiter=',')

  dinp_train, dinp_test = np.vsplit(dinp,[int(np.shape(dinp)[0]*ptrain)])
  dout_train, dout_test = np.vsplit(dout,[int(np.shape(dout)[0]*ptrain)])
  save_test_data(dinp_test,dout_test)

  return dinp_train.T, dout_train.T

# save Softmax's weights and Cost
def save_w(w,costo):
  print("Guardando la matriz de pesos")
  print("Guardando el los costos")
  np.savez("w_softmax.npz",w)
  np.savetxt("costo_softmax.csv", costo, delimiter=",", fmt="%.6f")

# Initialize weight    
def iniW(next,prev):
  print("Inicializando la matriz de pesos")
  r = np.sqrt(6/(next + prev))
  w = np.random.rand(next,prev)
  w = w*2*r-r
  return w

# Softmax's gradient
def grad_softmax(x,y,w):    
  z    = np.dot(w,x)
  a    = softmax(z)
  
  Cost = (-1/x.shape[1]) * np.sum(y * np.log(a))

  gW   = (-1/x.shape[1]) * np.dot((y-a),x.T)
  return gW,Cost

# Calculate Softmax
def softmax(z):
  exp_z = np.exp(z-np.max(z))
  return (exp_z/exp_z.sum(axis=0,keepdims=True))

# Softmax's training
def train_softmax(x,y,param):
  w = iniW(y.shape[0],x.shape[0])
  beta1 = 0.9
  beta2 = 0.999
  v = np.zeros_like(w)
  s = np.zeros_like(w)
  epsilon = 1e-7
  costo = []
  print("Comienzo del entrenamiento")

  for i in range(1,param[1]+1):
    gw, cost = grad_softmax(x,y,w)

    v = beta1 * v + (1 - beta1) * gw
    s = beta2 * s + (1 - beta2) * gw**2

    gadam = np.sqrt(1-beta2**i)/(1-beta1**i)*v/np.sqrt(s+epsilon)

    w = w - param[2] * gadam
    costo.append(cost)
  return w,costo

# Save Data based on Hankel's features
def save_test_data(Dinp,Dout):
  print("Escribiendo las nuevas caracteristicas en el archivo dinp.csv")
  np.savetxt("dinp_test.csv", Dinp, delimiter=",", fmt="%.6f")
  print("Escribiendo las label binarias en el archivo dout.csv")
  np.savetxt("dout_test.csv", Dout, delimiter=",", fmt="%i")
  return

# Beginning ...
def main():
  param  = load_cnf_softmax()            
  xe,ye  = load_data_trn(param[0])       
  W,cost = train_softmax(xe,ye,param)
  save_w(W,cost)
  print("Fin de la etapa de entrenamiento")

if __name__ == '__main__':   
	 main()
