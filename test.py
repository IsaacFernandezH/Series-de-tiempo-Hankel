#Isaac Fernandez Hernandez

import pandas     as pd
import numpy      as np

# Load data 
def load_data_tst():
  dinp = np.genfromtxt('dinp_test.csv',delimiter=',')
  dout = np.genfromtxt('dout_test.csv',delimiter=',')
  return dinp.T, dout.T

#Load weight of Softmax
def load_w():
  file = np.load("w_softmax.npz")
  return file.f.arr_0

# Calculate Softmax
def softmax(x,w):
  z = np.dot(w,x)
  exp_z = np.exp(z-np.max(z))
  return (exp_z/exp_z.sum(axis=0,keepdims=True))

#Confusion matrix
def confusion_matrix(x,y):
  print("Generando matriz de confusion")
  confussion_matrix = np.zeros((x.shape[0],y.shape[0]))
  for real, predicted in zip(x.T,y.T):
    confussion_matrix[np.argmax(real)][np.argmax(predicted)]+=1
  return confussion_matrix

def metricas(x,y):
  Fscore = []
  cf = confusion_matrix(x,y)
  print("Generando metricas")
  for index, caracteristica in enumerate(cf):
      TP          = caracteristica[index]
      FP          = cf.sum(axis=0)[index]-TP
      FN          = cf.sum(axis=1)[index]-TP
      recall      = TP/(TP+FN)
      presition   = TP/(TP+FP)
      Fscore.append(2*(presition*recall)/(presition+recall))
  Fscore.append(np.mean(Fscore))
  metrica = pd.DataFrame(Fscore)
  confussion_matrix = pd.DataFrame(cf)
  print("Matriz de confusion")
  print(confussion_matrix)
  return metrica,confussion_matrix

# Save metrica
def save_metrica(metrica,confussion_matrix):
  metrica.to_csv("fscores.csv", index=False, header=False)
  confussion_matrix.to_csv("cmatriz.csv.",index=False, header=False)
  return

# Beginning ...
def main():
	print("Inicio de la etapa de testeo")
	xv,yv  = load_data_tst()
	W      = load_w()
	zv     = softmax(xv,W)
	metrica, confussion_matrix = metricas(yv,zv)
	print("F1 score")
	print(metrica)
	save_metrica(metrica,confussion_matrix)
	print("Fin de la etapa de testeo")

if __name__ == '__main__':   
	 main()


