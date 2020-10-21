import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets , svm , metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Leer datos
data = pd.read_csv("Datasets/mnist_784.csv")
# Definir n m e r o de ejemplos
n_samples = 3500
# vamos a omitir class que es nuestro target o valor deseado
x = np.asanyarray (data.drop(columns =["class"]))[: n_samples ,:]
y = np.asanyarray (data [["class"]])[: n_samples ].ravel()
# Dibujar un ejemplo de manera aleatoria
sample = np.random.randint( n_samples )
plt.imshow(x[sample ].reshape ((28 ,28)) , cmap=plt.cm.gray)
plt.title("Target: %i" % y[sample ])
plt.show ()
# Separar conjuntos de entrenamiento y prueba
xtrain , xtest , ytrain , ytest = train_test_split (x, y, test_size =0.1)
# Instanciar el Pipeline
model = Pipeline ([
("PCA", PCA( n_components =50)) ,
("scaler", StandardScaler()),
("SVM", svm.SVC(gamma =0.0001))])
model.fit(xtrain , ytrain)
# Aplicar metrica al modelo
print("Train: ", model.score(xtrain , ytrain ))
print("Test: ", model.score(xtest , ytest ))
# Hacer predicciones del test
ypred = model.predict(xtest)
# Reporte de C l a s i f i c a c i n
print("Classification report: \n", metrics.classification_report (ytest , ypred ))
# Matrix de Confusion
print("Confusion matrix: \n", metrics.confusion_matrix (ytest , ypred ))
sample = np.random.randint(xtest.shape[0])
plt.imshow(xtest[sample ].reshape ((28 ,28)) , cmap=plt.cm.gray)
plt.title("Prediction: %i" % ypred[sample ])
plt.show ()
# Guardar modelo
import pickle
pickle.dump(model , open("Mnist classifier.sav", "wb"))
