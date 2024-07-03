#CODIGO TFG -- Jon Mañueco Rubio 

#CODIGO HMM

#Librerias necesarias.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from hmmlearn import hmm
import itertools
from tqdm import tqdm
import yfinance as yf

# Se importan el precio de las acciones de AAPL

accion = yf.Ticker("AAPL")
data = accion.history(period="max")
data = data.drop(data[data["Open"] == 0].index)
data.describe()

### Ahora se definen los conjuntos de entrenamiento y de test y se definen dos funciones con los que vamos a tratar los datos.

train_size = int(0.95*data.shape[0])# Se utiliza el 95% del conjunto de datos.
print(f'El conjunto de entrenamiento tiene un total de {train_size} datos')
train_data = data.iloc[0:train_size]
test_data = data.iloc[train_size+1:]

#### Esta función extrae los valores que se van a intentar predecir: las variaciones diarias en el precio de la acción. Particularmente se va a predecir el precio de cierre, y se van a utilizar el máximo, el mínimo, junto con el precio de apertura.

def augment_features(dataframe): # Esta función calcula los valoers que vamos a predecir:
    fracocp = (dataframe['Close']-dataframe['Open'])/dataframe['Open']# Para estimar el precio de cierre se va a utilizar el precio de apertura
    frachp = (dataframe['High']-dataframe['Open'])/dataframe['Open']# Idem para el precio de máximo
    fraclp = (dataframe['Open']-dataframe['Low'])/dataframe['Open']# Idem para el  mínimo
    new_dataframe = pd.DataFrame({'delOpenClose': fracocp,
                                 'delHighOpen': frachp,
                                 'delLowOpen': fraclp})
    new_dataframe.set_index(dataframe.index)

    return new_dataframe

def extract_features(dataframe): #La función que lo extrae del df
    return np.column_stack((dataframe['delOpenClose'], dataframe['delHighOpen'], dataframe['delLowOpen']))# Queremos que nos saque tan solo esas tres columnas.

features = extract_features(augment_features(train_data))# Con lo que ya se pueden extraer las características

### Ahora se debe crear el modelo y determinar los hiperparámetros que mejor se adapten al modelo. Para ello se va a realizar un grid search.

def mejor_modelo(features): #Búsqueda de hiperparámetros
    modelo=hmm.GaussianHMM()
    parametros={'n_components': [2, 4,6,8, 10],'covariance_type': ['full', 'tied', 'diag', 'spherical'],}
    grid_search = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5)
    grid_search.fit(features)
    print(f'Los mejores parámetros son {grid_search.best_params_}')
    return grid_search.best_estimator_,grid_search.best_params_

modelo,parametros=mejor_modelo(features) # Extraemos el modelo

### Una vez elegido el mejor modelo y habiendo entrenado ya a este, es el momento de realizar las primeras predicciones. Para ello se van a definir el conjunto de valores que pueden tomar las observaciones, y se va a utilizar un proceso similar al de la determinación de los hiperparámetros para ver qué combinación de los mismos dan un mejor resultado sobre el conjunto de test.

num_latent_days_values = [10,20,30] # El número de días que se utilizan para predecir el siguiente
num_steps_values = [10,20,30,50] # El número de posibles outcomes que tiene cada uno de los features definidos.
num_days_to_predict = 100 # Se va a fijar el número de días que se quieren predecir a 100.

#### En primer lugar se extraen los datos

test_augmented = augment_features(test_data) # Extraemos lso posibles valores de cada una de las variables
train_augmented=augment_features(train_data)
fracocp = test_augmented['delOpenClose']
frachp = test_augmented['delHighOpen']
fraclp = test_augmented['delLowOpen']

#### Sencillamente lo que se hace es comparar todos los parámetros posibles y se oberva para cuáles se obtienen los mejores resultados, utilizando el mismo modelo, y viendo cuál presenta un error menor.

mae_num_steps_latent=np.empty([len(num_latent_days_values),len(num_steps_values)]) # Se define la matriz donde se van a ir guardando los errores cometidos, tiene dimensiones 6x4
precios=np.empty([len(num_steps_values),num_days_to_predict]) # Tiene dimensiones 4x100
mae_num_steps=np.empty([len(num_steps_values),1]) #Vector de 4x1
best_model=[]# La estructura va a ser Latente, Predictions, Num_Steps
mejor_num_steps=np.empty([len(num_latent_days_values),1])
mejor_precio=np.empty([len(num_latent_days_values),num_days_to_predict])
mejor_error=np.empty([len(num_latent_days_values),1])

i=0
for baseline_num_latent_days in num_latent_days_values:
    print(f'Vamos por {baseline_num_latent_days} número de días latentes.')
    j=0
    for num_step in num_steps_values:
        print(f'Vamos por {num_step} número de intervalos.')
        sample_space_fracocp = np.linspace(fracocp.min(), fracocp.max(), num_step)
        sample_space_fraclp = np.linspace(fraclp.min(), frachp.max(), int(num_step/5))
        sample_space_frachp = np.linspace(frachp.min(), frachp.max(), int(num_step/5))
        possible_outcomes = np.array(list(itertools.product(sample_space_fracocp, sample_space_frachp, sample_space_fraclp)))
        outcome_scores=np.empty(possible_outcomes.shape[0])
        for k in tqdm(range(num_days_to_predict)):
            # Calculate start and end indices
            previous_data_start_index = max(0, k - baseline_num_latent_days)
            previous_data_end_index = max(0, k)
            # Acquire test data features for these days
            previous_data = extract_features(augment_features(test_data.iloc[previous_data_start_index:previous_data_end_index]))

            l=0
            for outcome in possible_outcomes:
                # Append each outcome one by one with replacement to see which sequence generates the highest score
                total_data = np.row_stack((previous_data, outcome))
                outcome_scores[l]=modelo.score(total_data)
                l+=1
            # Take the most probable outcome as the one with the highest score
            most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)]
            precios[j,k]=test_data.iloc[k]['Open'] * (1 + most_probable_outcome[0])
        mae_num_steps[j,0]=(abs(test_data.iloc[0:num_days_to_predict]['Close'] - precios[j,:])).mean()
        j+=1
    mejor_error[i,0]=min(mae_num_steps)[0]
    mejor_precio[i,:]=precios[np.argmin(mae_num_steps),:]
    mejor_num_steps[i,0]=num_steps_values[np.argmin(mae_num_steps)]
    mae_num_steps_latent[i,:]= mae_num_steps[:,0]
    i+=1

# Se elige el mejor modelo

best_model=[num_latent_days_values[np.argmin(mejor_error)],mejor_num_steps[np.argmin(mejor_error)],mejor_precio[np.argmin(mejor_error)],min(mejor_error)] # Se elige el mdejor modelo

# Se plotea la gráfica

plt.figure(figsize=(30,10), dpi=80) # Se plotean los valores esperados frente a los reales


x_axis = np.array(test_data.index[0:num_days_to_predict], dtype='datetime64[ms]')
plt.plot(x_axis, test_data.iloc[0:num_days_to_predict]['Close'], 'b+-', label="Actual close prices")
plt.plot(x_axis, best_model[2], 'ro-', label="Predicted close prices")
plt.xlabel('Tiempo')
plt.ylabel('Precio €')
plt.legend()
plt.show()

### Ahora se representa el error en la malla, para ver si se ha determinado de manera correcta el mínimo en el error.

def posicion_min(lista): #Función rápida, habrá que cambiarla
    num=np.argmin(lista)
    x=0
    y=0
    for i in range (0,len(lista)):
        if num< len(lista[i]):
            return [i,num]
        num -=len(lista[i])

pos=posicion_min(mae_num_steps_latent)

plt.contourf(num_latent_days_values,num_steps_values,np.transpose(mae_num_steps_latent),cmap='viridis') # Se plotea el grid con los errores
plt.xlabel('n_latent_days')
plt.ylabel('num_steps_values')
plt.colorbar()
#plt.scatter(num_latent_days_values,np.ones_like(num_latent_days_values)*num_steps_values[pos[0]],marker='o', color='red',label='Punto destacado')
plt.legend()

plt.show()

#### Y lo que se observa es que el resultado del modelo es independiente del numero de dias que  se usen para predecirlo, siempre se eligen las mismas combinaciones.

## Ahora vamos a intentar hacer predicciones para dos días, a ver cómo se desenvuelve el modelo.

# Primero se introducen los parámetros óptimos y el conjunto de valores posibles que se pueden tomar en un día. Lo vamos a hacer por lo tanto con el modelo que ha resultado óptimo para predicciones de un día.
num_latent_days=1
num_step=10
sample_space_fracocp_1 = np.linspace(fracocp.min(), fracocp.max(), num_step)
sample_space_fraclp_1 = np.linspace(fraclp.min(), frachp.max(), int(num_step/5))
sample_space_frachp_1 = np.linspace(frachp.min(), frachp.max(), int(num_step/5))
possible_outcomes_1 = np.array(list(itertools.product(sample_space_fracocp_1, sample_space_frachp_1, sample_space_fraclp_1)))
num_days_to_predict=test_data.iloc[:]['Close'].shape[0]
modelo=hmm.GaussianHMM(n_components=parametros['n_components'])
modelo.fit(features)

# Este modelo permite predecir a varios dias vista

def predict_several_days(n,possible_outcomes_1,num_latent_days,num_days_to_predict,test_data,modelo): # Algoritmo de predicción a varios días
    possible_outcomes=np.array(list(itertools.product(possible_outcomes_1,repeat=n)))
    predicted_close_prices = np.empty(n*num_days_to_predict)
    outcome_scores=np.empty(possible_outcomes.shape[0])

    for i in tqdm(range(num_days_to_predict)):
        # Calculate start and end indices
        previous_data_start_index = max(0, n*i - num_latent_days)
        previous_data_end_index = max(0, n*i)
        # Acquire test data features for these days
        previous_data = extract_features(augment_features(test_data.iloc[previous_data_start_index:previous_data_end_index]))
        k=0
        for outcome in possible_outcomes:
            # Append each outcome one by one with replacement to see which sequence generates the highest score
            total_data = np.row_stack((previous_data, outcome))
            outcome_scores[k]=modelo.score(total_data)
            k+=1

        # Take the most probable outcome as the one with the highest score
        most_probable_outcome = possible_outcomes[np.argmax(outcome_scores)]
        for j in range (0,n):
            predicted_close_prices[n*i+j]=(test_data.iloc[n*i+j]['Open'] * (1 + most_probable_outcome[j,0]))
    ae = abs(test_data.iloc[0:n*num_days_to_predict]['Close'] - predicted_close_prices)
    '''
    plt.figure(figsize=(30,10), dpi=80)
    x_axis = np.array(test_data.index[0:n*num_days_to_predict], dtype='datetime64[ms]')
    plt.plot(x_axis, test_data.iloc[0:n*num_days_to_predict]['Close'], 'b+-', label="Actual close prices")
    plt.plot(x_axis, predicted_close_prices, 'ro-', label="Predicted close prices")
    plt.legend()
    plt.show()
    plt.figure(figsize=(30,10), dpi=80)
    plt.plot(x_axis, ae, 'go-', label="Error")
    plt.legend()
    plt.show()
    '''
    return ae,predicted_close_prices

ae,prices=predict_several_days(1,possible_outcomes_1,num_latent_days,num_days_to_predict,test_data,modelo)

### Algoritmo de inversión
'''
Para terminar se plantea el siguiente algoritmo de inversión, que consiste en que para cada uno de los días se realiza una predicción sobre el precio de cierre del día siguiente de la acción. Pueden aparecer dos situaciones:
1. **Las predicciones mejoran al valor de apertura:** En este caso el algoritmo compra a precio de apertura y vende a precio de cierre, inversión a largo.
2. **La predicciones empeoran el valor de apertura:** En este caso el algoritmo vende a precio de apertura y compra a precio de cierre.

Se trata de un algoritmo muy sencillo que siempre compra o vende en el día, y con eso se genera un balance que después se compura de manera global. Para calcular el balance, el margen sobre compra de la operación es:

\begin{equation}
B=\frac{P_{venta}-P_{compra}}{P_{compra}}
\end{equation}

Entonces para el caso de la inversión a largo, resulta claro que se obtiene que en la expresión se reduce a:

\begin{equation}
B=\frac{P_{cierre}-P_{ap}}{P_{ap}}
\end{equation}

Mientras que para la inversión a corto lo que se tiene es:

\begin{equation}
B=\frac{P_{ap}-P_{cierre}}{P_{cierre}}
\end{equation}
'''

def compra_largo(pred,ap,real): # Esta función crea un algoritmo de compra - venta  a largo.
    g=0
    balance=0
    p=0
    for i in range (1,len(pred)):
        if pred[i] > ap[i]: #En este caso compramos a precio real (i-1)
            balance += (real[i]-ap[i])/ap[i]
            if real[i]<ap[i]:
                p+=1
            else:
                g+=1
    #print(f'Al final ganamos {g} días, perdimos {p} días, y el balance fue {balance}.')
    return balance

def venta_corto(pred,ap,real):
    g=0
    balance=0
    p=0
    for i in range (1,len(pred)):
        if pred[i]<ap[i]:
            balance += (ap[i]-real[i])/real[i]
            if real[i]>ap[i]:
                p+=1
            else:
                g+=1
    #print(f'Al final ganamos {g} días, perdimos {p} días, y el balance fue {balance}.')
    return balance

#### Ejecución del algoritmo

#Para testear la eficiencia del algoritmo, se va a probar sobre varias acciones, 200 veces y se van a estudiar las propiedades finales de los balances obtenidos.

# Primero se introducen los parámetros óptimos y el conjunto de valores posibles que se pueden tomar en un día. Lo vamos a hacer por lo tanto con el modelo que ha resultado óptimo para predicciones de un día.
def ejecuta_alg(n):
  resultado=np.empty(n)
  for i in range (n):
    num_latent_days=1
    num_step=10
    sample_space_fracocp_1 = np.linspace(fracocp.min(), fracocp.max(), num_step)
    sample_space_fraclp_1 = np.linspace(fraclp.min(), frachp.max(), int(num_step/5))
    sample_space_frachp_1 = np.linspace(frachp.min(), frachp.max(), int(num_step/5))
    possible_outcomes_1 = np.array(list(itertools.product(sample_space_fracocp_1, sample_space_frachp_1, sample_space_fraclp_1)))
    num_days_to_predict=int(0.05*data.shape[0])-1
    modelo=hmm.GaussianHMM(n_components=10)
    modelo.fit(features)
    ae,prices=predict_several_days(1,possible_outcomes_1,num_latent_days,num_days_to_predict,test_data,modelo)
    balance_1=compra_largo(prices,test_data.iloc[:]['Open'].values,test_data.iloc[:]['Close'].values)
    balance_2=venta_corto(prices,test_data.iloc[:]['Open'].values,test_data.iloc[:]['Close'].values)
    resultado[i]=balance_1+balance_2+1
  return resultado

# Primero se hace para una de las acciones.

resultado=ejecuta_alg(200)

plt.figure(figsize=(30,10), dpi=80)
plt.plot(resultado, 'r+-', label="Actual close prices")
print(f'Los resultados obtenidos han sido \n  Mínimo: { resultado.min()}\n Máximo: {resultado.max()} \n Media: {resultado.mean()}\n Mediana: {np.median(resultado)}\n Cuartil 1/4: {np.quantile(resultado,0.25)}\n Cuartil 3/4: {np.quantile(resultado,0.75)}')

'''
### Test final

Para testear de la manera más verosímil posible se va a ejecutar un experimento en el que se va a repetir este algoritmo sobre 100 empresas aleatorias del S&P 500, y se  van a estudiar los resultados obtenidos:

(OBS): DE MOMENTO NO SE HA PODIDO AUTOMATIZAR EL PROCESO, POR LO QUE SE HAN ELEGIDO UNAS POCAS EMPRESAS DEL NASDAQ
'''

import random


tickers=['MSFT', 'AAPL', 'AMZN', 'GOOG','NFLX','NVDA']


performance={}
# Ahora vamos a elegir los valores aleatorios de la lista con los que vamos a testear el algoritmo
for ticker in tickers:
  print(f'Empezamos con {ticker}')
  accion = yf.Ticker(ticker)
  data = accion.history(period="max")
  data = data.drop(data[data["Open"] == 0].index)
  train_size = int(0.95*data.shape[0])# Se utiliza el 95% del conjunto de datos.
  train_data = data.iloc[0:train_size]
  test_data = data.iloc[train_size+1:]
  features = extract_features(augment_features(train_data))# Con lo que ya se pueden extraer las características
  test_augmented = augment_features(test_data) # Extraemos lso posibles valores de cada una de las variables
  fracocp = test_augmented['delOpenClose']
  frachp = test_augmented['delHighOpen']
  fraclp = test_augmented['delLowOpen']
  resultado=ejecuta_alg(200)
  performance[ticker]=resultado

def ratio_largo(lista):
  ratios=np.empty(len(lista))
  k=0
  for ticker in lista:
    end_date = '2024-02-28' # Si se pone esa fecha, se coge hasta el día anterior
    accion = yf.Ticker(ticker)
    data = accion.history(period='max',end=end_date)
    data = data.drop(data[data["Open"] == 0].index)
    train_size = int(0.95*data.shape[0])# Se utiliza el 95% del conjunto de datos.
    test_data = data.iloc[train_size+1:]
    ratios[k]=test_data.iloc[-1]['Close']/test_data.iloc[0]['Close']
    k+=1
  return ratios
ratio_real=ratio_largo(tickers)

import matplotlib.pyplot as plt

matriz=np.empty([6,len(performance)])


for i in range (0,len(performance)):
  matriz[:,i]=[performance[tickers[i]].max(),performance[tickers[i]].min(),performance[tickers[i]].mean(),np.median(performance[tickers[i]]),np.quantile(performance[tickers[i]],0.25),np.quantile(performance[tickers[i]],0.75)]

dicc={tickers[i]: matriz[:,i] for i in range (0,len(tickers))}
df=pd.DataFrame(dicc)
nrow=3
ncol=2
fig, axs = plt.subplots(nrow, ncol,figsize=(30,15), dpi=80)
color = ["#000080", "#228B22", "#800000", "#4B0082", "#666666", "#F0E68C",'#FFC0CB']
colores_pastel = []
factor_pastel=0.3
for color in color:
    r, g, b = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
    r_pastel = int(r * (1 - factor_pastel) + 255 * factor_pastel)
    g_pastel = int(g * (1 - factor_pastel) + 255 * factor_pastel)
    b_pastel = int(b * (1 - factor_pastel) + 255 * factor_pastel)
    colores_pastel.append("#{:02x}{:02x}{:02x}".format(r_pastel, g_pastel, b_pastel))
color=colores_pastel


lista=['Maximo','Minimo','Media','Mediana','Q1','Q3','Real']
j=0
for i in range (0,len(tickers)):
  if i<(nrow-1):
    axs[i,j].bar(lista,np.concatenate((df.iloc[:][tickers[i]].values,np.array([ratio_real[i]]))),color=color)
    axs[i,j].set(xlabel=tickers[i], ylabel='Veces')

  elif i==(nrow-1):
    axs[i,j].bar(lista,np.concatenate((df.iloc[:][tickers[i]].values,np.array([ratio_real[i]]))),color=color)
    axs[i,j].set(xlabel=tickers[i], ylabel='Veces')
    j+=1
  else:
    axs[i%nrow,j].bar(lista,np.concatenate((df.iloc[:][tickers[i]].values,np.array([ratio_real[i]]))),color=color)
    axs[i%nrow,j].set(xlabel=tickers[i], ylabel='Veces')

i=0
for resultado in performance:
  print(f'Los resultados obtenidos para {resultado} han sido \n  Mínimo: { performance[resultado].min()}\n Máximo: {performance[resultado].max()} \n Media: {performance[resultado].mean()}\n Mediana: {np.median(performance[resultado])}\n Cuartil 1/4: {np.quantile(performance[resultado],0.25)}\n Cuartil 3/4: {np.quantile(performance[resultado],0.75)}\n Compra a largo: {ratio_real[i]}')
  i+=1

  ### Con lo que parece que se han obtenido unos resultados muy buenos, si los comparamos con los ratios que se habrían obtenido simplemente comprando al principio y vendiendo al final. De esta forma, en algunas ocasiones podemos batir incluso el desempeño del algoritmo de compra a largo.


### OBS: Tenían como día límite el 27/02/2024. El eperimento se realizó el 28/02



# CODIGO LSTM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import yfinance as yf
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout,SimpleRNN,Input,Lambda
from sklearn.metrics import mean_squared_error
import math
import keras_tuner
from keras_tuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
import os
import sys
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping

### Importación y tratamiento de los datos

def augment_features(dataframe): # Esta función calcula los valoers que vamos a predecir:
    fracocp = (dataframe['Close']-dataframe['Open'])/dataframe['Open']# Para estimar el precio de cierre se va a utilizar el precio de apertura
    frachp = (dataframe['High']-dataframe['Open'])/dataframe['Open']# Idem para el precio de máximo
    fraclp = (dataframe['Open']-dataframe['Low'])/dataframe['Open']# Idem para el  mínimo
    raw_open=dataframe['Open']# También sacamos el precio de apertura
    raw_close=dataframe['Close'] # Y sacamos el precio de cierre
    new_dataframe = pd.DataFrame({'delOpenClose': fracocp,
                                 'delHighOpen': frachp,
                                 'delLowOpen': fraclp,
                                 'open': raw_open,
                                 'close': raw_close} )
    new_dataframe.set_index(dataframe.index)


    return new_dataframe

def extract_features(dataframe): #La función que lo extrae del df
    stacked_columns= np.column_stack((dataframe['open'], dataframe['close'])) #Se crea una columna que contenga la variación del precio de apertura del día con respecto al cierre del día anterior.
    new_column=np.empty(stacked_columns[:,0].shape)
    new_column[:-1]=(stacked_columns[1:,0]-stacked_columns[:-1,1])/stacked_columns[:-1,0]# Se deja el último elemento vacío.
    matriz=np.column_stack((dataframe['delOpenClose'], dataframe['delHighOpen'], dataframe['delLowOpen'],new_column))
    scaler=MinMaxScaler(feature_range=(-1,1))

    #Esta línea solo si se hace min_max_scaling
    #matriz=scaler.fit_transform(matriz)
    return matriz,scaler

### Creamos una función que nos separe el conjunto de entrenamiento en train y val


### Código final para importar los datos dado un ticker

def hiperparameter_split(X_train,Y_train,split):
    train_size=int(split*X_train.shape[0])
    X_train_hip=X_train[:train_size]
    Y_train_hip=Y_train[:train_size]
    X_val_hip=X_train[train_size:]
    Y_val_hip=Y_train[train_size:]
    return X_train_hip,Y_train_hip,X_val_hip,Y_val_hip

def importacion_datos(ticker):
    #Extraemos los datos
    end_date = '2024-02-28'
    accion = yf.Ticker(ticker)
    data = accion.history(period="max",end=end_date)
    data = data.drop(data[data["Open"] == 0].index)
    #Dividimos entre conjunto de entrenamiento y conjunto de test
    train_size = int(0.95*data.shape[0])# Se utiliza el 95% del conjunto de datos.
    train_data = data.iloc[0:train_size]
    test_data = data.iloc[train_size+1:]
    # Y las adaptamos a las variabled que queremos predecir
    features_train,_ = extract_features(augment_features(train_data))
    features_test,scaler=extract_features(augment_features(test_data))
    # Ahora dividimos entre columnas auxiliares y etiquetas
    x_train=features_train[:,1:]
    X_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    Y_train=features_train[:,0]
    # Definimos ahora los X_test
    x_test=features_test[:,1:]
    X_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    Y_test=features_test[:,0]
    #Ahora vamos a realizar el desajuste temporal para que utilizemos X_t-1 para predecir los Y_t
    X_train=X_train[:-1]
    Y_train=Y_train[1:]
    X_test_aux=np.empty(X_test.shape)
    #Creamos una variable auxiliar para hacer el cambio
    X_test_aux[0]=X_train[-1]
    X_test_aux[1:]=X_test[:-1]
    X_test=X_test_aux #lo devolvemos a la variable original
    #Ahora vamos a dividir el conjunto de entrenamiento en entrenamiento y validación:
    split=0.8
    X_train_hip,Y_train_hip,X_val_hip,Y_val_hip=hiperparameter_split(X_train,Y_train,split)
    #Y  nos sacamos todo de la función, ya tenemos lo que necesitamos
    return X_train,Y_train,X_test,Y_test,X_train_hip,Y_train_hip,X_val_hip,Y_val_hip,train_data,test_data,scaler


X_train,Y_train,X_test,Y_test,X_train_hip,Y_train_hip,X_val_hip,Y_val_hip,train_data,test_data,scaler=importacion_datos('AAPL')

## Selección de hiperparámetros del modelo, entrenamiento, test y análisis

'''
### Se define el modelo sobre el que se va a ha construir los hiperparámetros. Estos son:
    - Número de unidades de la capa de entrada.
    - Número de capas LSTM y número de unidades de cada capa.
    - Dropout de la red a la salida de la última capa LSTM.
    - Dropout previo a la capa de salida.
Idém para las recurrentes.
'''

#### Se definen los modelos de redes que se van a utiliza, salvo el Baseline, que se define in situ.

def build_model_LSTM(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=2,max_value=12,step=2),return_sequences=True, input_shape=(X_train_hip.shape[1],1)))
    for i in range(hp.Int('n_layers', 0, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=2,max_value=12,step=2),return_sequences=True))
    model.add(LSTM(6))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(6))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
    return model

def build_model_RNN(hp):
    model = Sequential()
    model.add(SimpleRNN(hp.Int('input_unit',min_value=2,max_value=12,step=2),return_sequences=True, input_shape=(X_train_hip.shape[1],1)))
    for i in range(hp.Int('n_layers', 0, 4)):
        model.add(SimpleRNN(hp.Int(f'lstm_{i}_units',min_value=2,max_value=12,step=2),return_sequences=True))
    model.add(SimpleRNN(6))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(6))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mse'])
    return model

def build_model_linear():
# Definir la forma de entrada y salida
    input_shape = (X_train_hip.shape[1],)  # Tres columnas de entrada
    output_shape = (1,)  # Una etiqueta de salida

# Definir la capa de entrada
    inputs = Input(shape=input_shape)

# Capa Lambda (passthrough)
    pass_through = Lambda(lambda x: x)(inputs)

# Capa de salida lineal
    outputs = Dense(1, activation='linear')(pass_through)

# Crear el modelo
    model = Model(inputs=inputs, outputs=outputs)

# Compilar el modelo
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

#### Se definen las funciones que proporcionan los algoritmos de cálculo de errores y el algoritmo de compra-venta.

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    #print("The root mean squared error is {}.".format(rmse))
    return rmse
def compra_largo(pred,ap,real): # Esta función crea un algoritmo de compra - venta  a largo.
    g=0
    balance=0
    p=0
    for i in range (1,len(pred)):
        if pred[i] > ap[i]: #En este caso compramos a precio real (i-1)
            balance += (real[i]-ap[i])/ap[i]
            if real[i]<ap[i]:
                p+=1
            else:
                g+=1
    return balance

def venta_corto(pred,ap,real):
    g=0
    balance=0
    p=0
    for i in range (1,len(pred)):
        if pred[i]<ap[i]:
            balance += (ap[i]-real[i])/real[i]
            if real[i]>ap[i]:
                p+=1
            else:
                g+=1
    return balance

#### Se define la función que permite evaluar el modelo utilizando el error cuadrático medio.

def evaluate_model(x,data,model=None,predicciones=None):
    if model==None:
        precios=data[0:]['Open']*(1+predicciones)
    else:
        nuevas_predicciones=model.predict(x)
        nuevas_predicciones=nuevas_predicciones.reshape(nuevas_predicciones.shape[0])
        precios=data[0:]['Open']*(1+nuevas_predicciones)
    return return_rmse(data[0:]['Close'],precios)

#### Se define la función que permite plotear los modelos.

def plot_model(test_data,precios_predicciones):
    plt.figure(figsize=(30,10), dpi=80) # Se plotean los valores esperados frente a los reales
    x_axis = np.array(test_data.index[0:], dtype='datetime64[ms]')
    plt.plot(x_axis, test_data.iloc[0:]['Close'], 'b+-', label="Actual close prices")
    plt.plot(x_axis,precios_predicciones,'ro-', label="Predicted close prices")
    plt.xlabel('Tiempo')
    plt.ylabel('Precio €')
    plt.legend()
    plt.show()
    ae = abs(test_data.iloc[0:]['Close'] - precios_predicciones)
    plt.figure(figsize=(30,10), dpi=80)
    plt.plot(x_axis, ae, 'go-', label="Error")
    plt.legend()
    plt.show()
    print(f'El error absoluto obtenido ha sido {ae.mean()}')

    #### Se define la función que, dado un modelo, entrena, testea y evalúa el modelo en cuestión.

def compile_and_fit(model,X_train,Y_train,X_train_hip,Y_train_hip,X_val_hip,Y_val_hip,X_test,train_data,test_data,dicc,epochs,hip_epochs):
        if model=='LSTM'or model=='RNN':
                if model=='LSTM':
                        early_stopping = EarlyStopping(
                        monitor='mse',
                        patience=5,
                        restore_best_weights=True
                        )
                        modelo=build_model_LSTM
                        tuner= RandomSearch(
                        build_model_LSTM,
                        objective='mse',
                        max_trials=15,
                        executions_per_trial=2,overwrite=True)
                elif model=='RNN':
                        early_stopping = EarlyStopping(
                        monitor='mse',
                        patience=5,
                        restore_best_weights=True
                        )
                        modelo=build_model_RNN
                        tuner= RandomSearch(
                        build_model_RNN,
                        objective='mse',
                        max_trials=15,
                        executions_per_trial=2,overwrite=True,
                        )
                print('Empezamos la búsqueda de hiperparámetros')
                tuner.search(
                x=X_train_hip,
                y=Y_train_hip,
                epochs=5,
                batch_size=16,
                validation_data=(X_val_hip,Y_val_hip),callbacks=[early_stopping]
                )
                best_model = tuner.get_best_models(num_models=1)[0]
                best_model.summary()
                tuner.results_summary()
                best_hps = tuner.get_best_hyperparameters(5)
                # Build the model with the best hp.
                modelo = modelo(best_hps[0])

                # Evaluamos el modelo

                error_val=evaluate_model(X_val_hip,train_data[X_train_hip.shape[0]+1:],model=modelo,predicciones=None)



        # Se entrena el modelo con el conjunto de los datos completo y después se realizan las predicciones.
                print('Empezamos a entrenar el modelo')
                modelo.fit(x=X_train, y=Y_train,epochs=epochs,callbacks=[early_stopping])
                error_test=evaluate_model(X_test,test_data,model=modelo,predicciones=None)
                dicc[model]=np.array([error_val,error_test])

        #Ahora se deshace el cambio para poder comparar con los valores reales.
                predicted_stock_price_scaled = modelo.predict(X_test)

        #Esta linea si no hay escala
                predicted_stock_price=predicted_stock_price_scaled

        #Esta línea solo se ejecuta si se scalea.
                #predicted_stock_price= scaler.inverse_transform(np.column_stack((predicted_stock_price_scaled,X_test[:,0],X_test[:,1],X_test[:,2])))[:,0]
                predicciones=predicted_stock_price.reshape(predicted_stock_price.shape[0])
                precios_predicciones=np.array(test_data.iloc[0:]['Open'])*(1+predicciones)

        if model=='Baseline':
                predicciones_val=np.empty(Y_val_hip.shape[0])
                predicciones_val[0]=Y_train_hip[-1]
                predicciones_val[1:]=Y_val_hip[:-1]

                predicciones=np.empty(Y_test.shape[0])
                predicciones[0]=Y_train[-1]
                predicciones[1:]=Y_test[:-1]

                error_test=evaluate_model(X_test,test_data,predicciones=predicciones,model=None)
                error_val=evaluate_model(X_val_hip,train_data[X_train_hip.shape[0]+1:],predicciones=predicciones_val,model=None)
                dicc[model]=np.array([error_val,error_test])

                predicciones=predicciones.reshape(predicciones.shape[0])
                precios_predicciones=np.array(test_data.iloc[0:]['Open'])*(1+predicciones)
        if model=='Linear':
                modelo=build_model_linear()
                modelo.fit(x=X_train_hip,y=Y_train_hip,epochs=hip_epochs)
                predicciones_val=modelo.predict(X_val_hip)
                predicciones=modelo.predict(X_test)


                error_val=evaluate_model(X_val_hip,train_data[X_train_hip.shape[0]+1:],model=modelo,predicciones=None)


                modelo.fit(x=X_train,y=Y_train,epochs=epochs)
                error_test=evaluate_model(X_test,test_data,model=modelo,predicciones=None)
                dicc[model]=np.array([error_val,error_test])
                predicciones=modelo.predict(X_test)
                predicciones=predicciones.reshape(predicciones.shape[0])
                precios_predicciones=np.array(test_data.iloc[0:]['Open'])*(1+predicciones)



        # Vamos a plotear las prediccions junto con el modelo.
        plot_model(test_data,precios_predicciones)

        # Se analiza en rendimiento en cuanto al algoritmo de inversión
        balance_1=compra_largo(precios_predicciones,test_data.iloc[:]['Open'].values,test_data.iloc[:]['Close'].values)
        balance_2=venta_corto(precios_predicciones,test_data.iloc[:]['Open'].values,test_data.iloc[:]['Close'].values)
        resultado=balance_1+balance_2+1
        return precios_predicciones, resultado, dicc



#### Se almacenan los modelos que se quieren estudiar y diccionarios para almacenar los resultados.

modelos=['Baseline','Linear','RNN','LSTM']
dicc={ticker:np.empty(2) for ticker in modelos} # Ya que hay validación y test
precios_predicciones={ticker:np.empty(Y_test.shape[0]) for ticker in modelos}
resultado={ticker:0 for ticker in modelos}
epochs=100
hip_epochs=20

#Baseline
precios_predicciones['Baseline'], resultado['Baseline'], dicc=compile_and_fit('Baseline',X_train,Y_train,X_train_hip,Y_train_hip,X_val_hip,Y_val_hip,X_test,train_data,test_data,dicc,epochs,hip_epochs)

#Linear
precios_predicciones['Linear'], resultado['Linear'], dicc=compile_and_fit('Linear',X_train,Y_train,X_train_hip,Y_train_hip,X_val_hip,Y_val_hip,X_test,train_data,test_data,dicc,epochs,hip_epochs)

#RNN
precios_predicciones['RNN'], resultado['RNN'], dicc=compile_and_fit('RNN',X_train,Y_train,X_train_hip,Y_train_hip,X_val_hip,Y_val_hip,X_test,train_data,test_data,dicc,epochs,hip_epochs)

#LSTM
precios_predicciones['LSTM'], resultado['LSTM'], dicc=compile_and_fit('LSTM',X_train,Y_train,X_train_hip,Y_train_hip,X_val_hip,Y_val_hip,X_test,train_data,test_data,dicc,epochs,hip_epochs)


coc_baseline=dicc['Baseline'][1]/dicc['Baseline'][0]
coc_linear=dicc['Linear'][1]/dicc['Linear'][0]
coc_rnn=dicc['RNN'][1]/dicc['RNN'][0]
coc_rnn=dicc['LSTM'][1]/dicc['LSTM'][0]

#### Se plotean los resultados de la evaluación de los modelos.

lista1=[]
lista2=[]
for modelo in modelos:
    lista1.append(dicc[modelo][0])
    lista2.append(dicc[modelo][1])

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

x = np.arange(len(dicc))
width = 0.3

axs.bar(x - 0.17, lista1, width, label='Validation')
axs.bar(x + 0.17, lista2, width, label='Test')
axs.set_xticks(ticks=x, labels=modelos,rotation=45)
        #axs[0,i].set_ylabel(f'MAE (average over all times and outputs)')
axs.legend()
axs.set_title("Errores")

resultado

#FIN