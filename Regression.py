# Import modules
import ROOT
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os,sys
from HelperFunctions import *

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Import keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing

##################################################################
# Choose data
##################################################################

InputDataType  = 'Real' # Options: Example, Real
ActivationType = 'tanh' # Options: relu, tanh, linear, LeakyRelu
Nepochs        = 200

##################################################################
# DO NOT MODIFY (below this line)
##################################################################

OutBaseName = '{}_{}'.format(InputDataType,ActivationType)

# Get data
print('INFO: Get data')
if InputDataType == 'Example':
  header    = ['x','x+2','y']
  features  = ['x','x+2']
  labels    = ['y']
  InputFile = 'TestData.csv'
elif InputDataType == 'Real':
  Layers    = [0,1,2,3,12]
  header    = ['e_{}'.format(x) for x in Layers]
  features  = header
  #labels    = ['extrapWeight_{}'.format(x) for x in Layers] # Temporary
  labels    = ['extrapWeight_0']
  header   += labels
  InputFile = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/pid22_E65536_eta_20_25_phiCorrected.csv'
else:
  print('ERROR: InputDataType not recognized, exiting')
  sys.exit(1)
# Import dataset using pandas
raw_dataset = pd.read_csv(InputFile, names=header, na_values='?', comment='\t', sep=',', skiprows=[0] , skipinitialspace=True)
dataset     = raw_dataset.copy()
print('INFO: Last 5 rows of input data')
print(dataset.tail(5))

# Split data into train and test
print('INFO: Split data into train and test')
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset  = dataset.drop(train_dataset.index)

# Plot correlations
print('INFO: Plot variable distributions')
plt.figure('correlation')
plot = sns.pairplot(train_dataset, diag_kind='kde')
plot.savefig('{}_Correlations.pdf'.format(OutBaseName))

# Split features from labels
# Separate the target value (label") from the features. This label is the value that you will train the model to predict
print('INFO: Split features from labels')
train_features = train_dataset[features].copy()
test_features  = test_dataset[features].copy()
train_labels   = train_dataset[labels].copy()
test_labels    = test_dataset[labels].copy()

# Normalizer
Nfeatures = len(features)
Nlabels   = len(labels)
normalizer = preprocessing.Normalization(input_shape=[Nfeatures,])
normalizer.adapt(np.array(train_features))

# Choose model
print('INFO: Create model')
model = tf.keras.Sequential()
model.add(normalizer)
if ActivationType != 'LeakyRelu':
  model.add(layers.Dense(100, input_dim=Nfeatures, activation=ActivationType))
  model.add(layers.Dense(100, activation=ActivationType))
elif ActivationType != 'linear':
  model.add(layers.Dense(100, input_dim=Nfeatures))
  model.add(layers.Dense(100))
else: # LeakyRelu
  model.add(layers.Dense(100, input_dim=Nfeatures))
  model.add(layers.LeakyReLU())
  model.add(layers.Dense(100))
  model.add(layers.LeakyReLU())
model.add(layers.Dense(Nlabels))
model.summary()

# Compile model
print('INFO: Compile model')
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error',
    metrics=['MeanSquaredError']) # Computes the mean squared error between y_true and y_pred

# Save the model
print('INFO: Save model')
model_filename = '{}_model.hf'.format(OutBaseName)
model.save(model_filename)

# Fit
print('INFO: Fit model')
history = model.fit(
    train_features, train_labels,
    epochs=Nepochs,
    verbose=1,
    validation_split = 0.3)

# Compare true vs prediction
Features_dataset = dataset[features].copy().values.reshape(-1,Nfeatures)
Labels_dataset   = dataset[labels].copy().values.reshape(-1,Nlabels)
pred             = model.predict(Features_dataset)
counter = 0
for Label in labels:
  plt.figure('true_vs_prediction_{}'.format(Label))
  plt.hist(Labels_dataset[:,counter],label=Label+' true',bins=50)
  plt.hist(pred[:,counter],label=Label+' prediction',bins=50,alpha=0.5)
  plt.legend()
  plt.savefig('{}_true_vs_prediction_{}.pdf'.format(OutBaseName,Label))
  counter += 1

# Plot loss vs epochs
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plot_loss(history,OutBaseName)

# Evaluate
print('INFO: Evaluate')
test_results = model.evaluate(test_features, test_labels, verbose=0)
print(model.metrics_names)
print(test_results)

print('>>> ALL DONE <<<')

