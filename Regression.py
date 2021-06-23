# Import modules
import ROOT
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# Set seeds to get reproducible results
np.random.seed(1)
tf.random.set_seed(1)
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

InputDataType         = 'Real' # Options: Example, Real
ActivationType        = 'relu' # Options: relu, tanh, linear, LeakyRelu
Nepochs               = 200
LearningRate          = 0.001
Loss                  = 'MSE'  # Options: mean_absolute_error (MAE) and mean_squared_error (MSE)
UseBatchNormalization = False
UseNormalizer         = True
UseEarlyStopping      = True
UseModelCheckPoint    = True
NnodesHiddenLayers    = 100
Debug                 = False

##################################################################
# DO NOT MODIFY (below this line)
##################################################################

class Config:
  def __init__(self,datatype,activation,epochs,lr,earlyStop,useNormalizer,useMCP,useBatchNorm,loss,Nnodes):
    self.InputDataType         = datatype
    self.ActivationType        = activation
    self.Nepochs               = epochs
    self.LearningRate          = lr
    self.UseEarlyStopping      = earlyStop
    self.UseNormalizer         = useNormalizer
    self.UseModelCheckPoint    = useMCP
    self.UseBatchNormalization = useBatchNorm
    self.Loss                  = loss
    self.NnodesHiddenLayers    = Nnodes

# Print chosen setup
config = Config(InputDataType,ActivationType,Nepochs,LearningRate,UseEarlyStopping,UseNormalizer,UseModelCheckPoint,UseBatchNormalization,Loss,NnodesHiddenLayers)
print('###################################################################################')
print('INFO: Setup:')
print(vars(config))
print('###################################################################################')

OutBaseName = '{}_{}'.format(config.InputDataType,config.ActivationType)

# Protections
AllowedActivationTypes = ['relu','tanh','linear','LeakyRelu']
if config.ActivationType not in AllowedActivationTypes:
  print('ERROR: ActivationType not recognized, exiting')
  sys.exit(1)

# Get data
print('INFO: Get data')
if Debug: print('DEBUG: Get list of input CSV files')
if config.InputDataType == 'Example':
  header     = ['x','x+2','y']
  features   = ['x','x+2']
  labels     = ['y']
  InputFiles = ['TestData.csv']
elif config.InputDataType == 'Real':
  Layers     = [0,1,2,3,12]
  header     = ['e_{}'.format(x) for x in Layers]
  features   = header.copy()
  features  += ['etrue']
  labels     = ['extrapWeight_{}'.format(x) for x in Layers]
  header    += labels
  header    += ['etrue']
  InputFiles = []
  PATH       = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/v2/'
  for File in os.listdir(PATH):
    InputFiles.append(PATH+File)
else:
  print('ERROR: InputDataType not recognized, exiting')
  sys.exit(1)
# Import dataset using pandas
if Debug: print('DEBUG: Read each CSV file and create a single DF')
DFs = []
for InputFile in InputFiles:
  raw_dataset = pd.read_csv(InputFile, names=header, na_values='?', comment='\t', sep=',', skiprows=[0] , skipinitialspace=True)
  DFs.append(raw_dataset.copy())
dataset = DFs[0]
for idf in range(1,len(DFs)):
  dataset = pd.concat([dataset,DFs[idf]],ignore_index=True)
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

# Get number of features and labels
Nfeatures = len(features)
Nlabels   = len(labels)

# Normalizer
if config.UseNormalizer:
  normalizer = preprocessing.Normalization(input_shape=[Nfeatures,])
  normalizer.adapt(np.array(train_features))

# Choose model
print('INFO: Create model')
model = tf.keras.Sequential()
if config.UseNormalizer: model.add(normalizer)
if config.ActivationType != 'LeakyRelu' and config.ActivationType != 'linear':
  model.add(layers.Dense(config.NnodesHiddenLayers, input_dim=Nfeatures, activation=config.ActivationType))
  if config.UseBatchNormalization:
    model.add(layers.BatchNormalization(input_shape=[Nfeatures,]))
  model.add(layers.Dense(config.NnodesHiddenLayers, activation=config.ActivationType))
elif config.ActivationType == 'linear':
  model.add(layers.Dense(config.NnodesHiddenLayers, input_dim=Nfeatures))
  if config.UseBatchNormalization:
    model.add(layers.BatchNormalization(input_shape=[Nfeatures,]))
  model.add(layers.Dense(config.NnodesHiddenLayers))
else: # LeakyRelu
  model.add(layers.Dense(config.NnodesHiddenLayers, input_dim=Nfeatures))
  model.add(layers.LeakyReLU())
  if config.UseBatchNormalization:
    model.add(layers.BatchNormalization(input_shape=[Nfeatures,]))
  model.add(layers.Dense(config.NnodesHiddenLayers))
  model.add(layers.LeakyReLU())
model.add(layers.Dense(Nlabels))
model.summary()

# Compile model
print('INFO: Compile model')
Loss = 'mean_absolute_error' if config.Loss == 'MAE' else 'mean_squared_error'
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=config.LearningRate),
    loss=Loss,
    metrics=['MeanAbsoluteError','MeanSquaredError']) # Computes the mean absolute error and the mean squared error b/w y_true and y_pred

Callbacks = []

# Early stopping
if config.UseEarlyStopping:
  ES = tf.keras.callbacks.EarlyStopping(patience=20, mode='min', restore_best_weights=True)
  Callbacks.append(ES)

# ModelCheckPoint
if config.UseModelCheckPoint:
  MC = tf.keras.callbacks.ModelCheckpoint('{}_best_model.h5'.format(OutBaseName), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
  Callbacks.append(MC)

# Fit
print('INFO: Fit model')
if len(Callbacks) > 0:
  history = model.fit(
    train_features, train_labels,
    epochs=config.Nepochs,
    verbose=1,
    validation_split = 0.4,
    callbacks=Callbacks)
else:
  history = model.fit(
    train_features, train_labels,
    epochs=config.Nepochs,
    verbose=1,
    validation_split = 0.4)

# Save the model
if not config.UseModelCheckPoint:
  print('INFO: Save model')
  model_filename = '{}_model.hf'.format(OutBaseName)
  model.save(model_filename)

# Load the best model
if config.UseModelCheckPoint:
  if config.UseNormalizer:
    model = keras.models.load_model('{}_best_model.h5'.format(OutBaseName),custom_objects={'Normalization': preprocessing.Normalization})
  else:
    model = keras.models.load_model('{}_best_model.h5'.format(OutBaseName))

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
plot_loss(history,OutBaseName,config.Loss)

# Evaluate
print('INFO: Evaluate')
test_results = model.evaluate(test_features, test_labels, verbose=0)
print(model.metrics_names)
print(test_results)

# Print epoch at which the training stopped
if config.UseEarlyStopping:
  print('Trainning stopped at epoch = {}'.format(ES.stopped_epoch))

print('>>> ALL DONE <<<')
