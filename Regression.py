print('INFO: Running Regression.py')

# Import modules
import os,sys,argparse

##########################################################################################################
# DO NOT MODIFY (below this line)
##########################################################################################################

AllowedActivationTypes = ['relu','tanh','linear','LeakyRelu']
AllowedParticleTypes   = ['photons','electrons','pions','all','electronsANDphotons','pionsANDelectrons']

##########################################################################################################
# Read arguments
##########################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--inputDataType',         action='store',      dest="inputDataType"     , default='Real', help='Input data type (default="Real")')
parser.add_argument('--particleType',          action='store',      dest="particleType",                       help='Particle type (options: {})'.format([x for x in AllowedParticleTypes]))
parser.add_argument('--etaRange',              action='store',      dest="etaRange",                           help='|eta| range (example: "0_5")')
parser.add_argument('--outPATH',               action='store',      dest="outPATH",                            help='Output path')
parser.add_argument('--activation',            action='store',      dest="activationType"    , default='relu', help='Layer activation type (default="relu", options: {})'.format([x for x in AllowedActivationTypes]))
parser.add_argument('--nEpochs',               action='store',      dest="nEpochs"           , default=200,    help='Number of epochs (default=200)')
parser.add_argument('--learningRate',          action='store',      dest="learningRate"      , default=0.001,  help='Learning rate (default=0.001)')
parser.add_argument('--loss',                  action='store',      dest="loss"              , default='MSE',  help='Type of loss (default="MSE", options: "weighted_mean_squared_error", "MSE" [mean_squared_error] or "MAE" [mean_absolute_error])')
parser.add_argument('--nNodes',                action='store',      dest="nNodes"            , default=100,    help='Number of nodes per hidden layer (default=100)')
parser.add_argument('--nLayers',               action='store',      dest="nLayers"           , default=2,      help='Number of hidden layers (default=2)')
parser.add_argument('--verbose',               action='store',      dest="verbose"           , default=1,      help='Set fit verbosity (default=1)')
parser.add_argument('--useBatchNormalization', action='store_true', dest="useBatchNorm"      , default=False,  help='Use BatchNormalization (default=False)')
parser.add_argument('--useNormalizationLayer', action='store_true', dest="useNormLayer"      , default=False,  help='Use preprocessing.Normalization layer (default=False)')
parser.add_argument('--useEarlyStopping',      action='store_true', dest="useEarlyStopping"  , default=False,  help='Use EarlyStopping (default=False)')
parser.add_argument('--useModelCheckpoint',    action='store_true', dest="useModelCheckpoint", default=False,  help='Use ModelCheckpoint (default=False)')
parser.add_argument('--debug',                 action='store_true', dest="debug"             , default=False,  help='Run in debug mode (default=False)')
args = parser.parse_args()

InputDataType         = args.inputDataType
ParticleType          = args.particleType
EtaRange              = args.etaRange
ActivationType        = args.activationType
OutputPATH            = args.outPATH
Nepochs               = int(args.nEpochs)
LearningRate          = float(args.learningRate)
Loss                  = args.loss
UseBatchNormalization = args.useBatchNorm
UseNormalizer         = args.useNormLayer
UseEarlyStopping      = args.useEarlyStopping
UseModelCheckPoint    = args.useModelCheckpoint
NnodesHiddenLayers    = int(args.nNodes)
NhiddenLayers         = int(args.nLayers)
Debug                 = args.debug
Verbose               = int(args.verbose)

class Config:
  def __init__(self,datatype,activation,epochs,lr,earlyStop,useNormalizer,useMCP,useBatchNorm,loss,Nnodes,Nlayers,particle,eta,outPATH,verbose):
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
    self.NhiddenLayers         = Nlayers
    self.Particle              = particle
    self.EtaRange              = eta
    self.outPATH               = outPATH
    self.verbose               = verbose

# Print chosen setup
config = Config(InputDataType,ActivationType,Nepochs,LearningRate,UseEarlyStopping,UseNormalizer,UseModelCheckPoint,UseBatchNormalization,Loss,NnodesHiddenLayers,NhiddenLayers,ParticleType,EtaRange,OutputPATH,Verbose)
print('###################################################################################')
print('INFO: Setup:')
print(vars(config))
print('###################################################################################')

OutBaseName = '{}_{}'.format(config.InputDataType,config.ActivationType)

# Protections
if config.ActivationType not in AllowedActivationTypes:
  print('ERROR: ActivationType ({}) not recognized, exiting'.format(config.ActivationType))
  sys.exit(1)
if config.Particle not in AllowedParticleTypes:
  print('ERROR: Particle ({}) not recognized, exiting')
  sys.exit(1)

from InputFiles import PATH2InputFiles as PATHs

def get_layers(particle, eta_bin):
  layers_dict = {
    'photons' : {'3_1_0_12_2': 'eta_0_130', '3_1_0_17_2': 'eta_130_135', '4_5_18_3_1_0_6_17_2': 'eta_135_145', '4_5_18_1_0_6_17_2': 'eta_145_150', '4_5_6_17': 'eta_150_160', '4_5_6_7_8': 'eta_160_300', '21_23_6_7_22_8': 'eta_300_500'},
    'electrons' : {'3_1_0_12_2': 'eta_0_130', '3_1_0_17_2': 'eta_130_135', '4_5_18_3_1_0_6_17_2': 'eta_135_145', '4_5_18_1_0_6_17_2': 'eta_145_150', '4_5_6_17': 'eta_150_160', '4_5_6_7_8': 'eta_160_300', '21_23_6_7_22_8': 'eta_300_500'},
    'electronsANDphotons' : {'3_1_0_12_2': 'eta_0_130', '3_1_0_17_2': 'eta_130_135', '4_5_18_3_1_0_6_17_2': 'eta_135_145', '4_5_18_1_0_6_17_2': 'eta_145_150', '4_5_6_17': 'eta_150_160', '4_5_6_7_8': 'eta_160_300', '21_23_6_7_22_8': 'eta_300_500'},
    'pions' : {'3_1_0_12_14_2_13': 'eta_0_90', '20_18_19_3_1_17_0_12_14_2_13_16_15': 'eta_90_130', '20_18_3_6_7_2_15_4_19_1_14_12_13_16_5_0_17_9_8': 'eta_130_150', '4_9_5_11_18_19_6_10_7_17_8': 'eta_150_170', '4_9_5_11_18_6_10_7_8': 'eta_170_240', '4_9_5_11_6_10_7_8': 'eta_240_280', '4_9_5_11_21_23_6_10_7_22_8': 'eta_280_350', '22_23_21': 'eta_350_500'},
    'pionsANDelectrons' : {'3_1_0_12_14_2_13': 'eta_0_90', '20_18_19_3_1_17_0_12_14_2_13_16_15': 'eta_90_130', '20_18_3_6_7_2_15_4_19_1_14_12_13_16_5_0_17_9_8': 'eta_130_150', '4_9_5_11_18_19_6_10_7_17_8': 'eta_150_170', '4_9_5_11_18_6_10_7_8': 'eta_170_240', '4_9_5_11_6_10_7_8': 'eta_240_280', '4_9_5_11_21_23_6_10_7_22_8': 'eta_280_350', '22_23_21_6_7_8': 'eta_350_500'},
    'all' : {'3_1_0_12_14_2_13': 'eta_0_90', '20_18_19_3_1_17_0_12_14_2_13_16_15': 'eta_90_130', '20_18_3_6_7_2_15_4_19_1_14_12_13_16_5_0_17_9_8': 'eta_130_150', '4_9_5_11_18_19_6_10_7_17_8': 'eta_150_170', '4_9_5_11_18_6_10_7_8': 'eta_170_240', '4_9_5_11_6_10_7_8': 'eta_240_280', '4_9_5_11_21_23_6_10_7_22_8': 'eta_280_350', '22_23_21_6_7_8': 'eta_350_500'},
  }[particle]
  eta_min = int(eta_bin.split('_')[1])
  eta_max = int(eta_bin.split('_')[2])
  for key, eta_range in layers_dict.items():
    min_range = int(eta_range.split('_')[1])
    max_range = int(eta_range.split('_')[2])
    if eta_min >= min_range and eta_max <= max_range:
      layers = list(set([int(item) for item in key.split('_')]))
      return list(set([int(item) for item in key.split('_')]))
  return []

# Get data
print('INFO: Get data')
if Debug: print('DEBUG: Get list of input CSV files')
if config.InputDataType == 'Example':
  header     = ['x','x+2','y']
  features   = ['x','x+2']
  labels     = ['y']
  InputFiles = ['TestData.csv']
elif config.InputDataType == 'Real':
  Layers = get_layers(config.Particle, 'eta_{}'.format(config.EtaRange))
  print('INFO: Calorimeter layers: {}'.format(Layers))
  header     = ['e_{}'.format(x) for x in Layers]
  header    += ['ef_{}'.format(x) for x in Layers]
  features   = ['ef_{}'.format(x) for x in Layers]
  features  += ['etrue']
  if config.Particle=='all' or config.Particle=='electronsANDphotons' or config.Particle=='pionsANDelectrons':
    features  += ['pdgId']
  labels     = ['extrapWeight_{}'.format(x) for x in Layers]
  header    += labels
  header    += ['etrue']
  if config.Particle=='all' or config.Particle=='electronsANDphotons' or config.Particle=='pionsANDelectrons':
    header    += ['pdgId']
  # Get path to input files
  try:
    PATH = PATHs[config.Particle]
  except KeyError:
    print(f'{config.Particle} is not available in PATH2InputFiles from InputFiles.py, exiting')
    sys.exit(1)
  InputFiles = []
  for File in os.listdir(PATH):
    if '.csv' not in File or 'eta_{}'.format(config.EtaRange) not in File: continue # Select only files for the requested eta bin
    InputFiles.append(PATH+File)
else:
  print('ERROR: InputDataType not recognized, exiting')
  sys.exit(1)
# Protection
if len(InputFiles) == 0:
  print('ERROR: No input files found, exiting')
  sys.exit(1)
# Import dataset using pandas
import pandas as pd
if Debug: print('DEBUG: Read each CSV file and create a single DF')
DFs = []
for InputFile in InputFiles:
  print(f'INFO: Reading {InputFile}')
  raw_dataset = pd.read_csv(InputFile, names=header, na_values='?', comment='\t', sep=',', skiprows=[0] , skipinitialspace=True)
  DFs.append(raw_dataset.copy())
if len(DFs) == 0:
  print('ERROR: No DF is available, exiting')
  sys.exit(1)
dataset = DFs[0]
for idf in range(1,len(DFs)):
  dataset = pd.concat([dataset,DFs[idf]],ignore_index=True)
print('INFO: Last 5 rows of input data')
print(dataset.tail(5))

# Split data into train and test
print('INFO: Split data into train and test')
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset  = dataset.drop(train_dataset.index)

## Plot correlations (takes too much time, left in case of need)
#import seaborn as sns
#print('INFO: Plot variable distributions')
#plt.figure('correlation')
#plot = sns.pairplot(train_dataset, diag_kind='kde')
#plot.savefig('{}/{}_{}_{}_Correlations.pdf'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange))

# Import tensorflow and numpy
import tensorflow as tf
import keras.backend as K
import numpy as np
# Set seeds to get reproducible results
np.random.seed(1)
tf.random.set_seed(1)

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
  normalizer = tf.keras.layers.experimental.preprocessing.Normalization(input_shape=[Nfeatures,])
  normalizer.adapt(np.array(train_features))

# Choose model
print('INFO: Create model')
model = tf.keras.Sequential()
if config.UseNormalizer: model.add(normalizer)
if config.UseBatchNormalization:
  model.add(tf.keras.layers.BatchNormalization(input_shape=[Nfeatures,]))
if config.ActivationType != 'LeakyRelu' and config.ActivationType != 'linear':
  model.add(tf.keras.layers.Dense(config.NnodesHiddenLayers, input_dim=Nfeatures, activation=config.ActivationType))
  for ilayer in range(1,config.NhiddenLayers):
    model.add(tf.keras.layers.Dense(config.NnodesHiddenLayers, activation=config.ActivationType))
elif config.ActivationType == 'linear':
  model.add(tf.keras.layers.Dense(config.NnodesHiddenLayers, input_dim=Nfeatures))
  for ilayer in range(1,config.NhiddenLayers):
    model.add(tf.keras.layers.Dense(config.NnodesHiddenLayers))
else: # LeakyRelu
  model.add(tf.keras.layers.Dense(config.NnodesHiddenLayers, input_dim=Nfeatures))
  model.add(tf.keras.layers.LeakyReLU())
  for ilayer in range(1,config.NhiddenLayers):
    model.add(tf.keras.layers.Dense(config.NnodesHiddenLayers))
    model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Dense(Nlabels))
model.summary()
tf.keras.utils.plot_model(model, to_file='{}/model_{}_{}.png'.format(config.outPATH,config.Particle,config.EtaRange), show_shapes=True)
#for layer in model.layers: # left in case of need
#  print(f'layer name: {layer.name}')
#  print(layer.get_weights())

# Calculate a weight for each label
etotal = train_dataset[[x for x in header if 'e_' in x]].copy()
etotal['etotal'] = etotal.sum(axis=1)
weights = [train_dataset[x.replace('extrapWeight_', 'e_')].mean()/etotal['etotal'].mean() for x in labels]
print(f'weights = {weights}')

# Define custom loss
def weighted_mean_squared_error(y_true, y_pred):
  return K.mean(K.square(y_true-y_pred)*weights)

# Compile model
print('INFO: Compile model')
Loss = 'mean_squared_error'
if config.Loss == 'MAE':
  Loss = 'mean_absolute_error'
elif config.Loss == 'MSE':
  Loss = 'mean_squared_error'
else:
  Loss = weighted_mean_squared_error
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
  MC = tf.keras.callbacks.ModelCheckpoint('{}/{}_{}_{}_best_model.h5'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange), monitor='val_loss', mode='min', verbose=config.verbose, save_best_only=True)
  Callbacks.append(MC)

# Terminate on NaN such that it is easier to debug
Callbacks.append(tf.keras.callbacks.TerminateOnNaN())

# Fit
print('INFO: Fit model')
if len(Callbacks) > 0:
  history = model.fit(
    train_features, train_labels,
    epochs=config.Nepochs,
    #batch_size=132,        # left in case of need
    #steps_per_epoch=1,     # left in case of need
    #sample_weight=weights, # Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only)
    #class_weight=class_weights, # left in case of need
    verbose=config.verbose,
    validation_split = 0.4,
    callbacks=Callbacks)
else:
  history = model.fit(
    train_features, train_labels,
    epochs=config.Nepochs,
    verbose=config.verbose,
    validation_split = 0.4)

# Plot loss vs epochs
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
import HelperFunctions
HelperFunctions.plot_loss(history,config.outPATH,OutBaseName+'_'+config.Particle+'_'+config.EtaRange,config.Loss)

# Save the model
if not config.UseModelCheckPoint:
  print('INFO: Save model')
  model_filename = '{}/{}_{}_{}_model.hf'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange)
  model.save(model_filename)

# Load the best model
if config.UseModelCheckPoint:
  custom_objects_dict = {}
  if config.Loss == 'weighted_mean_squared_error':
    custom_objects_dict['weighted_mean_squared_error'] = weighted_mean_squared_error
  if config.UseNormalizer:
    custom_objects_dict['Normalization'] = preprocessing.Normalization
  if not custom_objects_dict: # dict is empty
    model = tf.keras.models.load_model('{}/{}_{}_{}_best_model.h5'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange))
  else:
    model = tf.keras.models.load_model('{}/{}_{}_{}_best_model.h5'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange), custom_objects=custom_objects_dict)

# Compare true vs prediction
Features_dataset = dataset[features].copy().values.reshape(-1,Nfeatures)
Labels_dataset   = dataset[labels].copy().values.reshape(-1,Nlabels)
pred             = model.predict(Features_dataset)
counter = 0
import matplotlib.pyplot as plt
for Label in labels:
  plt.figure('true_vs_prediction_{}'.format(Label))
  plt.hist(Labels_dataset[:,counter],label=Label+' true',bins=50)
  plt.hist(pred[:,counter],label=Label+' prediction',bins=50,alpha=0.5)
  plt.legend()
  plt.savefig('{}/{}_{}_{}_true_vs_prediction_{}.pdf'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange,Label))
  counter += 1

# Evaluate
print('INFO: Evaluate on training data')
train_results = model.evaluate(train_features, train_labels, verbose=config.verbose)
print(model.metrics_names)
print(train_results)
print('INFO: Evaluate on test data')
test_results = model.evaluate(test_features, test_labels, verbose=config.verbose)
print(model.metrics_names)
print(test_results)

# Print epoch at which the training stopped
if config.UseEarlyStopping and ES.stopped_epoch:
  print(f'Training stopped at epoch = {ES.stopped_epoch}')

print('>>> ALL DONE <<<')
