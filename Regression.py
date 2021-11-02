print('INFO: Running Regression.py')

# Import modules
import os,sys,argparse

##########################################################################################################
# DO NOT MODIFY (below this line)
##########################################################################################################

##########################################################################################################
# Read arguments
##########################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--inputDataType',         action='store',      dest="inputDataType",                      help='Input data type')
parser.add_argument('--particleType',          action='store',      dest="particleType",                       help='Particle type')
parser.add_argument('--etaRange',              action='store',      dest="etaRange",                           help='|eta| range')
parser.add_argument('--outPATH',               action='store',      dest="outPATH",                            help='Output path')
parser.add_argument('--activation',            action='store',      dest="activationType"    , default='relu', help='Layer activation type')
parser.add_argument('--nEpochs',               action='store',      dest="nEpochs"           , default=200,    help='Number of epochs')
parser.add_argument('--learningRate',          action='store',      dest="learningRate"      , default=0.001,  help='Learning rate')
parser.add_argument('--loss',                  action='store',      dest="loss"              , default='MSE',  help='Type of loss')
parser.add_argument('--nNodes',                action='store',      dest="nNodes"            , default=100,    help='Number of nodes per hidden layer')
parser.add_argument('--nLayers',               action='store',      dest="nLayers"           , default=2,      help='Number of hidden layers')
parser.add_argument('--useBatchNormalization', action='store_true', dest="useBatchNorm"      , default=False,  help='Use BatchNormalization')
parser.add_argument('--useNormalizationLayer', action='store_true', dest="useNormLayer"      , default=False,  help='Use preprocessing.Normalization layer')
parser.add_argument('--useEarlyStopping',      action='store_true', dest="useEarlyStopping"  , default=False,  help='Use EarlyStopping')
parser.add_argument('--useModelCheckpoint',    action='store_true', dest="useModelCheckpoint", default=False,  help='Use ModelCheckpoint')
parser.add_argument('--debug',                 action='store_true', dest="debug"             , default=False,  help='Run in debug mode')
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

class Config:
  def __init__(self,datatype,activation,epochs,lr,earlyStop,useNormalizer,useMCP,useBatchNorm,loss,Nnodes,Nlayers,particle,eta,outPATH):
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

# Print chosen setup
config = Config(InputDataType,ActivationType,Nepochs,LearningRate,UseEarlyStopping,UseNormalizer,UseModelCheckPoint,UseBatchNormalization,Loss,NnodesHiddenLayers,NhiddenLayers,ParticleType,EtaRange,OutputPATH)
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
  if 'pions' in config.Particle or config.Particle == 'all': Layers += [13,14]
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
  InputFiles = []
  if config.Particle == 'photons':
    PATH = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/photons/v7/' # normalized inputs
  elif config.Particle == 'pions':
    PATH = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/pions/v3/' # normalized inputs (using non-phiCorrected files)
  elif config.Particle == 'electrons':
    PATH = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/electrons/phiCorrected/' # normalized inputs
  elif config.Particle == 'all':
    #PATH = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/pions_and_electrons_and_photons/v1/' # w/o pdgId
    PATH = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/pions_and_electrons_and_photons/v4/' # w/  pdgId
  elif config.Particle == 'electronsANDphotons':
    PATH = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/electrons_and_photons/v1/'
  elif config.Particle == 'pionsANDelectrons':
    PATH = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/pions_and_electrons/v1/'
  else:
    print('ERROR: {} not supported yet, exiting'.format(config.Particle))
    sys.exit(1)
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
  print('INFO: Reading {}'.format(InputFile))
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

## Plot correlations (takes too much time)
#import seaborn as sns
#print('INFO: Plot variable distributions')
#plt.figure('correlation')
#plot = sns.pairplot(train_dataset, diag_kind='kde')
#plot.savefig('{}/{}_{}_{}_Correlations.pdf'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange))

# Import tensorflow and numpy
import tensorflow as tf
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
#for layer in model.layers:
#  print('layer name: {}'.format(layer.name))
#  print(layer.get_weights())

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
  MC = tf.keras.callbacks.ModelCheckpoint('{}/{}_{}_{}_best_model.h5'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
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
    verbose=1,
    validation_split = 0.4,
    callbacks=Callbacks)
else:
  history = model.fit(
    train_features, train_labels,
    epochs=config.Nepochs,
    verbose=1,
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
  if config.UseNormalizer:
    model = tf.keras.models.load_model('{}/{}_{}_{}_best_model.h5'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange),custom_objects={'Normalization': preprocessing.Normalization})
  else:
    model = tf.keras.models.load_model('{}/{}_{}_{}_best_model.h5'.format(config.outPATH,OutBaseName,config.Particle,config.EtaRange))

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
print('INFO: Evaluate')
test_results = model.evaluate(test_features, test_labels, verbose=0)
print(model.metrics_names)
print(test_results)

# Print epoch at which the training stopped
if config.UseEarlyStopping:
  print('Trainning stopped at epoch = {}'.format(ES.stopped_epoch))

print('>>> ALL DONE <<<')
