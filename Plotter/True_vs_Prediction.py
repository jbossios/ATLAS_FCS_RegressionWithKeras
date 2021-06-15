
Activation = 'relu'
Version    = 'v10'

################################################################
# DO NOT MODIFY (below this line)
################################################################

###############################
# Create output folders
###############################
import os,sys
os.system('mkdir Plots')
os.system('mkdir Plots/{}'.format(Version))
OutBaseName = 'Real_{}'.format(Activation)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

###############################
# Get model
###############################
#model = keras.models.load_model('../Outputs/{}/Real_{}_model.hf/'.format(Version,Activation))
model = keras.models.load_model('../Outputs/{}/Real_{}_best_model.h5'.format(Version,Activation))

###############################
# Get data
###############################
print('INFO: Get data')
Layers     = [0,1,2,3,12]
header     = ['e_{}'.format(x) for x in Layers]
#features   = header+['etrue'] # Temporary
features   = header.copy()
labels     = ['extrapWeight_{}'.format(x) for x in Layers]
header    += labels
header    += ['etrue']
InputFiles = []
PATH       = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/v2/'
for File in os.listdir(PATH):
  # Temporary
  if 'E65536' not in File: continue
  InputFiles.append(PATH+File)
# Import dataset using pandas
DFs = []
for InputFile in InputFiles:
  raw_dataset = pd.read_csv(InputFile, names=header, na_values='?', comment='\t', sep=',', skiprows=[0] , skipinitialspace=True)
  DFs.append(raw_dataset.copy())
dataset = DFs[0]
for idf in range(1,len(DFs)):
  dataset = pd.concat([dataset,DFs[idf]],ignore_index=True)
print('INFO: Last 5 rows of input data')
print(dataset.tail(5))

###############################
# Compare true vs prediction
###############################
# Prepare numpy arrays
Nfeatures = len(features)
Nlabels   = len(labels)
Features_dataset = dataset[features].copy().values.reshape(-1,Nfeatures)
Labels_dataset   = dataset[labels].copy().values.reshape(-1,Nlabels)
pred             = model.predict(Features_dataset)
counter = 0
# Make figure
for Label in labels:
  plt.figure('true_vs_prediction_{}'.format(Label))
  # Creat canvas with two panels 
  fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
  # Plot prediction and true distributions
  valsTrue, binsTrue, patchesTrue = axes[0].hist(Labels_dataset[:,counter],label=Label+' true',bins=50)
  valsPred, binsPred, patchesPred = axes[0].hist(pred[:,counter],label=Label+' prediction',bins=50,alpha=0.5)
  axes[0].legend()
  # Calculate and plot prediction/true ratio
  ratio = np.divide(valsPred,valsTrue)
  ratio[ratio == np.inf] = 0
  ratio[np.isnan(ratio)] = 0
  centers = [] # bin centers
  for ibin in range(0,len(binsTrue)-1):
    center = 0.5*(binsTrue[ibin+1]-binsTrue[ibin])+binsTrue[ibin]
    centers.append(center)
  axes[1].plot(centers,ratio)
  # Plot line at one
  axes[1].plot(centers,[1 for i in range(len(centers))],'--',color='k')
  axes[1].set_ylabel('Prediction/True')
  # Set y-axis range
  plt.ylim([0,3.5])
  # Set x-axis range
  if counter == 1:
    plt.xlim([0.5,1])
  elif counter == 2:
    plt.xlim([0.2,0.6])
  elif counter == 3:
    plt.xlim([0.2,0.7])
  elif counter == 4:
    plt.xlim([0,0.6])
  # Save figure
  plt.savefig('Plots/{}/{}_true_vs_prediction_{}.pdf'.format(Version,OutBaseName,Label))
  counter += 1

print('>>> ALL DONE <<<')
