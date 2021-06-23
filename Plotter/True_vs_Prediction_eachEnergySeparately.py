
Activation                = 'relu'
Version                   = 'v25'
NormalizationLayerInModel = True

################################################################
# DO NOT MODIFY (below this line)
################################################################

###############################
# Create output folders
###############################
import os,sys,ROOT
os.system('mkdir Plots')
os.system('mkdir Plots/{}'.format(Version))
OutBaseName = 'Real_{}'.format(Activation)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from HelperFunctions import *
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing

###############################
# Get model
###############################
if NormalizationLayerInModel:
  model = keras.models.load_model('../Outputs/{}/Real_{}_best_model.h5'.format(Version,Activation),custom_objects={'Normalization': preprocessing.Normalization})
else:
  model = keras.models.load_model('../Outputs/{}/Real_{}_best_model.h5'.format(Version,Activation))

###############################
# Get data
###############################
print('INFO: Get data')
Layers     = [0,1,2,3,12]
header     = ['e_{}'.format(x) for x in Layers]
features   = header.copy()
features  += ['etrue']
labels     = ['extrapWeight_{}'.format(x) for x in Layers]
header    += labels
header    += ['etrue']
InputFiles = dict()
PATH       = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/v2/'
for File in os.listdir(PATH):
  Energy = File.split('E')[1].split('_')[0]
  InputFiles[Energy] = PATH+File
# Import dataset using pandas
DFs = dict()
for energy in InputFiles:
  raw_dataset = pd.read_csv(InputFiles[energy], names=header, na_values='?', comment='\t', sep=',', skiprows=[0] , skipinitialspace=True)
  DFs[energy] = raw_dataset.copy()

###################
# Get FCS weights
###################
print('INFO: Get FCS weights')

# Get PCA bin for each event
print('INFO: Get PCA bin for each event')
header     = ['firstPCAbin']
InputFiles = dict()
PATH       = '/eos/user/j/jbossios/FastCaloSim/HasibInputsCSV/'
for File in os.listdir(PATH):
  Energy = File.split('E')[1].split('_')[0]
  InputFiles[Energy] = PATH+File
# Import dataset using pandas
FCSDFs = dict()
for E,InputFile in InputFiles.items():
  raw_dataset = pd.read_csv(InputFile, names=header, na_values='?', comment='\t', sep=',', skiprows=[0] , skipinitialspace=True)
  FCSDFs[E] = raw_dataset.copy() 

# Get TMatrix with weight for each PCA bin and layer (and energy)
print('INFO: Get matrix with weight for each PCA bin and layer (and energy)')
Matrices = dict()
PATH = '/eos/user/j/jbossios/FastCaloSim/HasibInputs/'
for Folder in os.listdir(PATH):
  if 'pid' not in Folder: continue
  Folder += '/'
  for FileName in os.listdir(PATH+Folder):
    if 'extrapol' not in FileName: continue
    Energy = FileName.split('E')[1].split('_')[0]
    File   = ROOT.TFile.Open(PATH+Folder+FileName)
    Matrices[Energy] = File.Get('tm_mean_weight')

# Get weight from matrix
def getWeight(row,e,layer):
  return Matrices[e](layer,int(row['firstPCAbin']))

# Update each DF adding weight_x for each layer
for energy in FCSDFs:
  for layer in Layers:
    FCSDFs[energy]['extrapWeight_{}'.format(layer)] = FCSDFs[energy].apply(lambda row: getWeight(row,energy,layer), axis=1)

# Protection
if len(DFs) != len(FCSDFs):
  print('ERROR: Number of energies cases do not match b/w data and FCS, exiting')
  sys.exit(0)

###############################
# Compare true vs prediction
###############################
# Choose binning
binning = [round(-1 + x*0.01,2) for x in range(300)]
#for i in range(300):
#  binning.append(round(-1 + i*0.01,2))
# Loop over energies
for energy in FCSDFs:
  print('Energy = {}'.format(energy))
  # Prepare numpy arrays
  Nfeatures  = len(features)
  Nlabels    = len(labels)
  dataset    = DFs[energy]
  FCSdataset = FCSDFs[energy]
  Features_dataset = dataset[features].copy().values.reshape(-1,Nfeatures)
  Labels_dataset   = dataset[labels].copy().values.reshape(-1,Nlabels)
  weights_FCS      = FCSdataset[labels].copy().values.reshape(-1,Nlabels)
  pred             = model.predict(Features_dataset)
  counter = 0
  # Make figure
  Diffs   = pred-Labels_dataset
  FCSdiff = weights_FCS-Labels_dataset
  for Label in labels:
    plt.figure('true_vs_prediction_{}'.format(Label))
    # Create canvas with two panels 
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
    #if counter == 1:
    #  plt.xlim([0.5,1])
    #elif counter == 2:
    #  plt.xlim([0.2,0.6])
    #elif counter == 3:
    #  plt.xlim([0.2,0.7])
    #elif counter == 4:
    #  plt.xlim([0,0.6])
    # Save figure
    plt.savefig('Plots/{}/E{}_{}_true_vs_prediction_{}.pdf'.format(Version,energy,OutBaseName,Label))
    # Plot now the distribution of prediction-true and FCS-true
    plt.figure('TruePredictionDiff_{}'.format(Label))
    ax = plt.gca()
    nKeras, binsKeras, _ = plt.hist(Diffs[:,counter],bins=binning,label='NN')
    nFCS, binsFCS, _     = plt.hist(FCSdiff[:,counter],bins=binning,label='FCS',color='r',alpha=0.5)
    plt.xlim([-0.75,0.75])
    plt.legend()
    ax.text(0.02, 0.95, 'Mean,RMS:', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
    ax.text(0.02, 0.9, 'NN: {},{}'.format(round(np.mean(Diffs[:,counter]),2),round(np.sqrt(np.mean(np.square(Diffs[:,counter]))),2)), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
    ax.text(0.02, 0.85, 'FCS: {},{}'.format(round(np.mean(FCSdiff[:,counter]),2),round(np.sqrt(np.mean(np.square(FCSdiff[:,counter]))),2)), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
    plt.xlabel('Estimation - True')
    plt.ylabel('Events')
    plt.savefig('Plots/{}/E{}_{}_TruePredictionDiff_{}.pdf'.format(Version,energy,OutBaseName,Label))
    plt.close('all')
    counter += 1

print('>>> ALL DONE <<<')
