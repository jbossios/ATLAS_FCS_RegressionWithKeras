
Particle = 'electrons'
Version  = 'v09'

################################################################
# DO NOT MODIFY (below this line)
################################################################

if Particle == 'photons' or Particle == 'electrons':
  EtaBins = ['{}_{}'.format(x*5,x*5+5) for x in range(26)]
elif Particle == 'pions':
  EtaBins = ['{}_{}'.format(x*5,x*5+5) for x in range(16)]
else:
  print('ERROR: {} not supported yet, exiting'.format(Particle))
  sys.exit(1)

Activation                = 'relu'
NormalizationLayerInModel = True if Version == 'v01' else False

PATH = '/eos/user/j/jbossios/FastCaloSim/Regression_Condor_Outputs/{}/'.format(Version)

###############################
# Create output folders
###############################
import os,sys,ROOT
os.system('mkdir Plots')
os.system('mkdir Plots/{}'.format(Particle))
os.system('mkdir Plots/{}/{}'.format(Particle,Version))

##########################
# Loop over eta bins
##########################
for EtaBin in EtaBins:

  OutBaseName = 'Real_{}_{}_{}'.format(Activation,Particle,EtaBin)
  
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
    model = keras.models.load_model('{}/Real_{}_{}_{}_best_model.h5'.format(PATH,Activation,Particle,EtaBin),custom_objects={'Normalization': preprocessing.Normalization})
  else:
    model = keras.models.load_model('{}/Real_{}_{}_{}_best_model.h5'.format(PATH,Activation,Particle,EtaBin))
  
  ###############################
  # Get data
  ###############################
  print('INFO: Get data')
  Layers     = [0,1,2,3,12]
  if Particle == 'pions': Layers += [13,14]
  header     = ['e_{}'.format(x) for x in Layers]
  header    += ['ef_{}'.format(x) for x in Layers]
  features   = ['ef_{}'.format(x) for x in Layers]
  features  += ['etrue']
  labels     = ['extrapWeight_{}'.format(x) for x in Layers]
  header    += labels
  header    += ['etrue']
  InputFiles = dict()
  if Particle == 'photons':
    if Version == 'v01':
      path = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/photons/v3/'
    elif Version == 'v07':
      path = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/photons/v6/'
    elif Version == 'v08':
      path = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/photons/v7/'
    else:
      print('ERROR: {} not supported yet, exiting'.format(Version))
      sys.exit(1)
  elif Particle == 'pions':
    path       = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/pions/v3/'
  elif Particle == 'electrons':
    path       = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/electrons/phiCorrected/'
  else:
    print('ERROR: {} not supported yet, exiting'.format(Particle))
    sys.exit(1)
  for File in os.listdir(path):
    if 'eta_{}'.format(EtaBin) not in File: continue # select only files for the requested eta bin
    if '.csv' not in File: continue # skip non-CSV files
    Energy = File.split('E')[1].split('_')[0]
    InputFiles[Energy] = path+File
  # Import dataset using pandas
  DFs = dict()
  for energy in InputFiles:
    raw_dataset = pd.read_csv(InputFiles[energy], names=header, na_values='?', comment='\t', sep=',', skiprows=[0] , skipinitialspace=True)
    DFs[energy] = raw_dataset.copy()

  NNenergies = [energy for energy in InputFiles]
  
  ###################
  # Get FCS weights
  ###################
  print('INFO: Get FCS weights')
  
  # Get PCA bin for each event
  print('INFO: Get PCA bin for each event')
  header     = ['firstPCAbin']
  InputFiles = dict()
  path       = '/eos/user/j/jbossios/FastCaloSim/HasibInputsCSV/'
  for File in os.listdir(path):
    if EtaBin+'_zv' not in File: continue # select only files for the requested eta bin
    if Particle == 'photons':
      if 'pid22_' not in File: continue
    elif Particle == 'pions':
      if 'pid211_' not in File: continue
    elif Particle == 'electrons':
      if 'pid11_' not in File: continue
    else:
      print('{} not supported yet, exiting'.format(Particle))
      sys.exit(1)
    Energy = File.split('E')[1].split('_')[0]
    InputFiles[Energy] = path+File
  # Import dataset using pandas
  FCSDFs = dict()
  for E,InputFile in InputFiles.items():
    raw_dataset = pd.read_csv(InputFile, names=header, na_values='?', comment='\t', sep=',', skiprows=[0] , skipinitialspace=True)
    FCSDFs[E] = raw_dataset.copy() 
  
  # Get TMatrix with weight for each PCA bin and layer (and energy)
  print('INFO: Get matrix with weight for each PCA bin and layer (and energy)')
  Matrices = dict()
  path = '/eos/user/a/ahasib/Data/ParametrizationProductionVer15/'
  for Folder in os.listdir(path):
    if Particle == 'photons':
      if 'pid22_' not in Folder: continue
    elif Particle == 'pions':
      if 'pid211_' not in Folder: continue
    elif Particle == 'electrons':
      if 'pid11_' not in Folder: continue
    else:
      print('{} not supported yet, exiting'.format(Particle))
      sys.exit(1)
    if 'eta'+EtaBin.split('_')[0] not in Folder: continue # select only files for the requested eta bin
    Folder += '/'
    for FileName in os.listdir(path+Folder):
      if 'extrapol' not in FileName: continue
      Energy = FileName.split('E')[1].split('_')[0]
      File   = ROOT.TFile.Open(path+Folder+FileName)
      Matrices[Energy] = File.Get('tm_mean_weight')
  
  # Get weight from matrix
  def getWeight(row,e,layer):
    return Matrices[e](layer,int(row['firstPCAbin']))
  
  # Update each DF adding weight_x for each layer
  for energy in FCSDFs:
    for layer in Layers:
      FCSDFs[energy]['extrapWeight_{}'.format(layer)] = FCSDFs[energy].apply(lambda row: getWeight(row,energy,layer), axis=1)
  
  FCSenergies = [energy for energy in FCSDFs]
  
  # Protection
  if len(DFs) != len(FCSDFs):
    print('WARNING: Number of energies cases do not match b/w data and FCS, energies missing in one of them will be skipped')
    print('NN energies:')
    print(NNenergies)
    print('FCS energies:')
    print(FCSenergies)
  
  ###############################
  # Compare true vs prediction
  ###############################
  # Choose binning
  binning = [round(-1 + x*0.01,2) for x in range(300)]
  # Loop over energies
  for energy in FCSDFs:
    if float(energy) < 1000: continue # skip true particle energies below 1 GeV
    print('Energy = {}'.format(energy))
    print('EtaBin = {}'.format(EtaBin))
    # Prepare numpy arrays
    Nfeatures  = len(features)
    Nlabels    = len(labels)
    if energy not in DFs:
      print('WARNING: skipping energy {}'.format(energy))
      continue
    dataset               = DFs[energy]
    FCSdataset            = FCSDFs[energy]
    LayerEnergies_dataset = dataset[['e_{}'.format(x) for x in Layers]].copy().values.reshape(-1,Nlabels)
    Features_dataset      = dataset[features].copy().values.reshape(-1,Nfeatures)
    Labels_dataset        = dataset[labels].copy().values.reshape(-1,Nlabels)
    weights_FCS           = FCSdataset[labels].copy().values.reshape(-1,Nlabels)
    pred                  = model.predict(Features_dataset)
    counter = 0
    # Make figure
    #Diffs   = pred-Labels_dataset
    #FCSdiff = weights_FCS-Labels_dataset
    for Label in labels:
      counter_energy = 5 if Particle == 'photons' or Particle == 'electrons' else 7 # column number with truth energy
      nndiff  = np.array([pred[i,counter]-Labels_dataset[i,counter] for i in range(pred[:,counter].size) if LayerEnergies_dataset[i,counter] > 100])
      fcsdiff = np.array([weights_FCS[i,counter]-Labels_dataset[i,counter] for i in range(pred[:,counter].size) if LayerEnergies_dataset[i,counter] > 100])
      if nndiff.size == 0: continue # skip

      # left just in case
      #plt.figure('true_vs_prediction_{}'.format(Label))
      ## Create canvas with two panels 
      #fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)
      ## Plot prediction and true distributions
      #valsTrue, binsTrue, patchesTrue = axes[0].hist(Labels_dataset[:,counter],label=Label+' true',bins=50)
      #valsPred, binsPred, patchesPred = axes[0].hist(pred[:,counter],label=Label+' prediction',bins=50,alpha=0.5)
      #axes[0].legend()
      ## Calculate and plot prediction/true ratio
      #ratio = np.divide(valsPred,valsTrue)
      #ratio[ratio == np.inf] = 0
      #ratio[np.isnan(ratio)] = 0
      #centers = [] # bin centers
      #for ibin in range(0,len(binsTrue)-1):
      #  center = 0.5*(binsTrue[ibin+1]-binsTrue[ibin])+binsTrue[ibin]
      #  centers.append(center)
      #axes[1].plot(centers,ratio)
      ## Plot line at one
      #axes[1].plot(centers,[1 for i in range(len(centers))],'--',color='k')
      #axes[1].set_ylabel('Prediction/True')
      ## Set y-axis range
      #plt.ylim([0,3.5])
      ## Save figure
      #plt.savefig('Plots/{}/{}/eta{}_E{}_{}_true_vs_prediction_{}.pdf'.format(Particle,Version,EtaBin,energy,OutBaseName,Label))

      # Plot now the distribution of prediction-true and FCS-true
      plt.figure('TruePredictionDiff_{}'.format(Label))
      ax = plt.gca()
      #nKeras, binsKeras, _ = plt.hist(Diffs[:,counter],bins=binning,label='NN')
      #nFCS, binsFCS, _     = plt.hist(FCSdiff[:,counter],bins=binning,label='FCS',color='r',alpha=0.5)
      nKeras, binsKeras, _ = plt.hist(nndiff,bins=binning,label='NN')
      nFCS, binsFCS, _     = plt.hist(fcsdiff,bins=binning,label='FCS',color='r',alpha=0.5)
      plt.xlim([-0.75,0.75])
      plt.legend()
      ax.text(0.02, 0.95, 'Mean,RMS:', verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
      ax.text(0.02, 0.9, 'NN: {},{}'.format(round(np.mean(nndiff),2),round(np.sqrt(np.mean(np.square(nndiff))),2)), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
      ax.text(0.02, 0.85, 'FCS: {},{}'.format(round(np.mean(fcsdiff),2),round(np.sqrt(np.mean(np.square(fcsdiff))),2)), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
      ax.text(0.02, 0.8, '{}'.format(Particle), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
      ax.text(0.02, 0.75, 'Energy [MeV]: {}'.format(energy), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
      ax.text(0.02, 0.7, 'EtaBin : {}'.format(EtaBin), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
      ax.text(0.02, 0.65, 'Layer : {}'.format(Label.split('_')[1]), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
      plt.xlabel('Estimation - True')
      plt.ylabel('Events')
      plt.savefig('Plots/{}/{}/eta{}_E{}_{}_TruePredictionDiff_{}.pdf'.format(Particle,Version,EtaBin,energy,OutBaseName,Label))
      plt.close('all')
      counter += 1

print('>>> ALL DONE <<<')
