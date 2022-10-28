####################################################################################################
# Purpose: Create HTCondor submission scripts for a given particle type (one job for each eta bin) #
# Author:  Jona Bossio                                                                             #
####################################################################################################

import os
import sys

# Choose settings
Versions = {
#  'v14' : {'Particle' : 'photons'},
#  'v15' : {'Particle' : 'pions'},
#  'v16' : {'Particle' : 'electrons'},
#  'v17' : {'Particle' : 'electronsANDphotons'},
#  'v18' : {'Particle' : 'pionsANDelectrons'},
#  'v19' : {'Particle' : 'photons'},
#  'v20' : {'Particle' : 'electrons'},
#  'v21' : {'Particle' : 'pions'},
#  'v22' : {'Particle' : 'electronsANDphotons'},
#  'v23' : {'Particle' : 'photons',   'ActivationType' : 'relu', 'LearningRate' : 0.0005, 'Nnodes' : 30, 'Nlayers' : 2},
#  'v24' : {'Particle' : 'electrons', 'ActivationType' : 'relu', 'LearningRate' : 0.0001, 'Nnodes' : 40, 'Nlayers' : 2},
#  'v25' : {'Particle' : 'pions',     'ActivationType' : 'relu', 'LearningRate' : 0.0005, 'Nnodes' : 40, 'Nlayers' : 2},
#  'v26' : {'Particle' : 'electronsANDphotons', 'ActivationType' : 'relu', 'LearningRate' : 0.0005, 'Nnodes' : 30, 'Nlayers' : 2},
#  'v27' : {'Particle' : 'pionsANDelectrons',   'ActivationType' : 'relu', 'LearningRate' : 0.0005, 'Nnodes' : 40, 'Nlayers' : 3},
#  'v28' : {'Particle' : 'pions',   'ActivationType' : 'relu', 'LearningRate' : 0.0005, 'Nnodes' : 40, 'Nlayers' : 3},
#  'v29' : {'Particle' : 'all',   'ActivationType' : 'relu', 'LearningRate' : 0.0001, 'Nnodes' : 100, 'Nlayers' : 2},
#  'v30' : {'Particle' : 'photons',   'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2},
#  'v31' : {'Particle' : 'photons',   'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2},
#  'v32' : {'Particle' : 'photons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2},
#  'v33' : {'Particle' : 'pions',   'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2},
#  'v34' : {'Particle' : 'pions',   'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 40, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},
#  'v35' : {'Particle' : 'pions',   'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'MSE'},
#  'v36' : {'Particle' : 'pions',   'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 40, 'Nlayers': 2, 'Loss': 'MSE'},
#  'v37' : {'Particle' : 'pions',   'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 40, 'Nlayers': 3, 'Loss': 'MSE'},
#  'v38' : {'Particle' : 'pions',   'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 40, 'Nlayers': 3, 'Loss': 'weighted_mean_squared_error'},
#  'v39' : {'Particle' : 'pions',   'ActivationType' : 'relu', 'LearningRate' : 0.0005, 'Nnodes' : 40, 'Nlayers' : 3, 'Loss': 'MSE'},  # should match v28 (equiv to v37 but v3 inputs)
#  'v40' : {'Particle' : 'pions',   'ActivationType' : 'relu', 'LearningRate' : 0.0005, 'Nnodes' : 40, 'Nlayers' : 3, 'Loss': 'MSE'},  # should match v28 (new CSV files but from same ROOT files)
#  'v41' : {'Particle' : 'pions',   'ActivationType' : 'relu', 'LearningRate' : 0.0005, 'Nnodes' : 40, 'Nlayers' : 3, 'Loss': 'MSE'},  # using now full_detector_v2 inputs but 200 epochs and upto eta=0.8
#  'v42': {'Particle': 'photons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'MSE'},  # using full_detector v10 inputs
#  'v43': {'Particle': 'photons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},  # using full_detector v10 inputs
#  'v44': {'Particle': 'electrons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'MSE'},  # using full_detector v3 inputs
#  'v45': {'Particle': 'electrons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},  # using full_detector v3 inputs
#  'v46': {'Particle': 'electronsANDphotons', 'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'MSE'},  # using full_detector v2 inputs
#  'v47': {'Particle': 'electronsANDphotons', 'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},  # using full_detector v2 inputs
#  'v48': {'Particle': 'pions', 'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 40, 'Nlayers': 3, 'Loss': 'MSE'},  # using full_detector_v2 inputs
#  'v49': {'Particle': 'pions', 'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 40, 'Nlayers': 3, 'Loss': 'weighted_mean_squared_error'},  # using full_detector_v2 inputs
#  'v50': {'Particle': 'photons', 'ActivationType': 'LeakyRelu', 'LearningRate': 0.0001, 'Nnodes': 150, 'Nlayers': 3, 'Loss': 'weighted_mean_squared_error'},  # using full_detector_v2 inputs
#  'v51': {'Particle': 'photons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},  # using R22 v0 inputs
#  'v52': {'Particle': 'electrons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},  # using R22 v0 inputs
#  'v53': {'Particle': 'pions', 'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 40, 'Nlayers': 3, 'Loss': 'weighted_mean_squared_error'},  # using R22 v0 inputs
#  'v54': {'Particle': 'photons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},  # using R22 v1 inputs
#  'v55': {'Particle': 'electrons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},  # using R22 v1 inputs
#  'v56': {'Particle': 'pions', 'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 40, 'Nlayers': 3, 'Loss': 'weighted_mean_squared_error'},  # using R22 v1 inputs
#  'v57': {'Particle': 'electronsANDphotons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},  # using R22 v1 inputs
  'v58': {'Particle': 'electronsANDphotons', 'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2, 'Loss': 'weighted_mean_squared_error'},  # using R22 v1 inputs
  'v59': {'Particle': 'pions', 'ActivationType': 'relu', 'LearningRate': 0.0005, 'Nnodes': 40, 'Nlayers': 3, 'Loss': 'weighted_mean_squared_error'},  # using R22 v1 inputs
}
Nepochs               = 300
UseBatchNormalization = False
UseNormalizer         = False
UseEarlyStopping      = True
UseModelCheckpoint    = True

# Choose where outputs will be saved
OutPATH = '/eos/atlas/atlascerngroupdisk/proj-simul/AF3_Run3/Jona/Regression_Condor_Outputs/'

#######################################################################################################
# DO NOT MODIFY (below this line)
#######################################################################################################

if not os.path.exists('SubmissionScripts'):
  os.makedirs("SubmissionScripts")

# Loop over versions
for Version, Dict in Versions.items():

  # Create folder for submission scripts
  os.system("rm SubmissionScripts/*{}*".format(Version))

  Particle           = Dict['Particle']
  ActivationType     = Dict['ActivationType'] # other options: relu, tanh, linear, LeakyRelu
  LearningRate       = Dict['LearningRate']
  NnodesHiddenLayers = Dict['Nnodes']
  Nlayers            = Dict['Nlayers']
  Loss               = Dict['Loss']

  # Supported eta bins
  #EtaBins = ['{}_{}'.format(x*5, x*5+5) for x in range(100)] # full detector  # Temporary
  #EtaBins = ['{}_{}'.format(x*5, x*5+5) for x in range(20)] # full detector  # up to |eta|<1.0
  EtaBins = ['{}_{}'.format(x*5, x*5+5) for x in range(50)] # full detector  # up to |eta|<2.5
  #EtaBins = ['170_175'] # full detector

  # Create output folder
  outPATH = OutPATH + Version
  if not os.path.exists(outPATH):
    os.makedirs(outPATH)

  # Prepare a single submission script for each eta bin
  for etabin in EtaBins:
    ScriptName = '{}_{}_{}'.format(Particle, Version, etabin)
    outputFile = open("SubmissionScripts/"+ScriptName+".sub","w")
    outputFile.write("executable = SubmissionScripts/"+ScriptName+".sh\n")
    outputFile.write("input      = FilesForSubmission.tar.gz\n")
    outputFile.write("output     = Logs/"+Version+"/"+ScriptName+".$(ClusterId).$(ProcId).out\n")
    outputFile.write("error      = Logs/"+Version+"/"+ScriptName+".$(ClusterId).$(ProcId).err\n")
    outputFile.write("log        = Logs/"+Version+"/"+ScriptName+".$(ClusterId).log\n")
    outputFile.write("RequestMemory = 4000\n")
    outputFile.write("transfer_output_files = \"\" \n")
    outputFile.write('+JobFlavour = "workday"\n')
    outputFile.write("arguments  = $(ClusterId) $(ProcId)\n")
    outputFile.write("queue")
    outputFile.close()

    # Create bash script which will run the Reader
    outputFile = open("SubmissionScripts/"+ScriptName+".sh","w")
    outputFile.write("#!/bin/bash\n")
    outputFile.write("tar xvzf FilesForSubmission.tar.gz\n")
    outputFile.write("source /cvmfs/sft.cern.ch/lcg/views/setupViews.sh LCG_98python3 x86_64-centos7-gcc8-opt\n")
    command  = "python Regression.py"
    command += " --inputDataType Real"
    command += " --particleType "+Particle
    command += " --etaRange "+etabin
    command += " --outPATH "+outPATH
    command += " --activation "+ActivationType
    command += " --nEpochs "+str(Nepochs)
    command += " --learningRate "+str(LearningRate)
    command += " --loss "+Loss
    command += " --nNodes "+str(NnodesHiddenLayers)
    command += " --nLayers "+str(Nlayers)
    if UseBatchNormalization: command += " --useBatchNormalization"
    if UseNormalizer:         command += " --useNormalizationLayer"
    if UseEarlyStopping:      command += " --useEarlyStopping"
    if UseModelCheckpoint:    command += " --useModelCheckpoint"
    outputFile.write(command)
    outputFile.close()

print('>>> ALL DONE <<<')
