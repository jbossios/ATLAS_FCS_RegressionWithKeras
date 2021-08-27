####################################################################################################
# Purpose: Create HTCondor submission scripts for a given particle type (one job for each eta bin) #
# Author:  Jona Bossio                                                                             #
####################################################################################################

# Choose settings

Version               = 'v10'
Particle              = 'pions'
ActivationType        = 'relu' # Options: relu, tanh, linear, LeakyRelu
Nepochs               = 200
LearningRate          = 0.001
Loss                  = 'MSE'  # Options: mean_absolute_error (MAE) and mean_squared_error (MSE)
UseBatchNormalization = False
UseNormalizer         = False
UseEarlyStopping      = True
UseModelCheckpoint    = True
NnodesHiddenLayers    = 100

#######################################################################################################
# DO NOT MODIFY (below this line)
#######################################################################################################

# Supported eta bins
if Particle == 'photons' or Particle == 'electrons':
  EtaBins = ['{}_{}'.format(x*5,x*5+5) for x in range(26)]
elif Particle == 'pions':
  EtaBins = ['{}_{}'.format(x*5,x*5+5) for x in range(16)]

import os,sys

# Create folder for submission scripts
os.system("rm SubmissionScripts/*")
os.system("mkdir SubmissionScripts")

# Create output folder
outPATH = '/eos/user/j/jbossios/FastCaloSim/Regression_Condor_Outputs/{}'.format(Version)
os.system("mkdir {}".format(outPATH))

# Prepare a single submission script for each eta bin
for etabin in EtaBins:
  ScriptName = '{}_{}'.format(Particle,etabin)
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
  if UseBatchNormalization: command += " --useBatchNormalization"
  if UseNormalizer:         command += " --useNormalizationLayer"
  if UseEarlyStopping:      command += " --useEarlyStopping"
  if UseModelCheckpoint:    command += " --useModelCheckpoint"
  outputFile.write(command)
  outputFile.close()

print('>>> ALL DONE <<<')
