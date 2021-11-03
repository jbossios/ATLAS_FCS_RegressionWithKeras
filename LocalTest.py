#############################################################
#                                                           #
# Purpose: Test Regression.py (not meant for full training) #
# Author:  Jona Bossio (jbossios@cern.ch)                   #
# Usage:   python LocalTest.py                              #
#                                                           #
#############################################################

Particle              = 'all'
ActivationType        = 'relu' # Options: relu, tanh, linear, LeakyRelu
Nepochs               = 200
Nlayers               = 2
NnodesHiddenLayers    = 100
LearningRate          = 0.001
Loss                  = 'MSE'  # Options: mean_absolute_error (MAE) and mean_squared_error (MSE)
UseBatchNormalization = False
UseNormalizer         = False
UseEarlyStopping      = True
UseModelCheckpoint    = True
EtaRange              = '0_5'

######################################################################
## DO NOT MODIFY (below this line)
######################################################################

import os,sys

# Create corresponding folder for logs
os.system('mkdir -p LocalTests/')

command  = "python Regression.py"
command += " --inputDataType Real"
command += " --particleType "+Particle
command += " --etaRange "+EtaRange
command += " --outPATH LocalTests"
command += " --activation "+ActivationType
command += " --nEpochs "+str(Nepochs)
command += " --nLayers "+str(Nlayers)
command += " --learningRate "+str(LearningRate)
command += " --loss "+Loss
command += " --nNodes "+str(NnodesHiddenLayers)
if UseBatchNormalization: command += " --useBatchNormalization"
if UseNormalizer:         command += " --useNormalizationLayer"
if UseEarlyStopping:      command += " --useEarlyStopping"
if UseModelCheckpoint:    command += " --useModelCheckpoint"
print(command)
os.system(command)
