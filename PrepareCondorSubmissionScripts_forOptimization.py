####################################################################################################
# Purpose: Create one HTCondor submission script for each combination of hyperparameters           #
# Author:  Jona Bossio                                                                             #
####################################################################################################

# Choose settings
Version               = 'HyperparameterOptimization_v5'
Particle              = 'electrons'
Loss                  = 'MSE'  # Options: mean_absolute_error (MAE) and mean_squared_error (MSE)
Nepochs               = 200
etabin                = '20_25'
UseBatchNormalization = False
UseNormalizer         = False
UseEarlyStopping      = True
UseModelCheckpoint    = True

# Choose where outputs will be saved
OutPATH = '/eos/atlas/atlascerngroupdisk/proj-simul/AF3_Run3/Jona/Regression_Condor_Outputs/'

#######################################################################################################
# DO NOT MODIFY (below this line)
#######################################################################################################

# Imports
import os,sys
from HyperParametersToOptimize import HyperParameters as Params

# Create folder for submission scripts
os.system("rm SubmissionScripts/*")
os.system("mkdir SubmissionScripts")

counter = 0
# Loop over values for ActivationType
for iActType in range(0,len(Params['ActivationType'])):
  ActivationType = Params['ActivationType'][iActType]
  # Loop over values for LearningRate
  for iLearningRate in range(0,len(Params['LearningRate'])):
    LearningRate = Params['LearningRate'][iLearningRate]
    # Loop over values for Nnodes
    for iNodes in range(0,len(Params['Nnodes'])):
      NnodesHiddenLayers = Params['Nnodes'][iNodes]
      # Loop over number of layers
      for iLayers in range(0,len(Params['Nlayers'])):
        Nlayers = Params['Nlayers'][iLayers]
        counter += 1
        # Create output folder
        outPATH = '{}/{}/{}'.format(OutPATH,Version,counter)
        os.system("mkdir -p {}".format(outPATH))
        # Prepare submission script
        ScriptName = '{}_{}_0{}'.format(Particle,etabin,counter) if counter < 10 else '{}_{}_{}'.format(Particle,etabin,counter)
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
