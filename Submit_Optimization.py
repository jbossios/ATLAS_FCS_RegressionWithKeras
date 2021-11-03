####################################################
#                                                  #
# Author: Jona Bossio (jbossios@cern.ch)           #
# Date:  25 June 2021                              #
#                                                  #
####################################################

Version  = 'HyperparameterOptimization_v3'
Particle = 'photons'
Test     = False
BasePATH = "/eos/user/j/jbossios/FastCaloSim/Regression_Condor_Outputs/" # output path
EtaBin   = '20_25'

######################################################################
## DO NOT MODIFY
######################################################################

import os,sys
from HyperParametersToOptimize import HyperParameters as Params

# Create corresponding folder for logs
os.system('mkdir -p Logs/{}'.format(Version))

counter = 0
path    = "SubmissionScripts/"

Iter = 0
# Loop over values for ActivationType
for iActType in range(0,len(Params['ActivationType'])):
  ActivationType = Params['ActivationType'][iActType]
  # Loop over values for LearningRate
  for iLearningRate in range(0,len(Params['LearningRate'])):
    LearningRate = Params['LearningRate'][iLearningRate]
    # Loop over values for Nnodes
    for iNodes in range(0,len(Params['Nnodes'])):
      Iter += 1
      for File in os.listdir(path): # Loop over submission scripts files
        extra = '0{}'.format(Iter) if Iter < 10 else str(Iter)
        if File != '{}_{}_{}.sub'.format(Particle,EtaBin,extra): continue
        # Check if there is an output already for this job
        ROOTfileFound = False
        FileName      = File.replace(".sub","")
        ROOTfiles     = [] # find all the local Reader's outputs
        PATH          = BasePATH + Version + '/' + str(Iter) + '/'
        AllFiles      = os.listdir(PATH)
        for File in AllFiles:
          if ".h5" in File:
            ROOTfiles.append(File)
	print(ROOTfiles)
        for rootFile in ROOTfiles: # look at reader's outputs
          if FileName in rootFile: # there is already an output for this submission script
            ROOTfileFound = True
            break
        if ROOTfileFound:
          continue
        counter += 1
        command = "condor_submit "+path+File+" &"
        if not Test: os.system(command)
if counter == 0:
  print("No need to send jobs")
else:
  if not Test: print(str(counter)+" jobs will be sent")
  else: print(str(counter)+" jobs need to be sent")
