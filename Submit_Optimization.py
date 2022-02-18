####################################################
#                                                  #
# Author: Jona Bossio (jbossios@cern.ch)           #
# Date:  25 June 2021                              #
#                                                  #
####################################################

Version  = 'HyperparameterOptimization_v13'
Particle = 'photons'
Test     = False
BasePATH = "/eos/atlas/atlascerngroupdisk/proj-simul/AF3_Run3/Jona/Regression_Condor_Outputs/" # output path
EtaBin   = '400_405'

######################################################################
## DO NOT MODIFY
######################################################################

import os,sys
from HyperParametersToOptimize import HyperParameters as Params

# Create corresponding folder for logs
os.system('mkdir -p Logs/{}'.format(Version))

counter = 0
path = "SubmissionScripts/"

# Remove all .dag files
os.system('rm SubmissionScripts/{}.dag*'.format(Version))

# Create .dag file
DAGFileName = 'SubmissionScripts/{}.dag'.format(Version)
outputFile = open(DAGFileName,'w')

totalJobs = 0

Iter = 0
# Loop over values for ActivationType
for iActType in range(0,len(Params['ActivationType'])):
  ActivationType = Params['ActivationType'][iActType]
  # Loop over values for LearningRate
  for iLearningRate in range(0,len(Params['LearningRate'])):
    LearningRate = Params['LearningRate'][iLearningRate]
    # Loop over values for Nnodes
    for iNodes in range(0,len(Params['Nnodes'])):
      # Loop over values for Nlayers
      for iLayers in range(0,len(Params['Nlayers'])):
        Iter += 1
        Files = [File for File in os.listdir(path) if '{}_{}'.format(Particle, EtaBin) in File]
        for File in Files: # Loop over submission scripts files
          extra = '0{}'.format(Iter) if Iter < 10 else str(Iter)
          if File != '{}_{}_{}.sub'.format(Particle,EtaBin,extra): continue
          # Check if there is an output already for this job
          ROOTfileFound = False
          FileName      = '{}_{}'.format(Particle,EtaBin)
          ROOTfiles     = [] # find all the local Reader's outputs
          PATH          = BasePATH + Version + '/' + str(Iter) + '/'
          AllFiles      = os.listdir(PATH)
          for File in AllFiles:
            if ".h5" in File:
              ROOTfiles.append(File)
          for rootFile in ROOTfiles: # look at reader's outputs
            if FileName in rootFile: # there is already an output for this submission script
              ROOTfileFound = True
              break
          totalJobs += 1
          if ROOTfileFound:
            continue
          counter += 1
          outputFile.write('JOB A{} {}/{}\n'.format(counter, path, File))
          outputFile.write('RETRY A{} 2\n'.format(counter))
outputFile.close()
if counter == 0:
  print("No need to send jobs")
else:
  command = 'condor_submit_dag {} &'.format(DAGFileName)
  if not Test:
    print("1 job will be sent with {} subjobs (out of {} possible subjobs)".format(counter,totalJobs))
    os.system(command)
  else:
    print("1 job needs to be sent with {} subjobs (out of {} possible subjobs)".format(counter,totalJobs))
    print(command)
