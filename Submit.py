####################################################
#                                                  #
# Author: Jona Bossio (jbossios@cern.ch)           #
# Date:  25 June 2021                              #
#                                                  #
####################################################

Version  = 'v31'
Particle = 'photons'
Test     = True
BasePATH = "/eos/atlas/atlascerngroupdisk/proj-simul/AF3_Run3/Jona/Regression_Condor_Outputs/" # output path

######################################################################
## DO NOT MODIFY
######################################################################

EtaBins  = ['{}_{}'.format(x*5,x*5+5) for x in range(100)] # full detector

PATH = BasePATH + Version + '/'

import os,sys

# Create corresponding folder for logs
os.system('mkdir -p Logs/{}'.format(Version))

# Find all the local Reader's outputs
ROOTfiles = []
AllFiles = os.listdir(PATH)
for File in AllFiles:
  if ".h5" in File:
    ROOTfiles.append(File)

counter = 0
path    = "SubmissionScripts/"
for EtaBin in EtaBins:
  for File in os.listdir(path): # Loop over submission scripts files
    if ".sub" not in File:
      continue
    if Particle+'_'+EtaBin not in File:
      continue
          
    # Check if there is an output already for this job
    ROOTfileFound = False
    FileName      = File.replace(".sub","")
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
