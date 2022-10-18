####################################################
#                                                  #
# Author: Jona Bossio (jbossios@cern.ch)           #
# Date:  25 June 2021                              #
#                                                  #
####################################################

import os
import sys

Version  = 'v57'
Particle = 'electronsANDphotons'
Test     = True
BasePATH = "/eos/atlas/atlascerngroupdisk/proj-simul/AF3_Run3/Jona/Regression_Condor_Outputs/" # output path

######################################################################
## DO NOT MODIFY
######################################################################

# Temporary
#eta_bins  = ['{}_{}'.format(x*5,x*5+5) for x in range(100)] # full detector
#eta_bins  = ['{}_{}'.format(x*5,x*5+5) for x in range(20)] # up to |eta|<1.0
eta_bins  = ['{}_{}'.format(x*5,x*5+5) for x in range(50)] # up to |eta|<2.5

PATH = BasePATH + Version + '/'

# Create corresponding folder for logs
os.system('mkdir -p Logs/{}'.format(Version))

# Find all the local Reader's outputs
output_files = []
all_files = os.listdir(PATH)
for output_file in all_files:
  if ".h5" in output_file:
    output_files.append(output_file)

counter = 0
path = "SubmissionScripts/"
for eta_bin in eta_bins:
  for sub_file in os.listdir(path): # Loop over submission scripts files
    if ".sub" not in sub_file:
      continue
    if Particle+'_'+Version+'_'+eta_bin not in sub_file:
      continue
          
    # Check if there is an output already for this job
    output_file_found = False
    file_name = sub_file.replace(".sub","")
    file_name = file_name.replace('_{}'.format(Version), '')
    for output_file in output_files:  # look at condor's outputs
      if file_name in output_file:  # there is already an output for this submission script
        output_file_found = True
        break
    if output_file_found:
      continue
    counter += 1
    command = "condor_submit "+path+sub_file+" &"
    if not Test: os.system(command)
if counter == 0:
  print("No need to send jobs")
else:
  if not Test: print(str(counter)+" jobs will be sent")
  else: print(str(counter)+" jobs need to be sent")
