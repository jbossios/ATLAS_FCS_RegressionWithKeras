
Particle = 'pions'

# PATH to input ROOT files
InputPATH = '/eos/user/j/jbossios/FastCaloSim/MicheleInputs/'

# Where to locate output CSV files
OutputPATH = '/eos/user/j/jbossios/FastCaloSim/MicheleInputsCSV/'

# Name of TTree
TTreeName = 'rootTree'

#############################################
# DO NOT MODIFY (everything below this line)
#############################################

InputPATH  += '{}/'.format(Particle)
OutputPATH += '{}/'.format(Particle)

import os,sys,csv
from ROOT import *

# CSV header
Layers  = [0,1,2,3,12]
header  = ['e_{}'.format(x) for x in Layers]
header += ['extrapWeight_{}'.format(x) for x in Layers]
header += ['etrue']

# Loop over input files
for File in os.listdir(InputPATH):

  if Particle == 'pions' and 'phiCorrected' in File: continue # skip phiCorrected files

  print('INFO: Preparing CSV file for {}'.format(InputPATH+File))

  # Read true energy from input filename
  Energy = File.split('_')[1].split('E')[1]

  # Get TTree
  tfile = TFile.Open(InputPATH+File)
  if not tfile:
    print('ERROR: {} not found, exiting'.format(InputPATH+File))
    sys.exit(1)
  tree  = tfile.Get(TTreeName)
  if not tree:
    print('WARNING: {} not found in {}, file will be skipped'.format(TTreeName,InputPATH+File))
    continue # skip file

  # Open the CSV file
  outFileName = OutputPATH+File.replace('.root','.csv')
  outFile     = open(outFileName, 'w')
  
  # Create the csv writer
  writer = csv.writer(outFile)

  # write the header
  writer.writerow(header)

  maxEvents = -1

  # Temporary fix due to problem with input TTree for E2097152
  if 'E2097152' in File:
    maxEvents = 2000
 
  # Loop over events
  counter = 0
  for event in tree: 
    counter += 1
    if maxEvents != -1:
      if counter > maxEvents: break
    # write a row to the csv file
    totalEnergy = 0 # total deposited energy
    for var in header:
      if 'e_' in var:
        totalEnergy += getattr(tree,var)
    # write fraction of energy deposited on each layer
    row = []
    for var in header:
      if 'e_' in var:
        row.append(getattr(tree,var)/totalEnergy) if totalEnergy !=0 else row.append(0)
    row += [getattr(tree,var) for var in header if 'extrapWeight_' in var]  # write extrapolation weight on each layer
    row.append(Energy)                                                      # write truth particle's energy
    writer.writerow(row)
  
  # Close the files
  outFile.close()
  tfile.Close()

print('>>> ALL DONE <<<')
