
Versions = { # version : particle
  'v4' : {'particles': 'photons', 'eta': '20_25'},
  'v5' : {'particles': 'electrons', 'eta': '20_25'},
  'v6' : {'particles': 'pions', 'eta': '20_25'},
  'v7' : {'particles': 'electronsANDphotons', 'eta': '20_25'},
  'v8' : {'particles': 'pionsANDelectrons', 'eta': '20_25'},
  'v9' : {'particles': 'all', 'eta': '20_25'},
  'v10' : {'particles': 'photons', 'eta': '20_25'},
  'v11' : {'particles': 'photons', 'eta': '140_145'},
  'v12' : {'particles': 'photons', 'eta': '250_255'},
  'v13' : {'particles': 'photons', 'eta': '400_405'},
  'v14' : {'particles': 'photons', 'eta': '150_155'},
  'v15' : {'particles': 'photons', 'eta': '330_335'},
  'v16' : {'particles': 'photons', 'eta': '350_355'},
  'v17' : {'particles': 'photons', 'eta': '150_155'},
  'v18' : {'particles': 'photons', 'eta': '250_255'},
  'v19' : {'particles': 'photons', 'eta': '330_335'},
  'v20' : {'particles': 'photons', 'eta': '20_25'},
  'v21' : {'particles': 'photons', 'eta': '140_145'},
  'v22' : {'particles': 'photons', 'eta': '150_155'},
  'v23' : {'particles': 'photons', 'eta': '250_255'},
  'v24' : {'particles': 'photons', 'eta': '330_335'},
  'v25' : {'particles': 'photons', 'eta': '350_355'},
  'v26' : {'particles': 'photons', 'eta': '400_405'},
  'v27' : {'particles': 'pions', 'eta': '20_25'},
  'v28' : {'particles': 'pions', 'eta': '140_145'},
  'v29' : {'particles': 'pions', 'eta': '150_155'},
  'v30' : {'particles': 'pions', 'eta': '250_255'},
  'v31' : {'particles': 'pions', 'eta': '330_335'},
  'v32' : {'particles': 'pions', 'eta': '350_355'},
  'v33' : {'particles': 'pions', 'eta': '400_405'},
}

version = 'v33'

Debug = True

# Compare loss value of optimal choice w.r.t the following baseline
Baseline = {'ActivationType': 'relu', 'LearningRate': 0.0001, 'Nnodes': 30, 'Nlayers': 2}

basePATH = '/afs/cern.ch/user/j/jbossios/work/public/FastCaloSim/Keras_Multipurpose_Regression/RegressionWithKeras/Logs/'

########################################################################
# DO NOT MODIFY (below this line)
########################################################################

Particle = Versions[version]['particles']
Eta = Versions[version]['eta']

Version = 'HyperparameterOptimization_{}'.format(version)

import os
import sys

OutputPATH = 'NewPlots/{}/{}_{}/'.format(version, Particle, Eta)
if not os.path.exists(OutputPATH):
  os.makedirs(OutputPATH)

# Loop over log files
MinLoss     = 1E9
OptimalComb = -1
TestLosses  = dict()
ValLosses   = dict()
for File in os.listdir(basePATH+Version):
  if '.out' not in File: continue # skip non-log files
  # identify hyperparameter combination number
  Iter = int(File.split('.')[0].replace('{}_{}_'.format(Particle, Eta), ''))
  # Read test_loss
  iFile = open(basePATH+Version+'/'+File,'r')
  Lines = iFile.readlines()
  if version == 'v5' and Iter == 896: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v10' and Iter == 672: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 640: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 636: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 660: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 668: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 656: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 672: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 664: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 888: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 896: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 892: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v12' and Iter == 880: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v13' and Iter == 672: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v13' and Iter == 668: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v13' and Iter == 896: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v15' and Iter == 672: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v15' and Iter == 668: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v15' and Iter == 664: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v15' and Iter == 896: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v16' and Iter == 668: continue # skip job that crashes (loss is too large anyway for this combination)
  if version == 'v16' and Iter == 672: continue # skip job that crashes (loss is too large anyway for this combination)
  if Debug: print('Iter = {}'.format(Iter))
  # Find out if training was stopped due to early stopping
  training_stopped = False
  for line in Lines:
    if 'Training' in line:
      training_stopped = True
  # Get test_loss
  if training_stopped:
    TestLosses[Iter] = float(Lines[len(Lines)-3].split(',')[0].replace('[',''))
  else:
    TestLosses[Iter] = float(Lines[len(Lines)-2].split(',')[0].replace('[',''))
  # Find combination with lowest test_loss
  if TestLosses[Iter] < MinLoss:
    MinLoss     = TestLosses[Iter]
    OptimalComb = Iter
  # Read val_loss
  if training_stopped:
    ValLosses[Iter] = float(Lines[len(Lines)-6].split(',')[0].replace('[',''))
  else:
    ValLosses[Iter] = float(Lines[len(Lines)-5].split(',')[0].replace('[',''))

print('############################################################')
print('Optimal choices with a test_loss of {} for combination #{}:'.format(MinLoss,OptimalComb))

# Display parameters for the optimal combination
sys.path.insert(1, '../') # insert at 1, 0 is the script path
from HyperParametersToOptimize import HyperParameters as Params
counter = 0
# Loop over values for ActivationType
for iActType in range(0,len(Params['ActivationType'])):
  # Loop over values for LearningRate
  for iLearningRate in range(0,len(Params['LearningRate'])):
    # Loop over values for Nnodes
    for iNodes in range(0,len(Params['Nnodes'])):
      # Loop over number of layers
      for iLayers in range(0,len(Params['Nlayers'])):
        counter += 1
        # Find baseline choice
        if Params['ActivationType'][iActType] == Baseline['ActivationType'] and Params['LearningRate'][iLearningRate] == Baseline['LearningRate'] and Params['Nnodes'][iNodes] == Baseline['Nnodes'] and Params['Nlayers'][iLayers] == Baseline['Nlayers']:
          BaselineLoss = TestLosses[counter]
	# Find optimal choice
        if counter == OptimalComb:
          OptimalActivationType = Params['ActivationType'][iActType]
          OptimalLearningRate   = Params['LearningRate'][iLearningRate]
          OptimalNnodes         = Params['Nnodes'][iNodes]
          OptimalNlayers        = Params['Nlayers'][iLayers]
          print('ActivationType = {}'.format(OptimalActivationType))
          print('LearningRate   = {}'.format(OptimalLearningRate))
          print('Nnodes         = {}'.format(OptimalNnodes))
          print('Nlayers        = {}'.format(OptimalNlayers))
          #break
print('############################################################')

# Compare losses b/w baseline and optimal choices
print('MinLoss = {}'.format(MinLoss))
print('BaselineLoss = {}'.format(BaselineLoss))

# Forcing optional n nodes to 80 for v5
if version == 'v5':
  OptimalNnodes = 80

#######################################################################
# Plot loss vs Learning rate for optimal ActivationType/Nnodes/Nlayers
#######################################################################

# collect info
counter = 0
CombinationsOfInterest = {'{}_{}_{}'.format(OptimalActivationType,OptimalNnodes,OptimalNlayers):{}}
# Loop over values for ActivationType
for iActType in range(0,len(Params['ActivationType'])):
  # Loop over values for LearningRate
  for iLearningRate in range(0,len(Params['LearningRate'])):
    # Loop over values for Nnodes
    for iNodes in range(0,len(Params['Nnodes'])):
      # Loop over number of layers
      for iLayers in range(0,len(Params['Nlayers'])):
        counter += 1
        ActivationType = Params['ActivationType'][iActType]
        LearningRate   = Params['LearningRate'][iLearningRate]
        Nnodes         = Params['Nnodes'][iNodes]
        Nlayers        = Params['Nlayers'][iLayers]
        if ActivationType==OptimalActivationType and Nnodes==OptimalNnodes and Nlayers==OptimalNlayers:
          CombinationsOfInterest['{}_{}_{}'.format(OptimalActivationType,OptimalNnodes,OptimalNlayers)][LearningRate] = counter

# Compare test vs val loss for optimal choice (not possible anymore since I'm running now with verbose=0)
Combination = '{}_{}_{}'.format(OptimalActivationType,OptimalNnodes,OptimalNlayers)
x = [key for key in CombinationsOfInterest[Combination]]
x.sort()
yTest = [TestLosses[CombinationsOfInterest[Combination][key]] for key in x]
yVal  = [ValLosses[CombinationsOfInterest[Combination][key]] for key in x]
import matplotlib.pyplot as plt
plt.figure('loss_vs_LearningRate')
ax = plt.gca()
plt.plot(x,yTest, label='test_loss')
plt.plot(x,yVal, label='val_loss')
#plt.xlim([-0.75,0.75])
plt.legend()
plt.xscale('log')
ax.text(0.02, 0.95, 'ActivationType,Nnodes,Nlayers: {},{},{}'.format(OptimalActivationType,OptimalNnodes,OptimalNlayers), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
plt.xlabel('Learning rate')
plt.ylabel('Loss')
plt.savefig('{}/loss_vs_LearningRate.pdf'.format(OutputPATH))
plt.close('all')

##################################################################################
# Plot loss vs Nnodes+Nlayers (for optimal ActivationType and learning rate)
##################################################################################

# collect info
counter = 0
CombinationsOfInterest = {'{}_{}'.format(OptimalActivationType,OptimalLearningRate):{}}
# Loop over values for ActivationType
for iActType in range(0,len(Params['ActivationType'])):
  # Loop over values for LearningRate
  for iLearningRate in range(0,len(Params['LearningRate'])):
    # Loop over values for Nnodes
    for iNodes in range(0,len(Params['Nnodes'])):
      # Loop over number of layers
      for iLayers in range(0,len(Params['Nlayers'])):
        counter += 1
        ActivationType = Params['ActivationType'][iActType]
        LearningRate   = Params['LearningRate'][iLearningRate]
        Nnodes         = Params['Nnodes'][iNodes]
        Nlayers        = Params['Nlayers'][iLayers]
        if ActivationType==OptimalActivationType and LearningRate==OptimalLearningRate:
          CombinationsOfInterest['{}_{}'.format(OptimalActivationType,OptimalLearningRate)][Nnodes+Nlayers] = counter

# Plot loss vs Nnodes+Nlayers for optimal ActivationType,Learning
Combination = '{}_{}'.format(OptimalActivationType,OptimalLearningRate)
x = [key for key in CombinationsOfInterest[Combination]]
x.sort()
yTest = [TestLosses[CombinationsOfInterest[Combination][key]] for key in x]
yVal  = [ValLosses[CombinationsOfInterest[Combination][key]] for key in x]
import matplotlib.pyplot as plt
plt.figure('loss_vs_NnodesPlusNlayers')
ax = plt.gca()
plt.plot(x,yTest, label='test_loss')
plt.plot(x,yVal, label='val_loss')
#plt.xlim([-0.75,0.75])
plt.legend()
#plt.xscale('log')
ax.text(0.02, 0.95, 'ActivationType,LearningRate: {},{}'.format(OptimalActivationType,OptimalLearningRate), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
plt.xlabel('Number of nodes + layers')
plt.ylabel('Loss')
plt.savefig('{}/loss_vs_NnodesPlusNlayers.pdf'.format(OutputPATH))
plt.close('all')

##################################################################################
# Plot loss vs Nlayers (for Nnodes=OptimalNnodes)
##################################################################################

# collect info
counter = 0
CombinationsOfInterest = {'{}_{}_{}'.format(OptimalActivationType,OptimalLearningRate,OptimalNnodes):{}}
# Loop over values for ActivationType
for iActType in range(0,len(Params['ActivationType'])):
  # Loop over values for LearningRate
  for iLearningRate in range(0,len(Params['LearningRate'])):
    # Loop over values for Nnodes
    for iNodes in range(0,len(Params['Nnodes'])):
      # Loop over number of layers
      for iLayers in range(0,len(Params['Nlayers'])):
        counter += 1
        ActivationType = Params['ActivationType'][iActType]
        LearningRate   = Params['LearningRate'][iLearningRate]
        Nnodes         = Params['Nnodes'][iNodes]
        Nlayers        = Params['Nlayers'][iLayers]
        if ActivationType==OptimalActivationType and LearningRate==OptimalLearningRate and Nnodes==OptimalNnodes:
          CombinationsOfInterest['{}_{}_{}'.format(OptimalActivationType,OptimalLearningRate,OptimalNnodes)][Nlayers] = counter

# Plot loss vs Nnodes+Nlayers for optimal ActivationType,Learning
Combination = '{}_{}_{}'.format(OptimalActivationType,OptimalLearningRate,OptimalNnodes)
x = [key for key in CombinationsOfInterest[Combination]]
x.sort()
yTest = [TestLosses[CombinationsOfInterest[Combination][key]] for key in x]
yVal  = [ValLosses[CombinationsOfInterest[Combination][key]] for key in x]
import matplotlib.pyplot as plt
plt.figure('loss_vs_Nlayers_Nnodes{}'.format(OptimalNnodes))
ax = plt.gca()
plt.plot(x,yTest, label='test_loss')
plt.plot(x,yVal, label='val_loss')
#plt.xlim([-0.75,0.75])
plt.legend()
#plt.xscale('log')
ax.text(0.02, 0.95, 'ActivationType,LearningRate: {},{}'.format(OptimalActivationType,OptimalLearningRate), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
plt.xlabel('Number of hidden layers')
plt.ylabel('Loss')
plt.savefig('{}/loss_vs_Nlayers_Nnodes{}.pdf'.format(OutputPATH,OptimalNnodes))
plt.close('all')


##################################################################################
# Plot loss vs Nnodes (for Nlayers=1,2,3,4)
##################################################################################

for N in range(1,5):

  # collect info
  counter = 0
  CombinationsOfInterest = {'{}_{}_{}'.format(OptimalActivationType,OptimalLearningRate,N):{}}
  # Loop over values for ActivationType
  for iActType in range(0,len(Params['ActivationType'])):
    # Loop over values for LearningRate
    for iLearningRate in range(0,len(Params['LearningRate'])):
      # Loop over values for Nnodes
      for iNodes in range(0,len(Params['Nnodes'])):
        # Loop over number of layers
        for iLayers in range(0,len(Params['Nlayers'])):
          counter += 1
          ActivationType = Params['ActivationType'][iActType]
          LearningRate   = Params['LearningRate'][iLearningRate]
          Nnodes         = Params['Nnodes'][iNodes]
          Nlayers        = Params['Nlayers'][iLayers]
          if ActivationType==OptimalActivationType and LearningRate==OptimalLearningRate and Nlayers==N:
            CombinationsOfInterest['{}_{}_{}'.format(OptimalActivationType,OptimalLearningRate,N)][Nnodes] = counter
  
  # Plot loss vs Nnodes+Nlayers for optimal ActivationType,Learning
  Combination = '{}_{}_{}'.format(OptimalActivationType,OptimalLearningRate,N)
  x = [key for key in CombinationsOfInterest[Combination]]
  x.sort()
  yTest = [TestLosses[CombinationsOfInterest[Combination][key]] for key in x]
  yVal  = [ValLosses[CombinationsOfInterest[Combination][key]] for key in x]
  import matplotlib.pyplot as plt
  plt.figure('loss_vs_Nnodes_Nlayers{}'.format(N))
  ax = plt.gca()
  plt.plot(x,yTest, label='test_loss')
  plt.plot(x,yVal, label='val_loss')
  #plt.xlim([-0.75,0.75])
  plt.legend()
  #plt.xscale('log')
  ax.text(0.02, 0.95, 'ActivationType,LearningRate,Nlayers: {},{}'.format(OptimalActivationType,OptimalLearningRate,N), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
  plt.xlabel('Number of nodes')
  plt.ylabel('Loss')
  plt.savefig('{}/loss_vs_Nnodes_Nlayers{}.pdf'.format(OutputPATH,N))
  plt.close('all')

