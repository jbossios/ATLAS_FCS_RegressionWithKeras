
Version = 'HyperparameterOptimization_v3'

########################################################################
# DO NOT MODIFY (below this line)
########################################################################

import os,sys

basePATH = '/afs/cern.ch/user/j/jbossios/work/public/FastCaloSim/Keras_Multipurpose_Regression/RegressionWithKeras/Logs/'

# Loop over log files
MinLoss     = 1E9
OptimalComb = -1
TestLosses  = dict()
ValLosses   = dict()
for File in os.listdir(basePATH+Version):
  if '.out' not in File: continue # skip non-log files
  # identify hyperparameter combination number
  Iter = int(File.split('.')[0].replace('photons_20_25_',''))
  # Read test_loss
  iFile            = open(basePATH+Version+'/'+File,'r')
  Lines            = iFile.readlines()
  test_loss        = float(Lines[len(Lines)-3].split(',')[0].replace('[',''))
  TestLosses[Iter] = test_loss
  # Read val_loss
  for line in Lines:
    if 'val_loss did not improve from' in line:
      val_loss        = float(line.split('from ')[1])
      ValLosses[Iter] = val_loss
  # Find combination with lowest test_loss
  if test_loss < MinLoss:
    MinLoss     = test_loss
    OptimalComb = Iter

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
        if counter == OptimalComb:
          OptimalActivationType = Params['ActivationType'][iActType]
          OptimalLearningRate   = Params['LearningRate'][iLearningRate]
          OptimalNnodes         = Params['Nnodes'][iNodes]
          OptimalNlayers        = Params['Nlayers'][iLayers]
          print('ActivationType = {}'.format(OptimalActivationType))
          print('LearningRate   = {}'.format(OptimalLearningRate))
          print('Nnodes         = {}'.format(OptimalNnodes))
          print('Nlayers        = {}'.format(OptimalNlayers))
          break

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

# Plot loss vs learning rate for optimal ActivationType,Nnodes,Nlayers
Combination = '{}_{}_{}'.format(OptimalActivationType,OptimalNnodes,OptimalNlayers)
x = [key for key in CombinationsOfInterest[Combination]]
x.sort()
yTest = [TestLosses[CombinationsOfInterest[Combination][key]] for key in x]
yVal  = [ValLosses[CombinationsOfInterest[Combination][key]] for key in x]
#TestValues = { key : TestLosses[CombinationsOfInterest['{}_{}'.format(OptimalActivationType,OptimalLearningRate)][key]] for key in CombinationsOfInterest['{}_{}'.format(OptimalActivationType,OptimalLearningRate)]}
#ValValues  = { key : ValLosses[CombinationsOfInterest['{}_{}'.format(OptimalActivationType,OptimalLearningRate)][key]]  for key in CombinationsOfInterest['{}_{}'.format(OptimalActivationType,OptimalLearningRate)]}
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
plt.savefig('Plots/v3/loss_vs_LearningRate.pdf')
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
plt.savefig('Plots/v3/loss_vs_NnodesPlusNlayers.pdf')
plt.close('all')

##################################################################################
# Plot loss vs Nlayers (for Nnodes=80)
##################################################################################

# collect info
counter = 0
CombinationsOfInterest = {'{}_{}_80'.format(OptimalActivationType,OptimalLearningRate):{}}
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
	if ActivationType==OptimalActivationType and LearningRate==OptimalLearningRate and Nnodes==80:
          CombinationsOfInterest['{}_{}_80'.format(OptimalActivationType,OptimalLearningRate)][Nlayers] = counter

# Plot loss vs Nnodes+Nlayers for optimal ActivationType,Learning
Combination = '{}_{}_80'.format(OptimalActivationType,OptimalLearningRate)
x = [key for key in CombinationsOfInterest[Combination]]
x.sort()
yTest = [TestLosses[CombinationsOfInterest[Combination][key]] for key in x]
yVal  = [ValLosses[CombinationsOfInterest[Combination][key]] for key in x]
import matplotlib.pyplot as plt
plt.figure('loss_vs_Nlayers_Nnodes80')
ax = plt.gca()
plt.plot(x,yTest, label='test_loss')
plt.plot(x,yVal, label='val_loss')
#plt.xlim([-0.75,0.75])
plt.legend()
#plt.xscale('log')
ax.text(0.02, 0.95, 'ActivationType,LearningRate: {},{}'.format(OptimalActivationType,OptimalLearningRate), verticalalignment='top', horizontalalignment='left', transform=ax.transAxes,fontsize=12)
plt.xlabel('Number of hidden layers')
plt.ylabel('Loss')
plt.savefig('Plots/v3/loss_vs_Nlayers_Nnodes80.pdf')
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
  plt.savefig('Plots/v3/loss_vs_Nnodes_Nlayers{}.pdf'.format(N))
  plt.close('all')

