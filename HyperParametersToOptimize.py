# HyperParameters dict collects all possible values for each hyperparameter that one wants to scan in the hyperparameter optimization

#HyperParameters = { # v0
#  'ActivationType' : ['relu','tanh','linear','LeakyRelu'],
#  'LearningRate'   : [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
#  'Nnodes'         : [25,50,75,100,125],
#}
#HyperParameters = { # v1
#  'ActivationType' : ['tanh'],
#  'LearningRate'   : [0.001],
#  'Nnodes'         : [20,30,40,50,60,65,70,80,90,100,100,125,150,200,250,300],
#}
#HyperParameters = { # v2
#  'ActivationType' : ['relu','tanh','linear','LeakyRelu'],
#  'LearningRate'   : [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
#  'Nnodes'         : [20,30,40,80,100,120,150,200],
#}
HyperParameters = { # v3
  'ActivationType' : ['relu','tanh','linear','LeakyRelu'],
  'LearningRate'   : [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1],
  'Nnodes'         : [20,30,40,80,100,120,150,200],
  'Nlayers'        : [1,2,3,4],
}
