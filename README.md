
# Purpose

Train a DNN to learn how to predict the extrapolation weight for any layer/particle/eta range.

# How to clone?

git clone git@github.com:jbossios/RegressionWithKeras.git

# Dependencies

- Python 3.6+
- tensorflow
- numpy
- pandas
- matplotlib.pyplot

# How it works?

**Regression.py**:

- Run with --help to get all options
- Import input data (CSV files) using pandas
- Split data into train and test (training data = 0.8 data)
  - Validation split = 0.4
- Use Sequential API from keras to create model to train
  - Seeds are set to get reproducible results (np.random.seed(1) and tf.random.set_seed(1))
  - Supported callbacks: EarlyStopping, ModelCheckpoint and TerminateOnNaN (always ON)
- Evaluate performance on test dataset and print Loss
- Features: energy fraction on each layer and true energy (all normalized such that mean=0 and stddev=1), pdgID is also used if more than one particle type is used during training.
- Outputs:
  - One model (h5 file) for each eta range \[Real\_{ACTIVATIONTYPE}\_{PARTICLES}\_{ETARANGE}_best_model.h5\]
  - Loss vs epochs (PDF) for each eta range \[Real\_{ACTIVATIONTYPE}\_{PARTICLES}\_{ETARANGE}_loss_vs_epochs.pdf\]

**InputFiles.py**:
- Define the path for each combination of particles that want to be used during training in the *PATH2InputFiles* dict

**Note**: Input CSV files should be created using *MakeCSVfiles_NN.py* from [PrepareCSVFiles](https://github.com/jbossios/PrepareCSVFiles)

# How to run?

## Test on lxplus

Setup:
```
source Setup.sh
```

Choose parameters in *LocalTest.py* and run it:
```
python LocalTest.py
```

This script will run *Regression.py* for a single eta range. Outputs will be written to *LocalTests/*

## Run on HTCondor

Compress all scripts with Tar.sh:

```
Tar.sh
```

The above will produce *FilesForSubmission.tar.gz*

Prepare condor submission scripts with *PrepareCondorSubmissionScripts.py*:

- It is possible to run different 'versions', each using a different set of particles and hyperparameters of the DNN
- Choose where outputs will be saved with *OutPATH*
 
Run it:

```
python PrepareCondorSubmissionScripts.py
```

This will produce one submission script per eta bin for each version which will be written to *SubmissionScripts/*

Submit condor jobs for a given version (Versions+Particle) with *Submit.py*:
- **Before submitting, choose where outputs will be located with BasePATH.**
  - This will be used to check if outputs are already available, this is helpful to resend jobs for only those which were not successful
- Set Test to True to check how many jobs need to be sent (no job will actually be sent)
- Logs will be written in *Logs/{Version}/*
- Follow jobs with ```condor_q```
  - Condor tips:
    - Remove all condor jobs with ```condor_rm -all```
    - If a job appears as HOLD, check why with ```condor_q --analyze JOB_ID```

## How to perform an hyperparameter optimization (optional)

It is possible to make a scan over possible values for the hyperparameters.

Define possible variations in *HyperParametersToOptimize.py*

Prepare condor submission scripts with *PrepareCondorSubmissionScripts_forOptimization.py*

And submit condor jobs with *Submit_Optimization.py*

Example script for making plots to find optimal hyperparameters: *HyperparameterOptimization/FindOptimal.py*

If the above script crashes, it is likely due to duplicated outputs (removed outputs from failed condor jobs and keep successful retry).

## Compare predicted-true extrapolation weight distribution b/w trained network and FCS (optional)

```
cd Plotter/
```

**First:** Produce CSV files with firstPCAbin values with *MakeCSVfiles_FCS.py* script from [PrepareCSVFiles](https://github.com/jbossios/PrepareCSVFiles). This is needed to obtain the current extrapolation weights and compare with those predicted by the network

Run Setup.sh:

```
source Setup.sh
```

Take a look at *CompareExtrapWeights_True_vs_Prediction.py*:

- Choose a single particle type (photons, electrons or pions)
- Choose version (needed to know from where to pick up inputs)
- Choose output format: pdf or png (png is needed for making HTML pages, see below)
- Choose activation type (needed to pick up corresponding files)
- Choose PATH (location of input models, i.e. output of condor jobs, hence should match *OutPATH* from *PrepareCondorSubmissionScripts.py*)
- Update if needed:
  - *VersionsWithPDGID*: Versions for which PDGID was saved (all versions in which training was done with more than one particle type)
  - *VersionsWithPions*: Versions in which more than one type of particle was used AND pions where one of those types
  - *ParticlesInMultPartJobs*: Specify the particles used in versions in which the training was done with more than one particle type
- Outputs will be saved in *Plots/{PARTICLE}/{VERSION}/{FORMAT}/*:
  - There will be one plot for each eta bin, energy value and layer

Run it:

```
python CompareExtrapWeights_True_vs_Prediction.py
```

### (optional) Make HTML page with all plots with *MakeHTMLpage.py*:

- Choose pages to create (Version+Particles)
- Specify which particles were used during training for each version with the *Msg* dict
- Choose path were pages will be created with outPATH (example: */eos/user/{FIRSTLETTEROFUSERNAME}/{USERNAME}/www/*)
