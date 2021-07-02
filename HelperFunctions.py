import matplotlib.pyplot as plt

# Plot loss vs epochs
def plot_loss(history,outPATH,extra,loss):
  plt.figure('{}_loss'.format(extra))
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss') # Temporary
  plt.yscale('log')
  plt.xlabel('Epoch')
  LossDef = 'Mean Absolute Error' if loss == 'MAE' else 'Mean Squared Error'
  plt.ylabel('{} (loss)'.format(LossDef))
  plt.legend()
  plt.grid(True)
  plt.savefig('{}/{}_loss_vs_epochs.pdf'.format(outPATH,extra))

