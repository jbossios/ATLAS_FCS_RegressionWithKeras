import matplotlib.pyplot as plt

# Plot loss vs epochs
def plot_loss(history,extra):
  plt.figure('{}_loss'.format(extra))
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss') # Temporary
  plt.yscale('log')
  plt.xlabel('Epoch')
  plt.ylabel('Mean Absolute Error (loss)')
  plt.legend()
  plt.grid(True)
  plt.savefig('{}_loss_vs_epochs.pdf'.format(extra))

