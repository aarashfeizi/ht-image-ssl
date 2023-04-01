class WarmUpExponentialLR():
  def __init__(self, gamma, warmup_epochs):
    self.gamma = gamma
    self.warmup_epochs = warmup_epochs
  
  def __call__(self, epoch):
    if (epoch + 1) <= self.warmup_epochs:
      return (epoch + 1) / self.warmup_epochs 
    else:
      lr_coeff = self.gamma
      self.gamma *= self.gamma
      return lr_coeff
    