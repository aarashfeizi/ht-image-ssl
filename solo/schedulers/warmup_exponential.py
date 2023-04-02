class WarmUpExponentialLR():
  def __init__(self, gamma, warmup_epochs):
    self.gamma = gamma
    self.coef = 1.0
    self.warmup_epochs = warmup_epochs
  
  def __call__(self, epoch):
    if (epoch + 1) <= self.warmup_epochs:
      return (epoch + 1) / self.warmup_epochs 
    else:
      self.coef *= self.gamma
      return self.coef