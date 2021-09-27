class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-4 # small weight initialisations
    print(f'Initial weights before training : {self.weights}')
    self.eta = eta
    self.epochs = epochs


  def activation_function(self, inputs, weights):
    z = np.dot(inputs, weights)
    return np.where(z > 0, 1, 0)

  def fit(self, X, y):
    self.X = X
    self.y = y
    x_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
    print(f'X with bias : \n{x_with_bias}')

    for each_epoch in range(self.epochs):
      print("--"*10)
      print(f'for epoch : {each_epoch}')
      print("--"*10)

      y_hat = self.activation_function(x_with_bias, self.weights)
      print(f"Predicted value after forwaird paa : \n{y_hat}")

      self.error = self.y - y_hat
      print(f'Error : \n{self.error}')
      self.weights = self.weights + self.eta * np.dot(x_with_bias.T, self.error)
      print(f'Updated weights after epoch : {each_epoch}/{self.epochs} : {self.weights}')
      print('#######' * 10)

  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(X), 1))]
    return self.activation_function(X_with_bias, self.weights)

  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f'total loss : {total_loss}')
    return total_loss
