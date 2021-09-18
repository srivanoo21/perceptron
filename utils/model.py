import numpy as np

class Perceptron:
  def __init__(self, eta, epochs):
    self.weights = np.random.randn(3) * 1e-5   # Small weight
    print(f"initial weights before training: \n{self.weights}")
    self.eta = eta   # learning rate
    self.epochs = epochs  # number of epochs

  def activationFunction(self, inputs, weights):
      z = np.dot(inputs, weights)   # z = W * X
      return np.where(z > 0, 1, 0)  # Condition If True, False

  def fit(self, X, Y):
      self.X = X
      self.Y = Y
      X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]  # concatenation
      print(f"X with bias is: \n{X_with_bias}")

      for epoch in range(self.epochs):
        print("--"*10)
        print(f"for epoch: {epoch}")
        print("--"*10)

        y_hat = self.activationFunction(X_with_bias, self.weights) # forward propagation
        print(f"predicted value after forward pass is: {y_hat}")
        self.error = self.Y - y_hat
        print(f"error is: \n{self.error}")
        self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error) # backward propagation
        print(f"updated weights after epoch: \n{epoch}/{self.epochs}: \n {self.weights}")
        print("######"*10)

  def predict(self, X):
      X_with_bias = np.c_[X, -np.ones((len(X), 1))]
      return self.activationFunction(X_with_bias, self.weights)

  def total_loss(self):
      total_loss = np.sum(self.error)
      print(f"total loss is: {total_loss}")
      return total_loss