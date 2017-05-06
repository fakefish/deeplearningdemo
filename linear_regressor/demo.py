import numpy as np

class LinearRegressor:
  def __init__(self, interation, learning_rate, inital_m, inital_b):
    self.interation = interation
    self.learning_rate = learning_rate
    self.m = inital_m
    self.b = inital_b

  def compute_gradient(self, data):
    total_gradient_m = 0
    total_gradient_b = 0
    for i in range(len(data)):
      x,y = data[i]
      predict_y = self.predict(x)
      gradient_m = 2 * x * (predict_y - y)
      gradient_b = 2 * (predict_y - y)
      total_gradient_m += gradient_m
      total_gradient_b += gradient_b

    average_gradient_m = total_gradient_m / float(len(data))
    average_gradient_b = total_gradient_b / float(len(data))
    return average_gradient_m, average_gradient_b
  
  def train(self, data):
    for i in range(self.interation):
      gradient_m, gradient_b = self.compute_gradient(data)
      self.m -= gradient_m * self.learning_rate
      self.b -= gradient_b * self.learning_rate

  def predict(self, x):
    return self.m * x + self.b

  def computeMSE(self, data):
    total_error = 0
    for i in range(len(data)):
      x, y = data[i]
      predict_y = self.predict(x)
      error = (y - predict_y) ** 2
      total_error += error
    average_error = total_error / float(len(data))
    return average_error


def run():
  data = np.genfromtxt('data.csv', delimiter=',')
  interation = 50000
  learning_rate = 0.0001
  inital_m = 0
  inital_b = 0

  model = LinearRegressor(interation, learning_rate, inital_m, inital_b)
  print('Before training, error is {0}'.format(model.computeMSE(data)))
  model.train(data)
  print('After training, error is {0}'.format(model.computeMSE(data)))
  print('After training, m is {0} and b is {0}'.format(model.m, model.b))
run()