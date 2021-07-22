import numpy as np

from hwml.nn import Linear, MSELoss, Sequential, Adam

BATCH_SIZE = 6

__w__ = np.random.randn(2, 1)
x_train = np.random.rand(BATCH_SIZE, 2)
y_train = x_train @ __w__
x_test = np.random.rand(BATCH_SIZE, 2)
y_test = x_test @ __w__


model = Sequential()

model.add(Linear(2, 3))
model.add(Linear(3, 1))
# model.set_optimizer(RawOptimizer(0.001))
# model.set_optimizer(SGD(0.001, momentum=0.1))
model.set_optimizer(Adam(0.001))
model.set_loss(MSELoss())

for i in range(30):
    loss_val = model.train(x_train, y_train)
    print(loss_val)

