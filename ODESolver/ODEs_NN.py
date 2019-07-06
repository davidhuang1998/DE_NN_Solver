# https://kitchingroup.cheme.cmu.edu/blog/2017/11/28/Solving-ODEs-with-a-neural-network-and-autograd/
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam
import time

# dCa/dt = -k Ca(t)

# Variables
seed = int(time.time())
# layer_sizes = [1, 4, 4, 4, 1]
# step_size = 10**-4
layer_sizes = [1, 8, 1]
step_size = 0.01
num_iters = 5001
scale = 0.01
Ca0 = 1.0
train_t1 = 0
train_t2 = 8
t1 = 0
t2 = 8
i = 0
et = 0.0


def init_random_params(scale, layer_sizes, rs=npr.RandomState(seed)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


def swish(x):
    # "see https://arxiv.org/pdf/1710.05941.pdf"
    return x / (1.0 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def Ca(params, inputs):
    # "Neural network functions"
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = swish(outputs)
    return outputs


# Here is our initial guess of params:
params = init_random_params(scale, layer_sizes=layer_sizes)

# Derivatives
dCadt = elementwise_grad(Ca, 1)
d2Cadt = elementwise_grad(dCadt, 1)

t = np.linspace(train_t1, train_t2).reshape((-1, 1))

# This is the function we seek to minimize


def objective(params, step):
    # These should all be zero at the solution
    # zeq = d2Cadt(params, t) + 1001.0*dCadt(params, t) + 1000.0*Ca(params, t)
    zeq = dCadt(params, t)+25*(Ca(params, t)-np.cos(t))+np.sin(t)
    ic = Ca(params, 0.0) - Ca0
    # bc1 = dCadt(params, 0.0)
    return np.mean(zeq**2) + ic**2


def callback(params, step, g):
    if step % 1000 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                      objective(params, step)))


t3 = time.time()
params = adam(grad(objective), params,
              step_size=step_size, num_iters=num_iters, callback=callback)

t4 = (time.time() - t3) / 60
et += t4

tfit = np.linspace(t1, t2).reshape(-1, 1)
plt.plot(tfit, Ca(params, tfit), label='soln')
plt.plot(tfit, np.cos(tfit), 'r--', label='analytical soln')
plt.legend()
plt.xlabel('time')
plt.ylabel('$C_A$')
plt.xlim([t1, t2])
print(f'{t4:1.1f} minutes elapsed this time. Total time = {et:1.2f} min.')
plt.savefig('nn-ode.png')
