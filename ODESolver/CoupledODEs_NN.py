from autograd.misc.optimizers import adam
import autograd.numpy.random as npr
from autograd import grad, elementwise_grad, jacobian
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

C0 = [1.0, 0.0, 0.0]
k1 = 1
k2 = 1
seed = int(time.time())
t = np.linspace(0, 10, 25).reshape(-1, 1)
layer_sizes = [1, 8, 3]
scale = 0.1
step_size = 0.001
i = 0    # number of training steps
N = 5001  # num_iters for training
et = 0.0  # total elapsed time


def ode(t, C):
    Ca, Cb, Cc = C
    dCadt = -k1 * Ca
    dCbdt = k1 * Ca - k2 * Cb
    dCcdt = k2 * Cb
    return [dCadt, dCbdt, dCcdt]


sol = solve_ivp(ode, (0, 10), C0)


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


def C(params, inputs):
    "Neural network functions"
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = swish(outputs)
    return outputs


# initial guess for the weights and biases
params = init_random_params(scale, layer_sizes=layer_sizes)

# Derivatives
jac = jacobian(C, 1)


def dCdt(params, t):
    i = np.arange(len(t))
    return jac(params, t)[i, :, i].reshape((len(t), 3))


def objective(params, step):
    Ca, Cb, Cc = C(params, t).T
    dCadt, dCbdt, dCcdt = dCdt(params, t).T

    z1 = np.sum((dCadt + k1 * Ca)**2)
    z2 = np.sum((dCbdt - k1 * Ca + k2 * Cb)**2)
    z3 = np.sum((dCcdt - k2 * Cb)**2)
    ic = np.sum((np.array([Ca[0], Cb[0], Cc[0]]) - C0)
                ** 2)  # initial conditions
    return z1 + z2 + z3 + ic


def callback(params, step, g):
    if step % 100 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                      objective(params, step)))


t0 = time.time()
params = adam(grad(objective), params,
              step_size=step_size, num_iters=N, callback=callback)

i += N
t1 = (time.time() - t0) / 60
et += t1

plt.plot(t, C(params, t), sol.t, sol.y.T, 'o')
plt.legend(['Ann', 'Bnn', 'Cnn', 'A', 'B', 'C'])
plt.xlabel('Time')
plt.ylabel('C')
print(f'{t1:1.1f} minutes elapsed this time. Total time = {et:1.2f} min. Total epochs = {i}.')
plt.savefig('nn-coupled_ode.png')
