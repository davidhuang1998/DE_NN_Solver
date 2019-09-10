from matplotlib import pyplot as plt
from pdebase import *
import time


class Problem_ODE(NNODE):
    # data to be modified
    def exactsol(self, x):
        return 30/29*np.exp(-x)-np.exp(-30*x)/29

    def A(self, x):
        return 1.0
        # return 0

    def B(self, x):
        return x[:, 0]**2

    def f(self, x):
        return 0
        # return -2 * np.pi ** 2 * tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1])

    def lf(self, x):
        return compute_dx2(self.u, self.x) + 31.0 * compute_dx(self.u, self.x) + 30.0 * self.u

    def loss_function(self):
        deltah = self.lf(self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res
        # end modification


class Problem1(NNPDE):
    # data to be modified
    def exactsol(self, x, y):
        return (1 / (np.exp(np.pi) - np.exp(-np.pi))) * np.sin(np.pi * x) * (np.exp(np.pi * y) - np.exp(-np.pi * y))
        # return np.sin(np.pi * x) * np.sin(np.pi * y)

    def A(self, x):
        return x[:, 1] * tf.sin(np.pi * x[:, 0])
        # return 0

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return 0
        # return -2 * np.pi ** 2 * tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1])

    def lf(self, x):
        return compute_delta(self.u, self.x)

    def loss_function(self):
        deltah = self.lf(self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res
        # end modification


class ProblemPeak(NNPDE):
    def __init__(self, batch_size, N, refn):
        self.alpha = 1000
        self.xc = 0.5
        self.yc = 0.5
        NNPDE.__init__(self, batch_size, N, refn)

    # data to be modified
    def exactsol(self, x, y):
        return np.exp(-1000 * ((x - self.xc) ** 2 + (y - self.yc) ** 2))

    def A(self, x):
        return tf.exp(-1000 * ((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2)) + tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1])

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def tfexactsol(self, x):
        return tf.exp(-1000 * ((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2))

    def f(self, x):
        return -4*self.alpha*self.tfexactsol(self.x) + 4*self.alpha**2*self.tfexactsol(self.x) * \
            ((x[:, 0] - self.xc) ** 2 + (x[:, 1] - self.yc) ** 2)

    def loss_function(self):
        deltah = compute_delta(self.u, self.x)
        delta = self.f(self.x)
        res = tf.reduce_sum((deltah - delta) ** 2)
        assert_shape(res, ())
        return res

    def train(self, sess, i=-1):
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        X = np.random.rand(self.batch_size, 2)
        # X = np.concatenate([X, rectspace(0.4, 0.5, 0.4, 0.5, 5)], axis=0)
        _, loss = sess.run([self.opt, self.loss], feed_dict={self.x: X})
        if i % 10 == 0:
            print("Iteration={}, loss= {}".format(i, loss))


class ProblemBLSingularity(NNPDE):
    # data to be modified
    def exactsol(self, x, y):
        return x**0.6

    def A(self, x):
        return x[:, 0]**0.6+tf.sin(np.pi * x[:, 0]) * tf.sin(np.pi * x[:, 1])

    def B(self, x):
        return x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])

    def f(self, x):
        return 0.6*(0.6-1)*x[:, 0]**(0.6-2)


dir = 'ODE'

if dir == 'p1':
    npde = Problem1(64, 3, 50)  # works very well
    with tf.Session() as sess:
        sess.run(npde.init)
        i = 0
        t = time.time()

        while True:
            if time.time()-t >= 1800:
                break

            if npde.train(sess, i):
                break
            else:
                i = i + 1

        npde.visualize(sess, False, i=i, savefig=dir)
        npde.visualize_point_wise_loss(sess, i=i, savefig=dir)
        npde.visualize_error(sess, i=i, savefig=dir)
        npde.fix(sess, i=i, savefig=dir)
elif dir == 'ODE':
    npde = Problem_ODE(64, 24, 50)  # works very well
    with tf.Session() as sess:
        sess.run(npde.init)
        i = 0
        t = time.time()

        while True:
            if time.time()-t >= 1800:
                break

            if npde.train(sess, i) or i >= 20000:
                break
            else:
                i = i + 1

        npde.visualize(sess, False, i=i, savefig=dir)
        npde.visualize_point_wise_loss(sess, i=i, savefig=dir)
        npde.visualize_error(sess, i=i, savefig=dir)
