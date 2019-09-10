import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from drawnow import drawnow, figure
from scipy.integrate import solve_ivp


def assert_shape(x, shape):
    S = x.get_shape().as_list()
    if len(S) != len(shape):
        raise Exception("Shape mismatch: {} -- {}".format(S, shape))
    for i in range(len(S)):
        if S[i] != shape[i]:
            raise Exception("Shape mismatch: {} -- {}".format(S, shape))


def compute_delta(u, x):
    grad = tf.gradients(u, x)[0]
    g1 = tf.gradients(grad[:, 0], x)[0]
    g2 = tf.gradients(grad[:, 1], x)[0]
    delta = g1[:, 0] + g2[:, 1]
    assert_shape(delta, (None,))
    return delta


def compute_delta_nd(u, x, n):
    grad = tf.gradients(u, x)[0]
    g1 = tf.gradients(grad[:, 0], x)[0]
    delta = g1[:, 0]
    for i in range(1, n):
        g = tf.gradients(grad[:, i], x)[0]
        delta += g[:, i]
    assert_shape(delta, (None,))
    return delta


def compute_dx(u, x):
    grad = tf.gradients(u, x)[0]
    dudx = grad[:, 0]
    assert_shape(dudx, (None,))
    return dudx


def compute_dx2(u, x):
    grad = tf.gradients(u, x)[0]
    g1 = tf.gradients(grad[:, 0], x)[0]
    delta = g1[:, 0]
    assert_shape(delta, (None,))
    return delta


def compute_dy(u, x):
    grad = tf.gradients(u, x)[0]
    dudy = grad[:, 1]
    assert_shape(dudy, (None,))
    return dudy


def compute_dy2(u, x):
    grad = tf.gradients(u, x)[0]
    g1 = tf.gradients(grad[:, 1], x)[0]
    delta = g1[:, 1]
    assert_shape(delta, (None,))
    return delta


def rectspace(a, b, c, d, n):
    x = np.linspace(a, b, n)
    y = np.linspace(c, d, n)
    [X, Y] = np.meshgrid(x, y)
    return np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)


class NNODE:
    def __init__(self, batch_size, N, refn):
        def ode(t, C):
            return [C[1], -30*C[0]-31*C[1]]

        self.sol = solve_ivp(ode, (0, 1), [1, 0], method='BDF')

        self.refn = refn  # reference points
        x = np.linspace(0, 1, refn)
        self.refnX = x.reshape((-1, 1))

        self.batch_size = batch_size  # batchsize
        self.N = N

        self.x = tf.compat.v1.placeholder(tf.float64, (None, 1))
        self.u = self.u_out(self.x)
        self.loss = self.loss_function()
        self.ploss = self.point_wise_loss()
        self.fig = plt.figure()

        self.opt = tf.compat.v1.train.AdamOptimizer(
            learning_rate=0.00001).minimize(self.loss)
        self.init = tf.compat.v1.global_variables_initializer()

    def exactsol(self, x):
        raise NotImplementedError

    # data to be modified
    def A(self, x):
        raise NotImplementedError

    def B(self, x):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def lf(self, x):
        raise NotImplementedError

    def loss_function(self):
        raise NotImplementedError

    # end modification

    def subnetwork(self, x):
        for i in range(self.N):
            x = tf.layers.dense(x, 256, activation=tf.nn.tanh)
        x = tf.layers.dense(x, 1, activation=None, name="last")
        x = tf.squeeze(x, axis=1)
        return x

    def u_out(self, x):
        res = self.A(x) + self.B(x) * self.subnetwork(x)
        assert_shape(res, (None,))
        return res

    def point_wise_loss(self):
        deltah = self.lf(self.x)
        delta = self.f(self.x)
        res = tf.abs(deltah - delta)
        assert_shape(res, (None,))
        return res

    def visualize(self, sess, showonlysol=False, i=None, savefig=None):

        x = np.linspace(0, 1, self.refn)
        uh = sess.run(self.u, feed_dict={self.x: self.refnX})

        def draw():
            fig = plt.figure()
            if not showonlysol:
                plt.plot(x, self.exactsol(x), label='analytical soln')
                plt.plot(self.sol.t, self.sol.y[0].T,
                         'g-.', label='scipy solution')

            plt.plot(x, uh, label='soln')

            plt.xlabel('$time$')
            plt.ylabel('$y$')
            plt.legend()
            if i:
                # plt.title('Loss={}\nIteration {}'.format(self.rloss, i))
                plt.title(
                    'Solutions when c=31 and k=30\nLoss={}\nIteration {}'.format(self.rloss, i))
            if savefig:
                plt.savefig("{}/fig{}".format(savefig, 0 if i is None else i))

        drawnow(draw)

    def visualize_point_wise_loss(self, sess, i=None, savefig=None):
        ploss = sess.run(self.ploss, feed_dict={self.x: self.refnX})
        x = np.linspace(0, 1, self.refn)

        def draw():
            fig = plt.figure()
            Z0 = np.log(ploss + 1e-16) / np.log(10.0)
            Z0[Z0 < 1e-10] = 0
            plt.plot(x, Z0)

            plt.xlabel('$time$')
            plt.ylabel('$y$')
            if i:
                # plt.title('Loss={}\nIteration {}'.format(self.rloss, i))
                plt.title(
                    'Solutions when c=31 and k=30\nLoss={}\nIteration {}'.format(self.rloss, i))
            if savefig:
                plt.savefig(
                    "{}/ploss{}".format(savefig, 0 if i is None else i))

        drawnow(draw)

    def visualize_error(self, sess, i=None, savefig=None):
        x = np.linspace(0, 1, self.refn)
        uh = sess.run(self.u, feed_dict={self.x: x.reshape((-1, 1))})

        def draw():
            fig = plt.figure()

            plt.plot(x, self.exactsol(x)-uh)

            plt.xlabel('$time$')
            plt.ylabel('$y$')
            if i:
                # plt.title('Loss={}\nIteration {}'.format(self.rloss, i))
                plt.title(
                    'Solutions when c=31 and k=30\nLoss={}\nIteration {}'.format(self.rloss, i))
            if savefig:
                plt.savefig("{}/err{}".format(savefig, 0 if i is None else i))

        drawnow(draw)

    def train(self, sess, i):
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        X = np.random.rand(self.batch_size, 1)
        loss = 1000.0
        loss = sess.run([self.loss], feed_dict={self.x: X})[0]
        if loss > 0.5:
            _, loss = sess.run([self.opt, self.loss], feed_dict={self.x: X})
            if i % 10 == 0:
                print("Iteration={}, loss= {}".format(i, loss))

            self.rloss = loss
            return False
        else:
            self.rloss = loss
            return True


class NNPDE:
    def __init__(self, batch_size, N, refn):
        self.refn = refn  # reference points
        x = np.linspace(0, 1, refn)
        y = np.linspace(0, 1, refn)
        self.X, self.Y = np.meshgrid(x, y)
        self.refnX = np.concatenate(
            [self.X.reshape((-1, 1)), self.Y.reshape((-1, 1))], axis=1)

        self.batch_size = batch_size  # batchsize
        self.N = N

        self.x = tf.compat.v1.placeholder(tf.float64, (None, 2))
        self.u = self.u_out(self.x)
        self.loss = self.loss_function()
        self.ploss = self.point_wise_loss()
        self.fig = plt.figure()

        self.opt = tf.compat.v1.train.AdamOptimizer(
            learning_rate=0.001).minimize(self.loss)
        self.init = tf.compat.v1.global_variables_initializer()

    def exactsol(self, x, y):
        raise NotImplementedError

    # data to be modified
    def A(self, x):
        raise NotImplementedError

    def B(self, x):
        raise NotImplementedError

    def f(self, x):
        raise NotImplementedError

    def lf(self, x):
        raise NotImplementedError

    def loss_function(self):
        raise NotImplementedError

    # end modification

    def subnetwork(self, x):
        for i in range(self.N):
            x = tf.layers.dense(x, 256, activation=tf.nn.tanh)
        x = tf.layers.dense(x, 1, activation=None, name="last")
        x = tf.squeeze(x, axis=1)
        return x

    def u_out(self, x):
        res = self.A(x) + self.B(x) * self.subnetwork(x)
        assert_shape(res, (None,))
        return res

    def point_wise_loss(self):
        deltah = self.lf(self.x)
        delta = self.f(self.x)
        res = tf.abs(deltah - delta)
        assert_shape(res, (None,))
        return res

    def visualize(self, sess, showonlysol=False, i=None, savefig=None):

        x = np.linspace(0, 1, self.refn)
        y = np.linspace(0, 1, self.refn)
        [X, Y] = np.meshgrid(x, y)

        uh = sess.run(self.u, feed_dict={self.x: self.refnX})
        Z = uh.reshape((self.refn, self.refn))

        uhref = self.exactsol(X, Y)

        def draw():
            ax = self.fig.gca(projection='3d')
            if not showonlysol:
                ax.plot_surface(X, Y, uhref, rstride=1, cstride=1, cmap=cm.autumn,
                                linewidth=0, antialiased=False, alpha=0.3)

            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.summer,
                            linewidth=0, antialiased=False, alpha=0.8)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            if i:
                plt.title('Loss={}\nIteration {}'.format(self.rloss, i))
            if savefig:
                plt.savefig("{}/fig{}".format(savefig, 0 if i is None else i))

        drawnow(draw)

    def visualize_point_wise_loss(self, sess, i=None, savefig=None):
        ploss = sess.run(self.ploss, feed_dict={self.x: self.refnX})
        x = np.linspace(0, 1, self.refn)
        y = np.linspace(0, 1, self.refn)
        [X, Y] = np.meshgrid(x, y)
        Z = ploss.reshape((self.refn, self.refn))

        def draw():
            ax = self.fig.gca(projection='3d')
            Z0 = np.log(Z + 1e-16) / np.log(10.0)
            Z0[Z0 < 1e-10] = 0
            ax.plot_surface(X, Y, Z0, rstride=1, cstride=1, cmap=cm.winter,
                            linewidth=0, antialiased=False)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            if i:
                plt.title('Loss={}\nIteration {}'.format(self.rloss, i))
            if savefig:
                plt.savefig(
                    "{}/ploss{}".format(savefig, 0 if i is None else i))

        drawnow(draw)

    def visualize_error(self, sess, i=None, savefig=None):
        x = np.linspace(0, 1, self.refn)
        y = np.linspace(0, 1, self.refn)
        [X, Y] = np.meshgrid(x, y)
        uh = sess.run(self.u, feed_dict={self.x: np.concatenate(
            [X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)})
        Z = uh.reshape((self.refn, self.refn))

        def draw():
            ax = self.fig.gca(projection='3d')

            ax.plot_surface(X, Y, self.exactsol(X, Y) - Z, rstride=1, cstride=1, cmap=cm.winter,
                            linewidth=0, antialiased=False, alpha=0.85)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            if i:
                plt.title('Loss={}\nIteration {}'.format(self.rloss, i))
            if savefig:
                plt.savefig("{}/err{}".format(savefig, 0 if i is None else i))

        drawnow(draw)

    def fix(self, sess, i=None, savefig=None):
        x = np.linspace(0, 1, self.refn)
        y1 = np.ones(self.refn) * 0.3
        y2 = np.ones(self.refn) * 0.6
        uh1 = sess.run(self.u, feed_dict={self.x: np.concatenate(
            [x.reshape((-1, 1)), y1.reshape((-1, 1))], axis=1)})
        uh2 = sess.run(self.u, feed_dict={self.x: np.concatenate(
            [x.reshape((-1, 1)), y2.reshape((-1, 1))], axis=1)})

        fig = plt.figure()
        plt.plot(x, self.exactsol(x, y1), label="analytical soln")
        plt.plot(x, uh1, label="soln when y=0.3")
        fig.legend()
        if i:
            plt.title('Loss={}\nIteration {}'.format(self.rloss, i))
        if savefig:
            plt.savefig(
                "{}/comparison1_{}".format(savefig, 0 if i is None else i))

        fig = plt.figure()
        plt.plot(x, self.exactsol(x, y2), label="analytical soln")
        plt.plot(x, uh2, label="soln when y=0.6")
        fig.legend()
        if i:
            plt.title('Loss={}\nIteration {}'.format(self.rloss, i))
        if savefig:
            plt.savefig(
                "{}/comparison2_{}".format(savefig, 0 if i is None else i))

    def train(self, sess, i):
        # self.X = rectspace(0,0.5,0.,0.5,self.n)
        X = np.random.rand(self.batch_size, 2)
        loss = 1000.0
        loss = sess.run([self.loss], feed_dict={self.x: X})[
            0] / (self.refn*self.refn)
        if loss > 1e-5:
            _, loss = sess.run([self.opt, self.loss], feed_dict={self.x: X})
            if i % 100 == 0:
                print("Iteration={}, loss= {}".format(i, loss))
            return False
        else:
            self.rloss = loss
            return True


# u(x;w_1, w_2) = A(x;w_1) + B(x) * N(x;w_2)
# L u(x;w_1, w_2) = L A(x;w_1) + L( B(x) * N(x;w_2) ) --> f
