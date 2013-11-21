from brummlearn.mlp import Mlp
import theano.tensor as T
import theano
import itertools
import numpy as np

__author__ = 'kosklain'


def squared_hinge(target, prediction):
    print target
    print prediction
    return T.maximum(1 - target * prediction, 0) ** 2


class TangMlp(Mlp):

    def __init__(
        self, n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss, optimizer, batch_size, noise_schedule,
        max_iter=1000, verbose=False):
        super(TangMlp, self).__init__(n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss, optimizer, batch_size,
            max_iter, verbose)

        self.noise_schedule = noise_schedule

    def _make_args(self, X, Z):
        args = super(TangMlp, self)._make_args(X, Z)
        def corrupt(x, level):
            return x + np.random.normal(0, level, x.shape).astype(theano.config.floatX)
        return (((corrupt(x, n), z), k) for n, ((x, z), k) in itertools.izip(self.noise_schedule, args))