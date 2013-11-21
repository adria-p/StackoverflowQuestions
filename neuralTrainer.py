import climin
import itertools
import time
from tangMlp import TangMlp, squared_hinge
import theano.tensor as T
import climin.initialize
import climin.stops

__author__ = 'kosklain'


class NeuralTrainer(object):

    def __init__(self, X, VX, Z, VZ):
        self.X = X
        self.VX = VX
        self.Z = Z
        self.VZ = VZ

    def run(self):
        max_passes = 400
        batch_size = 200
        max_iter = max_passes * self.X.shape[0] / batch_size
        n_report = max(self.X.shape[0] / batch_size, 1)

        noise_schedule = (1 - float(i) / max_iter for i in xrange(max_iter))
        noise_schedule = itertools.chain(noise_schedule, itertools.repeat(0))

        optimizer = 'rmsprop', {'steprate': 0.001, 'momentum': 0.9, 'decay': 0.9, 'step_adapt': 0.01}
        #optimizer = 'gd', {'steprate': climin.schedule.linear_annealing(0.1, 0, max_iter),
        # 'momentum': 0.5, 'momentum_type': 'nesterov'}
        #optimizer = dropout_optimizer_conf(steprate_0=1, n_repeats=1)

        m = TangMlp(self.X.shape[1], [4000], 1, hidden_transfers=['sigmoid'],
                    out_transfer='identity', loss=squared_hinge, noise_schedule=noise_schedule,
                    optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)
        climin.initialize.randomize_normal(m.parameters.data, 0, 0.02)
        m.parameters['out_bias'][...] = 0

        weight_decay = ((m.parameters.hidden_to_out ** 2).sum())
        #                + (m.parameters.hidden_to_hidden_0**2).sum()
        #                + (m.parameters.hidden_to_out**2).sum())
        weight_decay /= m.exprs['inpt'].shape[0]
        m.exprs['true_loss'] = m.exprs['loss']
        c_wd = 0.001
        m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

        f_wd = m.function(['inpt'], c_wd * weight_decay)
        n_wrong = abs(m.exprs['output'] - m.exprs['target']).mean()
        f_n_wrong = m.function(['inpt', 'target'], n_wrong)

        losses = []
        v_losses = []
        print 'max iter', max_iter
        start = time.time()
        # Set up a nice printout.
        keys = '#', 'loss', 'val loss', 'seconds', 'wd', 'train emp', 'test emp'
        max_len = max(len(i) for i in keys)
        header = '\t'.join(i for i in keys)
        print header
        print '-' * len(header)

        #f_loss = m.function(['inpt', 'target'], ['true_loss', 'loss'])


        stop = climin.stops.any_([
            climin.stops.after_n_iterations(max_iter),
            ])

        pause = climin.stops.modulo_n_iterations(n_report)

        for i, info in enumerate(m.powerfit((self.X, self.Z), (self.VX, self.VZ), stop, pause)):
            if info['n_iter'] % n_report != 0:
                continue
            passed = time.time() - start
            losses.append(info['loss'])
            v_losses.append(info['val_loss'])
            info.update({
                'time': passed,
                'l2-loss': f_wd(self.X),
                'train_emp': f_n_wrong(self.X, self.Z),
                'test_emp': f_n_wrong(self.VX, self.VZ),
            })
            row = '%(n_iter)i\t%(loss)g\t%(val_loss)g\t%(time)g\t%(l2-loss)g\t%(train_emp)g\t%(test_emp)g' % info
            print row
        return m