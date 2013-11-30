from brummlearn.glm import GeneralizedLinearSparseModel
import climin
from scipy.sparse import vstack
import climin.stops
import numpy as np
import theano

__author__ = 'kosklain'


class LogisticTrainer(object):

    def __init__(self, fit_data, eval_data, feature_size):
        self.fit_data = fit_data
        self.eval_data = eval_data
        self.feature_size = feature_size

    def run(self):
        batch_size = 500
        max_iter = 3000
        X = []
        Z = []
        for x, z in self.fit_data:
            X.append(x)
            Z.append(z)
        X = vstack(X, format="csr")
        Z = np.concatenate(Z, axis=0)

        VX = []
        VZ = []
        for vx, vz in self.eval_data:
            if len(vz) != 0:
                VX.append(vx)
                VZ.append(vz)
        VX = vstack(VX, format="csr")
        VZ = np.concatenate(VZ, axis=0)

        stop = climin.stops.any_([
            #climin.stops.converged('loss'),
            #climin.stops.rising('val_loss', 10, 1e-5, patience=5),
            climin.stops.after_n_iterations(max_iter),
            ])
        pause = climin.stops.modulo_n_iterations(10)
        optimizer = 'rmsprop', {'steprate': 0.0001, 'momentum': 0.9, 'decay': 0.9, 'step_adapt': 0.01}
        m = GeneralizedLinearSparseModel(self.feature_size, 1, out_transfer='sigmoid', loss='squared',
                                         optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)
        losses = []
        v_losses = []
        weight_decay = ((m.parameters.in_to_out ** 2).sum())# + (m.parameters.bias**2).sum())
        weight_decay /= m.exprs['inpt'].shape[0]
        m.exprs['true_loss'] = m.exprs['loss']
        c_wd = 0.001
        m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay


        # Set up a nice printout.
        keys = '#', 'loss', 'val loss', 'bias'#, 'step_length'
        max_len = max(len(i) for i in keys)
        header = '   '.join(i.ljust(max_len) for i in keys)
        print header
        print '-' * len(header)


        for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause)):
            losses.append(info['loss'])
            v_losses.append(info['val_loss'])

            #img = tile_raster_images(fe.parameters['in_to_hidden'].T, image_dims, feature_dims, (1, 1))
            #save_and_display(img, 'filters-%i.png' % i)

            row = '%i' % i, '%.6f' % info['loss'], '%.6f' % info['val_loss'], '%.6f' % m.parameters['bias']#, '%.6f' % info['step_length']
            print '   '.join(i.ljust(max_len) for i in row)