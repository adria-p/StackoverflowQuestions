from brummlearn.glm import GeneralizedLinearSparseModel
import climin
from scipy.sparse import vstack
import climin.stops
import numpy as np
import time

__author__ = 'kosklain'


class LogisticTrainer(object):

    def __init__(self, fit_data, eval_data, feature_size, tags_per_example):
        self.fit_data = fit_data
        self.eval_data = eval_data
        self.feature_size = feature_size
        self.tags_per_example = tags_per_example

    def run(self):
        num_examples = 50
        batch_size = num_examples*self.tags_per_example
        max_iter = 3000
        actual_time = time.time()
        new_time = time.time()
        print "Time spent in transforming the training dataset: "+str(new_time-actual_time)
        actual_time = new_time
        VX = []
        VZ = []
        for vx, vz in self.eval_data:
            if len(vz) != 0:
                VX.append(vx)
                VZ.append(vz)
        VX = vstack(VX, format="csr")
        VZ = np.concatenate(VZ, axis=0)
        new_time = time.time()
        print "Time spent in transforming the validation dataset: "+str(new_time-actual_time)
        stop = climin.stops.any_([
            #climin.stops.converged('loss'),
            #climin.stops.rising('val_loss', 10, 1e-5, patience=5),
            climin.stops.after_n_iterations(max_iter),
            ])
        pause = climin.stops.modulo_n_iterations(10)
        optimizer = 'rmsprop', {'steprate': 0.0001, 'momentum': 0.9, 'decay': 0.9, 'step_adapt': 0.01}
        m = GeneralizedLinearSparseModel(self.feature_size, 1, out_transfer='sigmoid', loss='fmeasure',
                                         optimizer=optimizer, batch_size=batch_size, max_iter=max_iter,
                                         num_examples=num_examples)
        losses = []
        v_losses = []
        weight_decay = ((m.parameters.in_to_out ** 2).sum())# + (m.parameters.bias**2).sum())
        weight_decay /= m.exprs['inpt'].shape[0]
        m.exprs['true_loss'] = m.exprs['loss']
        c_wd = 0.001
        m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay


        # Set up a nice printout.
        keys = '#', 'val loss', 'bias'#, 'step_length'
        max_len = max(len(i) for i in keys)
        header = '   '.join(i.ljust(max_len) for i in keys)
        print header
        print '-' * len(header)


        for i, info in enumerate(m.powerfit(self.fit_data, (VX, VZ), stop, pause)):
            v_losses.append(info['val_loss'])

            #img = tile_raster_images(fe.parameters['in_to_hidden'].T, image_dims, feature_dims, (1, 1))
            #save_and_display(img, 'filters-%i.png' % i)

            row = '%i' % i, '%.6f' % (1-info['val_loss']), '%.6f' % m.parameters['bias']#, '%.6f' % info['step_length']
            print '   '.join(i.ljust(max_len) for i in row)