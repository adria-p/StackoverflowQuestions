from brummlearn.glm import GeneralizedLinearSparseModel
from scipy.sparse import csr_matrix
import numpy as np
import time


__author__ = 'kosklain'


class LogisticPredictor(object):

    def __init__(self, eval_data, feature_size, num_tags, parameters_file=None):
        self.eval_data = eval_data
        self.feature_size = feature_size
        self.num_tags = num_tags
        self.parameters_file = parameters_file

    def get_validation_data(self):
        indices_VX = []
        data_VX = []
        indptr_VX = []
        first = True
        for (new_data, new_indices, new_indptr) in self.eval_data:
            indices_VX.append(new_indices)
            data_VX.append(new_data)
            if first:
                indptr_VX.append(new_indptr)
                first = False
            else:
                indptr_VX.append(indptr_VX[-1][-1]+new_indptr[1:])
        indptr_VX = np.concatenate(indptr_VX)
        data_VX = np.concatenate(data_VX)
        indices_VX = np.concatenate(indices_VX)
        VX = csr_matrix((data_VX, indices_VX, indptr_VX),
                        shape=(len(indices_VX)/self.num_tags, self.feature_size),
                        dtype=np.float64)
        return VX

    def run(self):
        num_examples = 20
        batch_size = num_examples*self.num_tags
        max_iter = 3000
        actual_time = time.time()
        new_time = time.time()
        print "Time spent in transforming the training dataset: "+str(new_time-actual_time)
        actual_time = new_time
        new_time = time.time()
        print "Time spent in transforming the validation dataset: "+str(new_time-actual_time)
        optimizer = 'rmsprop', {'steprate': 0.0001, 'momentum': 0.9, 'decay': 0.9, 'step_adapt': False} #0.01
        m = GeneralizedLinearSparseModel(self.feature_size, 1, out_transfer='sigmoid', loss='fmeasure',
                                         optimizer=optimizer, batch_size=batch_size, max_iter=max_iter,
                                         num_examples=num_examples)
        weight_decay = ((m.parameters.in_to_out ** 2).sum())# + (m.parameters.bias**2).sum())
        weight_decay /= m.exprs['inpt'].shape[0]
        m.exprs['true_loss'] = m.exprs['loss']
        c_wd = 0.001
        m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

        m.parameters.data = np.load(self.parameters_file)

        tags = np.array(self.eval_data.cv.get_feature_names())
        for (data, indices, indptr) in self.eval_data:
            TX = csr_matrix((data, indices, indptr),
                        shape=(self.num_tags, self.feature_size),
                        dtype=np.float64)
            predictions = np.array(m.predict(TX)).flatten()
            selected_tags = tags[predictions > 0.5]
            print selected_tags


