"""
=================
Dataset utilities
=================
"""
import struct
import numpy as np

def read_choir_dat(filename):
    """ Parse a choir_dat file """
    with open(filename, 'rb') as f:
        n_features = struct.unpack('i', f.read(4))[0]
        n_classes = struct.unpack('i', f.read(4))[0]
        
        X = []
        y = []

        while True:
            v_in_bytes = f.read(4 * n_features)
            if v_in_bytes is None or len(v_in_bytes) == 0:
                break

            sample = struct.unpack('f' * n_features, v_in_bytes)
            label = struct.unpack('i', f.read(4))[0]
            X.append(np.array(sample, dtype=np.float32))
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, n_classes

class MatrixNormalizer(object):
    """
    Matrix Normalizer (convert values between 0. and 1.)
    Warning: Will be deprecated. Use the normalize function instead
    """
    def __init__(self):
        self.X_max = None
        self.X_min = None

    def norm(self, X, axis=0):
        self._set_max_min(X, axis)
        return (X - self.X_min) / (self.X_max - self.X_min)

    def norm_two(self, X1, X2, axis=0):
        X = np.concatenate((X1, X2))
        self._set_max_min(X, axis)
        return self.norm(X1, axis), self.norm(X2, axis)

    def _set_max_min(self, X, axis):
        assert(axis == 0 or axis == 1)
        if self.X_max is not None: # already set
            assert(self.X_min is not None)
            return

        # Compute boundary
        self.X_max = np.max(X, axis=axis)
        self.X_min = np.min(X, axis=axis)
        if len(X.shape) == 1:
            assert(self.X_max > self.X_min)
        else:
            assert(any(self.X_max > self.X_min))

        if axis == 1:
            self.X_max = self.X_max.reshape((self.X_max.shape[0], 1))
            self.X_min = self.X_min.reshape((self.X_min.shape[0], 1))


    def denorm(self, X):
        assert(self.X_max is not None)
        assert(self.X_min is not None)
        return X * (self.X_max - self.X_min) + self.X_min


def normalize(x, x_test=None, normalizer='l2'):
    if normalizer == 'l2':
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer(norm='l2').fit(x)
        x_norm = scaler.transform(x)
        if x_test is None:
            return x_norm, None
        else:
            return x_norm, scaler.transform(x_test)

    elif normalizer == 'minmax': # Here, we assume knowing the global max & min
        from sklearn.preprocessing import MinMaxScaler
        if x_test is None:
            x_data = x
        else:
            x_data = np.concatenate((x, x_test), axis=0)

        scaler = MinMaxScaler().fit(x_data)
        x_norm = scaler.transform(x)
        if x_test is None:
            return x_norm, None
        else:
            return x_norm, scaler.transform(x_test)

    raise NotImplemented
