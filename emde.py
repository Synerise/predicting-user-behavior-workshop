import numpy as np
from sklearn.random_projection import GaussianRandomProjection


class Emde:
    def __init__(self, sketch_depth: int, sketch_width: int):
        bits = 16
        self.sketch_depth = sketch_depth
        self.sketch_width = sketch_width
        self.sp = GaussianRandomProjection(n_components=bits * sketch_depth)

    def init_biases(self, v):
        self.biases = np.array([np.percentile(v[:, i], q=50.0, axis=0) for i in range(v.shape[1])])

    def discretize(self, v):
        v = ((np.sign(v - self.biases) + 1) / 2).astype(np.uint8)
        v = np.packbits(v, axis=-1)
        v = np.frombuffer(np.ascontiguousarray(v), dtype=np.uint16).reshape(v.shape[0], -1) % self.sketch_width
        return v

    def fit(self, v):
        self.sp = self.sp.fit(v)
        vv = self.sp.transform(v)
        self.init_biases(vv)

    def transform(self, v):
        v = self.sp.transform(v)
        v = self.discretize(v)
        return v

    def transform_to_absolute_codes(self, codes: np.array):
        pos_index = np.array([i * self.sketch_width for i in range(self.sketch_depth)], dtype=np.int_)
        index = codes + pos_index
        return index


def calculate_absolute_emde_codes(sketch_depth: int, sketch_width: int, embeddings: np.array):
    emde = Emde(sketch_depth=sketch_depth, sketch_width=sketch_width)
    emde.fit(v=embeddings)
    codes = emde.transform(v=embeddings)
    return emde.transform_to_absolute_codes(codes=codes)