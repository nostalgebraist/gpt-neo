import tensorflow.compat.v1 as tf
import pandas as pd

from tensorflow.python.framework import tensor_util
from collections import defaultdict


def load_scalar_summaries(path):
    loaded = defaultdict(dict)

    for e in tf.train.summary_iterator(path):
        for v in e.summary.value:
            if hasattr(v, "tensor"):
                array = tensor_util.MakeNdarray(v.tensor)
                if array.ndim == 0:
                    loaded[e.step][v.tag] = array.item()

    return pd.DataFrame.from_dict(loaded, orient="index")
