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


def load_multi(prefix, fns, extras=[], alpha=0.99):
    paths = [prefix + fn for fn in fns] + list(extras)
    df = pd.concat([load_scalar_summaries(path) for path in paths]).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df.loc[df.S_noise > 100, "G_noise"] = None
    df.loc[df.S_noise > 100, "S_noise"] = None

    df["avg_G_noise"] = df.G_noise.ewm(alpha=1 - alpha).mean()
    df["avg_S_noise"] = df.S_noise.ewm(alpha=1 - alpha).mean()

    df["B_simple"] = df.avg_S_noise / df.avg_G_noise
    df["B_inst"] = df.S_noise / df.G_noise

    return df
