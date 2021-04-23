import numpy as np
import tensorflow.compat.v1 as tf
from data.encoders import encode

cur_text_input = ["<|endoftext|>"]


def rob_pred_input(params, logger, enc=None, path_to_prompt=""):
    global cur_text_input

    def _data_gen():
        global cur_text_input
        while len(cur_text_input) > 0:
            text = cur_text_input[0]
            tokens = encode(enc, text)

            logger.info(f"tokens:\n{repr(tokens)}\n")

            if len(tokens) > params["n_ctx"]:
                logger.info(
                    "The length of your input prompt is longer than the model's context length - truncating input."
                )
                tokens = tokens[len(tokens) - params["n_ctx"] :]
            if len(tokens) < params["n_ctx"]:
                tokens = tokens + (params["n_ctx"] - len(tokens)) * [
                    params["padding_id"]
                ]
                logger.info(f"tokens (padded):\n{repr(tokens)}\n")
            t = np.array(tokens).reshape((1, params["n_ctx"]))
            logger.info(f"t:\n{repr(t)}\n")
            yield t

    dataset = tf.data.Dataset.from_generator(
        _data_gen,
        output_types=tf.int64,
        output_shapes=(
            1,
            params["n_ctx"],
        ),
    )

    def _dummy_labels(x):
        return x, x

    dataset = dataset.map(_dummy_labels)
    return dataset


def rob_handle_pred_output(predictions, logger, enc, params, out_name="test"):
    p = next(predictions)
    p = p["outputs"]

    idx = np.argmax(p == params["padding_id"])
    if idx > 0:
        p = p[:idx]

    text = enc.decode(p)

    return text
