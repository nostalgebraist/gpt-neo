from pprint import pprint

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from models.gpt2 import gpt2
from utils import get_batch_size, simd_mesh_setup
from models.utils import biasmask_attn_weights
from data.encoders import fetch_encoder
from sample import sample_autoregressive

from main import main, make_argparser


def restore_ckpt_to_tf1_style(model_name: str, ckpt: str, restore_sampling: bool = False, main_extra_args: str = ""):
    ### STEP: construct params.  TODO: remove this hack and replace with the relevant fragment of main.py

    argstr = f"--model {model_name} --predict --return_to_caller " + main_extra_args
    parser = make_argparser()
    args = parser.parse_args(argstr.split())

    _, params = main(args)

    print("params:")
    pprint(params)

    ### STEP: other_features

    # Construct mtf graph + mesh from params
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])
    layout_rules = mtf.convert_to_layout_rules(params["layout"])

    # Mesh setup
    if params["use_tpu"]:
        var_placer, mesh_impl = simd_mesh_setup(params, mesh_shape, layout_rules)
    else:
        # var_placer = None
        var_placer = None
        gpu_ids = params["gpu_ids"]
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
            mesh_shape, layout_rules, gpu_ids
        )

    mesh = mtf.Mesh(graph, "my_mesh", var_placer)

    if params["precision"] == "bfloat16":
        variable_dtype = mtf.VariableDType(
            master_dtype=tf.bfloat16,
            slice_dtype=tf.float32,
            activation_dtype=tf.bfloat16,
        )
    elif params["precision"] == "bfloat16_load_to_float16":
        variable_dtype = mtf.VariableDType(
            master_dtype=tf.bfloat16,
            slice_dtype=tf.float32,
            activation_dtype=tf.float16,
        )
    elif params["precision"] == "float32_load_to_float16":
        variable_dtype = mtf.VariableDType(
            master_dtype=tf.float32, slice_dtype=tf.float32, activation_dtype=tf.float16
        )
    elif params["precision"] == "float32_load_all_to_float16":
        variable_dtype = mtf.VariableDType(
            master_dtype=tf.float32, slice_dtype=tf.float16, activation_dtype=tf.float16
        )
    elif params["precision"] == "bfloat16_load":
        variable_dtype = mtf.VariableDType(
            master_dtype=tf.bfloat16,
            slice_dtype=tf.float32,
            activation_dtype=tf.float32,
        )
    elif params["precision"] == "mixed_precision_load_bfloat16_once":
        variable_dtype = mtf.VariableDType(
            master_dtype=tf.float32,
            slice_dtype=tf.float32,
            activation_dtype=tf.bfloat16,
        )
    elif params["precision"] == "mixed_precision":
        variable_dtype = mtf.VariableDType(
            master_dtype=tf.float32,
            slice_dtype=tf.float32,
            activation_dtype=tf.bfloat16,
        )
    else:
        variable_dtype = mtf.VariableDType(
            master_dtype=tf.float32, slice_dtype=tf.float32, activation_dtype=tf.float32
        )

    sequence_length_dict = {"inputs": params["n_ctx"], "labels": params["n_ctx"]}

    params["mode"] = "predict"
    batch_size = get_batch_size(params)

    batch_dim = mtf.Dimension("batch", batch_size)
    batch_dims = [batch_dim]
    feature_length = sequence_length_dict["inputs"]
    length_dim = mtf.Dimension("sequence", feature_length)

    # Instantiate dict for dimensions, bias, etc that can be calculated here once then passed into model
    other_features = {}
    memory_length_dim = mtf.Dimension("memory_length", length_dim.size)

    attn_bias = (
        biasmask_attn_weights(mesh, length_dim, memory_length_dim, variable_dtype)
        if params["causal"]
        else None
    )

    # Add attn_bias into mtf_features
    other_features["attn_bias"] = attn_bias

    # Define other Dimensions that we'll need inside the model
    embd_dim = mtf.Dimension("embd", params["n_embd"])
    vocab_dim = mtf.Dimension("vocab", params["n_vocab"])
    # We need this because gathering when both the args have the same dimension in them breaks things
    # This dim is specifically for the weights
    # This prevents the "Einsum has lhs dimension without corresponding rhs or output dimension." error
    embed_sequence_dim = mtf.Dimension("embed_sequence", params["n_ctx"])

    other_features["embd_dim"] = embd_dim
    other_features["vocab_dim"] = vocab_dim
    other_features["embed_sequence_dim"] = embed_sequence_dim
    other_features["memory_length_dim"] = memory_length_dim

    ### STEP: mtf_features

    x = tf.placeholder(dtype=tf.int32, shape=(batch_size, params["n_ctx"]))

    features_dict = {"inputs": x}

    mtf_features = {}
    for key, x in features_dict.items():
        if x is not None:
            feature_shape = mtf.Shape(batch_dims + [length_dim])
            if type(features_dict[key]) == dict:
                features_dict[key] = features_dict[key]["feature"]
            x = tf.cast(features_dict[key], tf.int32)
            x = tf.reshape(x, feature_shape.to_integer_list)
            mtf_features[key] = mtf.import_fully_replicated(
                mesh, x, feature_shape, name=key
            )

    ### STEP: make graph + session

    if restore_sampling:
        inputs = mtf_features["inputs"]
        if params["remove_partial_sequences"] is None:
            params["remove_partial_sequences"] = False

        stop_at_token = None if params.get(
            'predict_continue_past_eot') else params["eos_id"]

        kwargs = {}
        temperature = params.get('predict_temperature')
        top_k = params.get('predict_top_k')
        if temperature:
            kwargs['temperature'] = temperature
        if top_k:
            kwargs['sampling_keep_top_k'] = top_k

        mtf_samples = sample_autoregressive(
            inputs, other_features=other_features, params=params, variable_dtype=variable_dtype,
            remove_partial_sequences=params["remove_partial_sequences"], stop_at_token=stop_at_token,
            sampling_use_entmax=params['sampling_use_entmax'], max_steps=params["predict_max_steps"],
            **kwargs
        )

        mtf_return_value = mtf_samples
    else:
        with mtf.utils.outside_all_rewrites():
            with tf.variable_scope("gpt2"):
                mtf_logits, loss, loss_batch = gpt2.model(
                    mtf_features,
                    other_features,
                    params,
                    mesh,
                    variable_dtype=variable_dtype,
                    context=None,
                )
                mtf_return_value = mtf_logits

    mtf_return_value = mtf.anonymize(mtf_return_value)
    lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=True)
    return_value = lowering.export_to_tf_tensor(mtf_return_value)

    sess = tf.Session()

    saver = tf.train.Saver(
        tf.global_variables(),
        sharded=True,
        max_to_keep=3,
        keep_checkpoint_every_n_hours=20000,
        defer_build=False,
        save_relative_paths=True,
    )

    saver.restore(sess, ckpt)

    restore_hook = mtf.MtfRestoreHook(lowering)
    restore_hook.begin()
    restore_hook.after_create_session(sess, None)

    enc = fetch_encoder(params)
    return sess, x, return_value, enc
