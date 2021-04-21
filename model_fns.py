import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.tpu import tpu_estimator
import mesh_tensorflow.transformer as mtf_transformer
from optimizers import get_optimizer
from utils import (create_host_call, get_graph_info, simd_mesh_setup, add_mode_to_params,
                   get_batch_size, auto_layout, auto_layout_and_mesh_shape, serialize_training_step,
                   squared_global_norm)
from models.utils import biasmask_attn_weights
from tensorflow.python.ops import resources
from sample import sample_autoregressive
from models.gpt2 import gpt2
import math


from tensorflow.python.training.saver import BaseSaverBuilder


class CastFromBFloat16SaverBuilder(BaseSaverBuilder):
    # Based on tensorflow.python.training.saver.BulkSaverBuilder.bulk_restore
    def bulk_restore(self, filename_tensor, saveables, preferred_shard,
                     restore_sequentially):
        from tensorflow.python.framework import ops
        from tensorflow.python.ops import io_ops
        # Ignored: bulk restore is internally sequential.
        del restore_sequentially
        restore_specs = []
        orig_spec_dtypes = []
        for saveable in saveables:
            for spec in saveable.specs:
                _f16 = tf.as_dtype('bfloat16')
                _f32 = tf.as_dtype('float32')
                expect_dtype = _f16 if (spec.dtype == _f32 and (
                    'adam' not in spec.name)) else spec.dtype
                restore_specs.append(
                    (spec.name, spec.slice_spec, expect_dtype))
                orig_spec_dtypes.append(spec.dtype)

        names, slices, dtypes = zip(*restore_specs)
        # Load all tensors onto CPU 0 for compatibility with existing code.
        with ops.device("cpu:0"):
            restored = io_ops.restore_v2(
                filename_tensor, names, slices, dtypes)
            casted = []
            for r, dt, rs, osd in zip(restored, dtypes, restore_specs, orig_spec_dtypes):
                c = tf.cast(r, osd)
                tf.logging.info(
                    f"{repr(r)}\n{repr(c)}\n\n{repr(rs)}\n\t{r.dtype}\n\t{c.dtype}\n")
                casted.append(c)
            return casted


def model_fn(features, labels, mode, params):
    # Get global step
    global_step = tf.train.get_global_step()

    # Construct mtf graph + mesh from params
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(params["mesh_shape"])
    layout_rules = mtf.convert_to_layout_rules(params["layout"])

    # Mesh setup
    if params["use_tpu"]:
        var_placer, mesh_impl = simd_mesh_setup(
            params, mesh_shape, layout_rules)
    else:
        var_placer = None
        gpu_ids = params["gpu_ids"]
        mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
            mesh_shape, layout_rules, gpu_ids)

    # Trainable variable precision
    # Store to checkpoints in master type, train in slice type, compute in activation type
    if params["precision"] == "bfloat16":
        variable_dtype = mtf.VariableDType(master_dtype=tf.bfloat16, slice_dtype=tf.float32,
                                           activation_dtype=tf.bfloat16)
    elif params["precision"] == "bfloat16_load_to_float16":
        variable_dtype = mtf.VariableDType(master_dtype=tf.bfloat16, slice_dtype=tf.float32,
                                           activation_dtype=tf.float16)
    elif params["precision"] == "float32_load_to_float16":
        variable_dtype = mtf.VariableDType(master_dtype=tf.float32, slice_dtype=tf.float32,
                                           activation_dtype=tf.float16)
    elif params["precision"] == "float32_load_all_to_float16":
        variable_dtype = mtf.VariableDType(master_dtype=tf.float32, slice_dtype=tf.float16,
                                           activation_dtype=tf.float16)
    elif params["precision"] == "bfloat16_load":
        variable_dtype = mtf.VariableDType(master_dtype=tf.bfloat16, slice_dtype=tf.float32,
                                           activation_dtype=tf.float32)
    elif params["precision"] == "mixed_precision_load_bfloat16_once":
        variable_dtype = mtf.VariableDType(master_dtype=tf.float32, slice_dtype=tf.float32,
                                           activation_dtype=tf.bfloat16)
    elif params["precision"] == "mixed_precision":
        variable_dtype = mtf.VariableDType(master_dtype=tf.float32, slice_dtype=tf.float32,
                                           activation_dtype=tf.bfloat16)
    else:
        variable_dtype = mtf.VariableDType(
            master_dtype=tf.float32, slice_dtype=tf.float32, activation_dtype=tf.float32)

    # Build mtf mesh object
    mesh = mtf.Mesh(graph, "my_mesh", var_placer)

    # Build mtf_features & seq length dict for getting number of microbatches
    # We need to pack inputs into a dict to pass into serialize_training_step
    features_dict = {"inputs": features, "labels": labels}
    sequence_length_dict = {
        "inputs": params["n_ctx"], "labels": params["n_ctx"]}

    if mode == tf.estimator.ModeKeys.PREDICT and params['predict_no_pad']:
        sequence_length_dict["inputs"] = features.shape[1]

    params = add_mode_to_params(params, mode)
    batch_size = get_batch_size(params)

    batch_dim = mtf.Dimension("batch", batch_size)
    batch_dims = [batch_dim]
    feature_length = sequence_length_dict["inputs"]
    length_dim = mtf.Dimension("sequence", feature_length)

    mtf_features = {}
    for key, x in features_dict.items():
        if x is not None:
            feature_shape = mtf.Shape(batch_dims + [length_dim])
            if type(features_dict[key]) == dict:
                features_dict[key] = features_dict[key]["feature"]
            x = tf.cast(features_dict[key], tf.int32)
            x = tf.reshape(x, feature_shape.to_integer_list)
            mtf_features[key] = mtf.import_fully_replicated(
                mesh, x, feature_shape, name=key)

    # Instantiate dict for dimensions, bias, etc that can be calculated here once then passed into model
    other_features = {}
    memory_length_dim = mtf.Dimension("memory_length", length_dim.size)

    attn_bias = biasmask_attn_weights(
        mesh, length_dim, memory_length_dim, variable_dtype) if params["causal"] else None

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

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Set up the model for prediction
        inputs = mtf_features["inputs"]
        if params["remove_partial_sequences"] is None:
            params["remove_partial_sequences"] = False

        export = params.get("export", False)

        if not export:
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

        else:
            with mtf.utils.outside_all_rewrites():
                with tf.variable_scope('gpt2'):
                    mtf_samples, loss, loss_batch = gpt2.model(mtf_features, other_features, params, mesh,
                                                               variable_dtype=variable_dtype, context=None)

        mtf_samples = mtf.anonymize(mtf_samples)
        inputs = mtf.anonymize(inputs)
        lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=True)
        inputs = lowering.export_to_tf_tensor(inputs)
        outputs = lowering.export_to_tf_tensor(mtf_samples)
        predictions = {
            "inputs": inputs,
            "outputs": outputs}

        def scaffold_fn():
            return tf.train.Scaffold(
                local_init_op=tf.group(
                    tf.train.Scaffold.default_local_init_op(),
                    lowering.copy_masters_to_slices(),
                    name="mtf_local_init_op"),
                ready_op=tf.concat(
                    [tf.report_uninitialized_variables(),
                     resources.report_uninitialized_resources()],
                    axis=0,
                    name="mtf_ready_op"))

        return tpu_estimator.TPUEstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            scaffold_fn=scaffold_fn,
            prediction_hooks=[mtf.MtfRestoreHook(lowering)])

    # We're not predicting, so we better be training or evaluating
    assert (mode == tf.estimator.ModeKeys.TRAIN or mode ==
            tf.estimator.ModeKeys.EVAL)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Gets number of microbatches per batch for serialized training
        # if param tokens_per_mb_per_replica = None, this defaults to 1 and no microbatching is performed
        num_microbatches = int(mtf_transformer.utils.serialize_num_microbatches(batch_dim=batch_dim,
                                                                                sequence_length=sequence_length_dict,
                                                                                mesh_shape=mesh_shape,
                                                                                layout_rules=layout_rules,
                                                                                tokens_per_microbatch_per_replica=params["tokens_per_mb_per_replica"]))
    else:
        num_microbatches = 1

    # Add num microbatches to params
    params["num_microbatches"] = num_microbatches

    if num_microbatches > 1:
        # For serialize_training_step we need to modify the model to output results in a dict
        def serialized_fn(mtf_features):
            if params["model"] == "GPT":
                with tf.variable_scope('gpt2'):
                    logits, loss, loss_batch = gpt2.model(mtf_features, other_features, params, mesh,
                                                          variable_dtype=variable_dtype)
                return {"logits": logits, "loss": loss, "loss_batch": loss_batch}
            else:
                raise Exception(
                    f"'{params['model']}' is not a valid model - please select from [GPT]")

        # Serialize the training step - Gradients are accumulated locally and reduced once.
        grad_fn = None
        if params['noise_scale']:
            grad_fn = squared_global_norm
        var_grads, output_dict = serialize_training_step(
            mtf_features, serialized_fn, batch_dim, num_microbatches, grad_fn=grad_fn)

        loss = output_dict["loss"]
        loss_batch = output_dict["loss_batch"]
        logits = output_dict["logits"]

        if params['noise_scale']:
            gn_small = output_dict['squared_global_norm']
            gn_big = squared_global_norm(var_grads)['squared_global_norm']

            # cancels an extra factor due to `loss` being scaled down by 1/num_microbatches in `gpt2`
            #
            # if L is the unscaled loss, then we are summing this over the microbatches:
            #       (grad norm of L / num_microbatches)^2
            # the sum has num_microbatches terms, so it's
            #        ~ num_microbatches * (grad norm of L)^2 / (num_microbatches)^2
            # with an extra 1/num_microbatches` left
            gn_small = gn_small * num_microbatches

            B_small = params['tokens_per_mb_per_replica'] / params['n_ctx']
            B_big = batch_size

            G_noise = (B_big * gn_big - B_small * gn_small) / (
                B_big - B_small
            )
            S_noise = (gn_small - gn_big) / (1 / B_small - 1 / B_big)
    else:
        # If we're not splitting into microbatches, return logits & loss as is
        if params["model"] == "GPT":
            with mtf.utils.outside_all_rewrites():
                with tf.variable_scope('gpt2'):
                    logits, loss, loss_batch = gpt2.model(mtf_features, other_features, params, mesh,
                                                          variable_dtype=variable_dtype, context=None)
        else:
            raise Exception(
                f"'{params['model']}' is not a valid model - please select from [GPT]")

    # Auto layout generation
    if params["auto_layout"]:
        auto_layout(graph, mesh_shape, logits, loss)
    if params["auto_layout_and_mesh_shape"]:
        auto_layout_and_mesh_shape(graph, params["num_cores"], logits, loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # In TRAIN mode, get optimizer
        if params["num_microbatches"] > 1:
            # If we are splitting the batch into microbatches, var grads are created in the serialize_training_step fn
            # So we pass them in here
            _, update_ops, var_grads = get_optimizer(mesh, loss, params, variable_dtype=variable_dtype,
                                                     inp_var_grads=var_grads)
        else:
            # Otherwise, they are created in the get_optimizer fn, so we leave inp_var_grads blank
            _, update_ops, var_grads = get_optimizer(
                mesh, loss, params, variable_dtype=variable_dtype)
        # Log summaries to tensorboard
        mtf.scalar_summary("loss", loss)
        # Log gradients if in params
        if params["log_grads"] not in [None, False]:
            for g in var_grads:
                grad_norm = mtf.sqrt(mtf.reduce_sum(mtf.square(g)))
                mtf.scalar_summary("grads/norm" + g.name[:-2], grad_norm)
        if params['noise_scale']:
            mtf.scalar_summary("gn_small", gn_small)
            mtf.scalar_summary("gn_big", gn_big)
            mtf.scalar_summary("G_noise", G_noise)
            mtf.scalar_summary("S_noise", S_noise)
    else:
        # For now, we can only export fully-replicated tensors.
        # This has to be done before lowering or they will not be included in the graph
        mean_logits = mtf.reduce_mean(logits, reduced_dim=vocab_dim)
        max_logits = mtf.argmax(logits, vocab_dim)
        del logits
        fully_replicated_mean_logits = mtf.anonymize(mean_logits)
        fully_replicated_max_logits = mtf.anonymize(max_logits)
        fully_replicated_loss_batch = mtf.anonymize(loss_batch)

    # Gets & prints info about no. trainable vars in the model & dimension names
    get_graph_info(graph)

    # 'lowers' mtf tensors into a tf graph - this enables us to export results as tf tensors
    lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=True)
    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_loss = tf.cast(tf_loss, tf.float32)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Use our patched version until mtf updates theirs
        host_call = create_host_call(params['model_path'])
        mtf.utils.remove_summaries()

        # Creates train_op
        tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
        # Need to manually increment global_step
        tf_update_ops.append(tf.assign_add(global_step, 1))
        tf.logging.info(f"tf_update_ops: {tf_update_ops}")
        train_op = tf.group(tf_update_ops)
    else:
        tf_mean_logits = lowering.export_to_tf_tensor(
            fully_replicated_mean_logits)
        tf_max_logits = lowering.export_to_tf_tensor(
            fully_replicated_max_logits)
        tf_loss_batch = tf.to_float(
            lowering.export_to_tf_tensor(fully_replicated_loss_batch))

    with mtf.utils.outside_all_rewrites():
        # Copy master variables to slices. Must be called first.
        restore_hook = mtf.MtfRestoreHook(lowering)
        if mode == tf.estimator.ModeKeys.TRAIN:
            # Set up the checkpoint server and return the TPUEstimatorSpec
            saver_kwargs = {}
            if params["precision"] == "mixed_precision_load_bfloat16_once":
                saver_kwargs['builder'] = CastFromBFloat16SaverBuilder()
            saver = tf.train.Saver(
                tf.global_variables(),
                sharded=True,
                max_to_keep=3,
                keep_checkpoint_every_n_hours=2,
                defer_build=False,
                save_relative_paths=True,
                **saver_kwargs)
            tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
            saver_listener = mtf.MtfCheckpointSaverListener(lowering)
            saver_hook = tf.train.CheckpointSaverHook(
                params["model_path"],
                save_steps=params["steps_per_checkpoint"],
                saver=saver,
                listeners=[saver_listener])

            training_hooks = [restore_hook, saver_hook]

            return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.TRAIN,
                loss=tf_loss,
                host_call=host_call,
                train_op=train_op,
                training_hooks=training_hooks)

        elif mode == tf.estimator.ModeKeys.EVAL:
            # Evaluation metrics
            def _perplexity(loss):
                perplexity = tf.exp(loss)
                return tf.metrics.mean(perplexity)

            def _bits_per_byte(loss):
                bpb = loss * (0.29335 / math.log(2))
                return tf.metrics.mean(bpb)

            def _metric_fn(tf_mean_logits, tf_loss_batch):
                mean_logits = tf.metrics.mean(tf_mean_logits)
                loss = tf.reduce_mean(tf_loss_batch)
                perp = _perplexity(loss)
                bpb = _bits_per_byte(loss)
                return {"mean_logits": mean_logits, "perplexity": perp, "bits per byte": bpb}

            def _lambada_metric_fn(labels, tf_max_logits, tf_loss_batch):
                eos_token = params["eos_id"]
                answer_positions = tf.where(
                    tf.math.not_equal(labels, eos_token))

                correct_answers = tf.gather_nd(tf.math.equal(
                    tf_max_logits, labels), answer_positions)
                accuracy = tf.metrics.mean(
                    tf.cast(correct_answers, tf.float32))

                # I guess tf_loss_batch has z_loss and maybe other stuff added to it
                # so maybe this should be calculated separately in the future
                answer_loss = tf.gather_nd(tf_loss_batch, answer_positions)
                log_perplexity = tf.metrics.mean(answer_loss)

                return {"lambada_acc": accuracy, "lambada_log_ppl": log_perplexity}

            eval_task = params["eval_task"]
            if eval_task == "lambada":
                eval_metrics = (_lambada_metric_fn, [
                                labels, tf_max_logits, tf_loss_batch])
            else:
                eval_metrics = (_metric_fn, [tf_mean_logits, tf_loss_batch])

            return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.EVAL,
                evaluation_hooks=[restore_hook],
                loss=tf_loss,
                eval_metrics=eval_metrics)
