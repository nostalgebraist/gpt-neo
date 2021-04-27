from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training.tracking import graph_view
from tensorflow.python.training.tracking import util as trackable_util

from tf.train import Scaffold


class ScaffoldNonFinalizing(Scaffold):
  def finalize(self):
    """Creates operations if needed and finalizes the graph."""
    if self._init_op is None:

      def default_init_op():
        return control_flow_ops.group(
            variables.global_variables_initializer(),
            resources.initialize_resources(resources.shared_resources()))

      self._init_op = Scaffold.get_or_default('init_op', ops.GraphKeys.INIT_OP,
                                              default_init_op)
    if self._ready_op is None:

      def default_ready_op():
        return array_ops.concat([
            variables.report_uninitialized_variables(),
            resources.report_uninitialized_resources()
        ], 0)

      self._ready_op = Scaffold.get_or_default('ready_op',
                                               ops.GraphKeys.READY_OP,
                                               default_ready_op)
    if self._ready_for_local_init_op is None:

      def default_ready_for_local_init_op():
        return array_ops.concat([
            variables.report_uninitialized_variables(
                variables.global_variables()),
            resources.report_uninitialized_resources(
                resources.shared_resources())
        ], 0)

      self._ready_for_local_init_op = Scaffold.get_or_default(
          'ready_for_local_init_op', ops.GraphKeys.READY_FOR_LOCAL_INIT_OP,
          default_ready_for_local_init_op)
    if self._local_init_op is None:
      self._local_init_op = Scaffold.get_or_default(
          'local_init_op', ops.GraphKeys.LOCAL_INIT_OP,
          Scaffold.default_local_init_op)
    if self._summary_op is None:
      self._summary_op = Scaffold.get_or_default('summary_op',
                                                 ops.GraphKeys.SUMMARY_OP,
                                                 summary.merge_all)
    # pylint: disable=g-long-lambda
    if self._saver is None:
      self._saver = training_saver._get_saver_or_default()  # pylint: disable=protected-access
    # pylint: enable=g-long-lambda
    if isinstance(self._saver, trackable_util.Checkpoint):
      self._saver = training_saver.Saver(
          var_list=graph_view.ObjectGraphView(
              self._saver).frozen_saveable_objects(),
          sharded=True)
    else:
      self._saver.build()

    logging.info('Graph was NOT finalized ;)')
    return self
