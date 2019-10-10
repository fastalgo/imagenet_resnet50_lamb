from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
                     poly_power, start_warmup_step, weight_decay_input, beta1_input, beta2_input, eps_input):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  global_step_int = tf.cast(global_step, tf.int32)
  step = tf.cast(global_step_int, tf.float32)
  warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)
  warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)
  warmup_sched = lambda: step * learning_rate / warmup_steps_float
  total_steps_int = tf.constant(num_train_steps, dtype=tf.int32)
  total_steps_float = tf.cast(total_steps_int, tf.float32)
  decay_sched = lambda: (total_steps_float - step) * learning_rate / (total_steps_float - warmup_steps_float)
  local_lr = tf.cond(tf.less_equal(step, warmup_steps_float), warmup_sched, decay_sched)
  tf.logging.info("++++++ warmup starts at step " + str(start_warmup_step) + ", for " + str(num_warmup_steps) + " steps ++++++")
  optimizer = LAMBOptimizer(
      learning_rate=local_lr,
      weight_decay_rate=weight_decay_input,
      beta_1=beta1_input,
      beta_2=beta2_input,
      epsilon=eps_input,
      exclude_from_weight_decay=["batch_normalization", "LayerNorm", "layer_norm"])
      # exclude_from_weight_decay=["batch_normalization", "LayerNorm", "layer_norm", "bias"])
      # exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
      #beta_1=0.9,
      #beta_2=0.999,
      #epsilon=1e-6,

  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `LAMBOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


class LAMBOptimizer(tf.train.Optimizer):
  """LAMB (Layer-wise Adaptive Moments optimizer for Batch training)."""
  # A new optimizer that includes correct L2 weight decay, adaptive
  # element-wise updating, and layer-wise justification. The LAMB optimizer
  # was proposed by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song,
  # James Demmel, and Cho-Jui Hsieh in a paper titled as Reducing BERT
  # Pre-Training Time from 3 Days to 76 Minutes (arxiv.org/abs/1904.00962)

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="LAMBOptimizer"):
    """Constructs a LAMBOptimizer."""
    super(LAMBOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      ratio = 1.0
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param
        w_norm = linalg_ops.norm(param, ord=2)
        g_norm = linalg_ops.norm(update, ord=2)
        ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
            math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

      tf.logging.info("*********** I'm using LAMB correction ***********")
      update_with_lr = ratio * self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
