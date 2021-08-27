# Ben Ma
# 12-17-20
# Python 3.x, Tensorflow 1.x

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("num_features", 40, "Number of audio features in every frame.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer("predict_batch_size", None, "Number of examples to output for visualization.")

def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    print("*** Features ***")
    for name in sorted(features.keys()):
      print("  name = {}, shape = {}".format(name, features[name].shape))

    input_frames = features["input_frames"]
    input_mask = features["input_mask"]
    original_frames = features["original_frames"]
    pretrain_mask = features["pretrain_mask"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.MusicoderModel(
        config=bert_config,
        is_training=is_training,
        input_frames=input_frames,
        input_mask=input_mask)

    (pretrain_loss, pretrain_example_loss, reconstructed) = get_pretrain_output(
        bert_config, model.get_sequence_output(), original_frames, pretrain_mask
    )

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    print("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      print("  name = {}, shape = {}{}".format(var.name, var.shape,
                      init_string))

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          pretrain_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      # hook to output training loss every n steps
      logging_hook = tf.train.LoggingTensorHook({"loss": pretrain_loss}, every_n_iter=1000)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=pretrain_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(pretrain_example_loss):
        """Computes the loss of the model."""
        pretrain_mean_loss = tf.metrics.mean(values=pretrain_example_loss)
        return {
            "masked_ccm_cfm_loss": pretrain_mean_loss,
        }

      eval_metrics = (metric_fn, [
          pretrain_example_loss
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=pretrain_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'input_frames': input_frames,
            'original_frames': original_frames,
            'reconstructed': reconstructed
        }
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions=predictions
        )
    else:
      raise ValueError("Only TRAIN, EVAL, and PREDICT modes are supported: {}".format(mode))

    return output_spec

  return model_fn

#TODO: change loss function so it reflects original musicoder!
def get_pretrain_output(bert_config, input_tensor, original_frames, pretrain_mask):
  """Get loss and log probs for the reconstruction objective."""
  # Also returns the reconstructed frame
  # Input tensor should be [batch_size, num_frames, hidden_size]
  # (in original MusiCoder paper, and BERT, hidden_size is 768)
  print("Finding example losses")

  with tf.variable_scope("cls/predictions"):
    # "In the pre-training process, a reconstruction module, which
    # consists of two layers of feed-forward network with GeLU activation [16] and
    # layer-normalization [1], is appended to predict the masked inputs."
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      x = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      x = modeling.layer_norm(x)
      x = tf.layers.dense(
          x,
          units=FLAGS.num_features, # reconstructed frame width
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      reconstructed = modeling.layer_norm(x) # Shape should be: [batch_size, seq_length, num_features]

  # could also use pretrain_mask to only penalize reconstruction error on values that were originally masked,
  # rather than all values.
  # TODO: get reconstructed at the mask and the original frames. See the lines above that Ben authored!
  # Pretrain mask is already passed so it is accessible.
  example_loss = tf.losses.huber_loss(reconstructed, original_frames) # automatically divided by batch size
  loss = example_loss * FLAGS.eval_batch_size

  return (loss, example_loss, reconstructed)

def input_fn_builder(input_files,
                     max_seq_length,
                     num_features,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_frames":
            tf.FixedLenFeature([max_seq_length, num_features], tf.float32),
        "input_mask":
            tf.FixedLenFeature([max_seq_length, num_features], tf.int64),
        "original_frames":
            tf.FixedLenFeature([max_seq_length, num_features], tf.float32),
        "pretrain_mask":
            tf.FixedLenFeature([max_seq_length, num_features], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
    num_features = FLAGS.num_features
    max_seq_length = FLAGS.max_seq_length

    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example

def main(_):
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    tf.logging.set_verbosity(tf.logging.INFO)

    print("Hello pretraining!")
    print("use_tpu:", FLAGS.use_tpu)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    print("*** Input Files ***")
    for input_file in input_files:
        print("  {}".format(input_file))

    run_config = tf.contrib.tpu.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps
    )

    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=FLAGS.num_warmup_steps,
        use_tpu=FLAGS.use_tpu
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size
    )

    if FLAGS.do_train:
        print("***** Running pre-training *****")
        print("  Batch size = {}".format(FLAGS.train_batch_size))
        train_input_fn = input_fn_builder(
            max_seq_length=FLAGS.max_seq_length,
            num_features=FLAGS.num_features,
            input_files=input_files,
            is_training=True)
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

    if FLAGS.do_eval:
        print("***** Running evaluation *****")
        print("  Batch size = {}".format(FLAGS.eval_batch_size))

        eval_input_fn = input_fn_builder(
            max_seq_length=FLAGS.max_seq_length,
            num_features=FLAGS.num_features,
            input_files=input_files,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            print("***** Eval results *****")
            for key in sorted(result.keys()):
                print("  {} = {}".format(key, str(result[key])))
                writer.write("{} = {}\n".format(key, str(result[key])))

        # output examples if predict_batch_size is set
        output_stem = FLAGS.output_dir+"example"
        if FLAGS.predict_batch_size is not None:
            predictions = estimator.predict(eval_input_fn)
            for i in range(FLAGS.predict_batch_size):
                p = next(predictions)
                np.save(output_stem+str(i)+"_input", p["input_frames"])
                np.save(output_stem + str(i) + "_original", p["original_frames"])
                np.save(output_stem + str(i) + "_reconstructed", p["reconstructed"])

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
