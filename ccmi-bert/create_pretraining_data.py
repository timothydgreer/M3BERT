# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from copy import deepcopy
import ntpath
import os
import time

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("input_dir", None,
                    "Input directory (all .npy files in this directory will be considered input files)")

flags.DEFINE_integer("max_files_per_tfrecord", 0,
                     "Max number of numpy files per output TFrecord (limit this to prevent overloading system resources)")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("maxes_file", None,
                    "Maxes file (provide 1D vector of maxes for each feature if normalization desired.)")

flags.DEFINE_string("mins_file", None,
                    "Mins file (provide 1D vector of mins for each feature if normalization desired.)")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("num_features", 40, "Number of audio features in every frame.")

# Contiguous Frame Masking (CFM) & Contiguous Channel Masking (CCM) parameters: from https://arxiv.org/abs/2008.00781
flags.DEFINE_float("cfm_p", 0.2, "CFM p value.")
flags.DEFINE_integer("cfm_lmin", 2, "CFM l_min value.")
flags.DEFINE_integer("cfm_lmax", 6, "CFM l_max value.")
flags.DEFINE_float("cfm_budget", 0.15, "CFM Budget; that is, what portion of total frames are masked.")
flags.DEFINE_float("cfm_zero_out_chance", 0.7, "CFM zero-out policy chance.")
flags.DEFINE_float("cfm_replace_chance", 0.2, "CFM replace policy chance.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 1,
    "Number of times to duplicate the input data (with different masks).")


class TrainingInstance(object):
  """A single training instance (musical cue frames)."""

  def __init__(self, frames, original_frames, pretrain_mask):
    self.frames = frames
    self.original_frames = original_frames
    self.pretrain_mask = pretrain_mask

  def __str__(self):
    s = ""
    s += "frames shape: %s\n" % self.frames.shape
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, max_seq_length, output_files, sess):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  num_features = FLAGS.num_features

  writer_index = 0

  total_written = 0
  for inst_index in tqdm(range(instances.shape[0])):
    instance = instances[inst_index]
    input_frames = np.zeros( (max_seq_length, num_features), dtype=np.float32 )
    input_mask = np.zeros((max_seq_length, num_features), dtype=np.int64 )
    original_frames = np.zeros((max_seq_length, num_features), dtype=np.float32)
    pretrain_mask = np.zeros((max_seq_length, num_features), dtype=np.int64)
    shp = instance.frames.shape
    input_frames[0:shp[0], :] = instance.frames
    input_mask[0:shp[0], :] = np.ones( shp, dtype=np.int64 )
    original_frames[0:shp[0], :] = instance.original_frames
    pretrain_mask[0:shp[0], :] = instance.pretrain_mask

    features = collections.OrderedDict()
    features["input_frames"] = create_float_feature(input_frames, flatten=True)
    features["input_mask"] = create_int_feature(input_mask, flatten=True)
    features["original_frames"] = create_float_feature(original_frames, flatten=True)
    features["pretrain_mask"] = create_int_feature(pretrain_mask, flatten=True)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

  for writer in writers:
    writer.close()

  print("Wrote {} total instances".format(total_written))


def create_int_feature(values, flatten=False):
    if(flatten):
        values = values.flatten()
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values, flatten=False):
    if (flatten):
        values = values.flatten()
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature

# note: SLOW
def create_serialized_feature(values, sess):
    serial = tf.serialize_tensor(values)
    if isinstance(serial, type(tf.constant(0))):
        with sess.as_default():
            serial = serial.eval()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serial]))

def create_training_instances(input_files, max_seq_length, dupe_factor):
  """Create `TrainingInstance`s from numpy files containing low-level audio features."""
  all_arrs = []

  num_errors = 0
  for input_file in input_files:
    try:
      arr = np.load(input_file, allow_pickle=True)
    except Exception as e:
      print("{} on file {}: {}".format(type(e), input_file, e))
      num_errors+=1
      continue
    # normalize if arguments provided
    if FLAGS.maxes_file or FLAGS.mins_file:
      if FLAGS.maxes_file and FLAGS.mins_file:
        maxes = np.load(FLAGS.maxes_file)
        mins = np.load(FLAGS.mins_file)
        assert maxes.shape[0] == FLAGS.num_features, "Maxes length {} must match num_features {}".format(
          maxes.shape[0],FLAGS.num_features)
        assert mins.shape[0] == FLAGS.num_features, "Mins length {} must match num_features {}".format(
          mins.shape[0], FLAGS.num_features)
        arr = (arr - mins)/(maxes-mins)
      else:
        raise ValueError("Either maxes_file or mins_file was provided, but not both! Need both to normalize")

    # if it's a list or numpy array of numpy arrays, verify shapes
    if not (len(arr.shape) == 1 and len(arr[0].shape) == 2):
      # expand dims from 2D to 3D if necessary
      if len(arr.shape) == 2:
        arr = np.expand_dims(arr, 0)
      # if features are in dimension 1, swap to dimension 2
      if arr.shape[1] == FLAGS.num_features:
        arr = np.swapaxes(arr, 1, 2)
      elif arr.shape[2] != FLAGS.num_features:
        raise Exception("{}: Input array of shape {} does not have num_features = {}".format(
          input_file, arr.shape, FLAGS.num_features))
    all_arrs.append(arr)

  print("{} valid files ({} errors)".format(len(all_arrs), num_errors))

  instances = []
  for _ in range(dupe_factor):
    for arr in all_arrs:
      instances.extend(create_instances_from_arr(arr, max_seq_length))

  instances = np.asarray(instances)
  np.random.shuffle(instances)
  return instances


def create_instances_from_arr(
    arr, max_seq_length
):
  """Creates `TrainingInstance`s for a single numpy arr (NUM_SEQUENCES, NUM_FRAMES, NUM_FEATURES)."""
  instances = []
  for i in range(arr.shape[0]):
    # take the middle max_seq_len if oversize
    if arr[i].shape[0] > max_seq_length:
      s = (arr[i].shape[0] - max_seq_length)//2
      X = arr[i][s:s+max_seq_length]
    else:
      print("Fixing it at the source")
      print(arr[i].shape)
      X = np.zeros((max_seq_length, FLAGS.num_features))
      X[:arr[i].shape[0],:arr[i].shape[1]] = arr
      print(X.shape)
      X.reshape((max_seq_length,FLAGS.num_features))
    (masked_X, cfm_mask) = cfm(X)
    (masked_X, ccm_mask) = ccm(masked_X)
    pretrain_mask = np.logical_or(cfm_mask, ccm_mask, dtype=np.int32)
    instance = TrainingInstance(
      frames=masked_X,
      original_frames = X,
      pretrain_mask= pretrain_mask
    )
    instances.append(instance)
  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

# Contiguous Frame Masking (CFM) pre-training objective; as defined in https://arxiv.org/abs/2008.00781
# Input: Numpy Array X with shape [sequence_length, num_features]
# Returns: modified copy of array and corresponding array of which positions were masked
def cfm(orig_X):
  X = deepcopy(orig_X)
  mask = np.zeros(X.shape, dtype=np.int32)
  frames_altered = 0
  cfm_budget = FLAGS.cfm_budget
  num_frames = FLAGS.max_seq_length
  num_features = FLAGS.num_features
  lmin = FLAGS.cfm_lmin
  lmax = FLAGS.cfm_lmax
  p = FLAGS.cfm_p
  zero_out_chance = FLAGS.cfm_zero_out_chance
  replace_chance = FLAGS.cfm_replace_chance
  while frames_altered < cfm_budget * num_frames:  # keep going until budget allocated
    # Sample span length
    l = np.random.geometric(p)
    l = lmin if l < lmin else lmax if l > lmax else l  # clamp to min and max vals

    # Sample starting frame of span
    s = np.random.randint(num_frames)
    l = (num_frames - s) if s + l > num_frames else l

    # Select which policy (of 3) to apply, and apply it
    rand = np.random.rand()
    if X.shape[0] < num_frames:
      #print(num_frames)
      #print(num_features)
      #print(s)
      #print(rand_start)
      #print(X.shape)
      #print(X[s:s+l].shape)
      print("Time to pad a short musical song...")
      print(X.shape)
      X.resize((num_frames,num_features))
      print(X.shape)
      print(X[-1,-1])
      return (X,mask)
    if rand < zero_out_chance:
      # zero out frames
      X[s:s + l] = np.zeros((l, num_features))
    elif rand < zero_out_chance + replace_chance:
      # replace frames with random frames
      rand_start = np.random.randint(num_frames - l)
      X[s:s+l] = X[rand_start:rand_start+l]
      frames_altered += l
      mask[s:s + l] = np.ones((l, num_features))
  return (X, mask)

# Contiguous Channel Masking (CCM) pre-training objective; as defined in https://arxiv.org/abs/2008.00781
# Input: Numpy Array X with shape [sequence_length, num_features]
# Returns: modified copy of array and corresponding array of which positions were masked
def ccm(orig_X):
  X = deepcopy(orig_X)
  mask = np.zeros(X.shape, dtype=np.int32)
  num_frames = FLAGS.max_seq_length
  num_features = FLAGS.num_features

  # sample n
  n = np.random.randint(num_features)
  # NOTE: if features eventually becomes more than just mel bins, will need to alter this to sample mel-bins only

  # sample starting index
  s = np.random.randint(num_features - n)

  # mask to 0
  X[:, s:s + n] = np.zeros((num_frames, n))
  mask[:, s:s + n] = np.ones((num_frames, n))

  return (X, mask)


def main(_):
  start = time.time()
  sess = tf.Session()
  input_files = []

  if FLAGS.input_dir:
    if FLAGS.input_file:
      print("Warning: both input_dir and input_file parameters were specified. Defaulting to using input_dir")
    for f in os.listdir(FLAGS.input_dir):
      if f.endswith(".npy"):
        input_files.append(os.path.join(FLAGS.input_dir, f))

  else:
    for input_pattern in FLAGS.input_file.split(","):
      input_files.extend(tf.gfile.Glob(input_pattern))

  # remove maxes.npy and mins.npy if they're in there
  to_remove = []
  for i in range(len(input_files)):
    file = ntpath.basename(input_files[i])
    if file == "maxes.npy" or file == "mins.npy":
      to_remove.append(input_files[i])
  input_files = [x for x in input_files if x not in to_remove]

  print("*** Reading from input files ***")
  if len(input_files) < 20:
    for input_file in input_files:
      print("  ", input_file)
  else:
    print("{} files read.".format(len(input_files)))

  np.random.seed(FLAGS.random_seed)

  # Split into multiple batches if number of files exceeds max_files_per_tfrecord
  input_batches = [] # each batch is a list of files
  output_files = [] # each batch has an output filename
  if FLAGS.max_files_per_tfrecord and len(input_files) > FLAGS.max_files_per_tfrecord:
    j = -1
    for i in range(len(input_files)):
      if i % FLAGS.max_files_per_tfrecord == 0:
        j+=1
        input_batches.append([])
        output_files.append(FLAGS.output_file[:-9]+"_batch"+str(j)+".tfrecord")
      input_batches[j].append(input_files[i])

  for b in range(len(input_batches)):
    if os.path.exists(output_files[b]):
      print("{} already exists. Skipping...".format(output_files[b]))
      continue

    instances = create_training_instances(
      input_batches[b], FLAGS.max_seq_length, FLAGS.dupe_factor
    )

    # TODO: restore this multiple-output-file functionality to work with the max_files_per_tfrecord functionality I added
    # output_files = FLAGS.output_file.split(",")
    # print("*** Writing to output files ***")
    # for output_file in output_files:
    #   print("  ", output_file)
    #   if not os.path.exists(os.path.dirname(output_file)):
    #     print("Creating output dir {}".format(os.path.dirname(output_file)))
    #     os.makedirs(os.path.dirname(output_file))

    if not os.path.exists(os.path.dirname(output_files[b])):
      print("Creating output dir {}".format(os.path.dirname(output_files[b])))
      os.makedirs(os.path.dirname(output_files[b]))

    write_instance_to_example_files(instances, FLAGS.max_seq_length, [output_files[b]], sess)
  print("Total time elapsed: {} seconds".format(time.time() - start))


if __name__ == "__main__":
  flags.mark_flag_as_required("output_file")
  tf.app.run()
