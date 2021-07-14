# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import os
import numpy as np
import tensorflow as tf
import pandas as pd


def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      label_name='outcome',
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
  return dataset

def get_image_label_from_record(image_record, label_record):

  """Decodes the image and label information from one data record."""
  # Convert from tf.string to tf.uint8.
  image = tf.decode_raw(image_record, tf.uint8)
  # Convert from tf.uint8 to tf.float32.
  image = tf.cast(image, tf.float32)

  # Reshape image to correct shape.
  image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
  # Normalize from [0, 255] to [0.0, 1.0]
  image /= 255.0

  # Convert from tf.string to tf.uint8.
  label = tf.decode_raw(label_record, tf.uint8)
  # Convert from tf.uint8 to tf.int32.
  label = tf.to_float32(label)
  # Reshape label to correct shape.
  label = tf.reshape(label, [])  # label is a scalar
  return image, label

def get_image_label_from_record_tabular(image_record, label_record):

  """Decodes the image and label information from one data record."""
  # Convert from tf.string to tf.uint8.
  record_defaults = []  
  for idx in range(64):   # german one hot processed set has 64 columns
      record_defaults.append([0.])
  image = tf.decode_csv(image_record, record_defaults=record_defaults)
  # Convert from tf.uint8 to tf.float32.
  image = tf.cast(image, tf.float32)

  # Convert from tf.string to tf.uint8.
  label = tf.decode_csv(label_record, record_defaults=[[0]])
  # Convert from tf.uint8 to tf.int32.
  label = tf.to_int32(label)
  # Reshape label to correct shape.
  label = tf.reshape(label, [])  # label is a scalar

  return image, label


def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value))

def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_freatures = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_freatures]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels


class GermanCredit(object):
  r"""Create Dataset from MNIST files.

  First download MNIST data.

    Data extraction flow.

    Extract data records.
          |
    Shuffle dataset (if is_training)
          |
    Repeat dataset after finishing all examples (if is_training)
          |
    Map parser over dataset
          |
    Batch by batch size
          |
    Prefetch dataset
          |
    Create one shot iterator
  Attributes:
    images: 4D Tensors with images of a batch.
    labels: 1D Tensor with labels of a batch.
    num_examples: (integer) Number of examples in the data.
    subset: Data subset 'train_valid', 'train', 'valid', or, 'test'.
    num_classes: (integer) Number of classes in the data.
  """

  def __init__(self, data_dir, subset, batch_size, is_training=False):

    if is_training:
        images_file = os.path.join(data_dir, "german_train_onehot.csv")
        labels_file = os.path.join(data_dir, "german_train_label.csv")
    else:
        images_file = os.path.join(data_dir, "german_test_onehot.csv")
        labels_file = os.path.join(data_dir, "german_test_label.csv")

    
    df = pd.read_csv(images_file)
    X = df.to_numpy()
    num_examples = X.shape[0]
    
    # Construct fixed length record dataset.
    dataset_images = tf.data.TextLineDataset(images_file).skip(1)
    dataset_labels = tf.data.TextLineDataset(labels_file).skip(1)

    dataset = tf.data.Dataset.zip((dataset_images, dataset_labels))

    dataset = dataset.repeat(-1 if is_training else 1)
    dataset = dataset.map(get_image_label_from_record_tabular, num_parallel_calls=32)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=max(1, int(100 / batch_size)))
    
    iterator = dataset.make_one_shot_iterator()
    self.images, labels = iterator.get_next()
    self.labels = tf.squeeze(labels)
    self.subset = subset
    self.num_classes = 2
    self.num_examples = num_examples

