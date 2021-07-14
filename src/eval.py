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

"""Evaluate model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf


import src.margin_loss as margin_loss
import src.rf_margin as rf_margin

from datasets.meps import data_provider as meps
from datasets.meps import meps_config
from datasets.meps import meps_model

from datasets.german import data_provider as german
from datasets.german import german_config
from datasets.german import german_model

from datasets.adult import data_provider as adult
from datasets.adult import adult_config
from datasets.adult import adult_model

from sklearn.metrics import confusion_matrix
import pickle

# build dictionary that contains the tensors associating protected attribute to data input 
def get_fairness_groups_np(input_data, labels, feature_dict):      # numpy version 
    fairness_dict = {}

    for key, value in feature_dict.items():
        #protected_attr_vec = tf.zeros_like(labels)
        protected_attr_vec = np.zeros_like(labels)
        for val in feature_dict[key]:
            protected_attr_vec += np.cast[np.int32](input_data[:, val])*val
        
        fairness_dict[key] = protected_attr_vec 

    return fairness_dict

def get_fairness_groups(input_data, labels, feature_dict):          # tf version
    fairness_dict = {}

    for key, value in feature_dict.items():
        protected_attr_vec = tf.zeros_like(labels)

        for val in feature_dict[key]:
            protected_attr_vec += tf.cast(input_data[:, val], tf.int32)*val
        
        fairness_dict[key] = protected_attr_vec 

    return fairness_dict

def _eval_once(session_creator, ops_dict, summary_writer, merged_summary,
               global_step, num_examples, input_data, labels, unique_groups, fair_margin_over_epochs, FLAGS, config):
  """Runs evaluation on the full data and saves results.

  Args:
    session_creator: session creator.
    ops_dict: dictionary with operations to evaluate.
    summary_writer: summary writer.
    merged_summary: merged summaries.
    global_step: global step.
    num_examples: number of examples in the data.
  """
  num_batches = int(num_examples / float(FLAGS.batch_size))

  list_ops = []
  list_phs = []
  list_keys = []
  for key, value in ops_dict.items():
    if value[0] is not None and value[1] is not None:
      list_keys.append(key)
      list_ops.append(value[1])
      list_phs.append(value[0])

  with tf.train.MonitoredSession(session_creator=session_creator) as sess:
    list_results = []
    count = 0.
    total_correct = 0

    for _ in range(num_batches):
      res, top1, indv_logits, images, true_label, fair_value, robust_value = sess.run((list_ops, ops_dict["top1"][1], ops_dict['logits'][1], input_data, labels, fair_margin_over_epochs, ops_dict["losses/robustness_loss"][1]))
      number_correct = np.sum(top1)
      total_correct += number_correct
      count += FLAGS.batch_size
      list_results.append(res)

    overall_confusion = confusion_matrix(true_label, np.argmax(indv_logits,1))

    protected_attributes = list(config.feature_dict.keys())
    fairness_dict = get_fairness_groups_np(images, true_label, config.feature_dict) 

    accuracy = total_correct / count
    mean_results = np.mean(np.array(list_results), axis=0)
    feed_dict = {ph: v for ph, v in zip(list_phs, list(mean_results))}
    feed_dict[ops_dict["top1_accuracy"][0]] = accuracy
    summary, g_step = sess.run((merged_summary, global_step), feed_dict)
    tf.logging.info("\n\n\n\n\n\n\n\n"
                    "Global step: %d \n"
                    "top1 accuracy on %s set is %.6f"
                    "\n\n\n\n\n\n\n\n" % (g_step, FLAGS.subset, accuracy))
    tf.logging.info(num_examples)
    tf.logging.info(count)
    summary_writer.add_summary(summary, global_step=g_step)

    predictions = np.argmax(indv_logits,1)

    conf_matrix_per_group = { attr : {} for attr in protected_attributes}
    for attr in protected_attributes:
        conf_matrix_per_group[attr] = { ii : 0 for ii in unique_groups[attr]}

        # vectorized confusion matrix
        for ii in config.feature_dict[attr]:
            bool_vec = np.equal(fairness_dict[attr], ii)   # get boolean for a particular group
        
            group_predictions = predictions[bool_vec]   # valid preds for group
            true_label_per_group  = true_label[bool_vec]    # true labels for valid protected group
           
            conf_matrix_per_group[attr][ii] = confusion_matrix(true_label_per_group, group_predictions)   # confusion matrix per group

    TPR_dict, FPR_dict = {}, {}
    for attr in protected_attributes:
        TPR_dict[attr] = {ii : [] for ii in config.feature_dict[attr]} 
        FPR_dict[attr] = {ii : [] for ii in config.feature_dict[attr]}

        for ii in conf_matrix_per_group[attr]:
            FP = conf_matrix_per_group[attr][ii][0][1]
            FN = conf_matrix_per_group[attr][ii][1][0]
            TP = conf_matrix_per_group[attr][ii][1][1]
            TN = conf_matrix_per_group[attr][ii][0][0]

            TPR_dict[attr][ii].append(TP/(TP+FN))
            FPR_dict[attr][ii].append(FP/(FP+TN))

    # Save out variables
    results = {}
    results['fairness'] = fair_value
    results['robustness'] = robust_value
    results['acc'] = accuracy 
    results['confusion_matrix'] = conf_matrix_per_group 
    results['TRP'] = TPR_dict 
    results['FPR'] = FPR_dict
    print(results)
    with open(f"save/{FLAGS.experiment_type}_test_fair_{FLAGS.fair_loss_wt}_robust_{FLAGS.margin_loss_wt}_lr_{FLAGS.init_lr}_iter_{FLAGS.iteration}.pkl", 'wb') as handle:
          pickle.dump(results, handle)

    return(accuracy)


def evaluate(FLAGS):
  """Evaluating function."""
  data_dir = "datasets/"+FLAGS.experiment_type+"/"
  g = tf.Graph()
  ops_dict = {}
  with g.as_default():
    # Data set.
    if FLAGS.experiment_type == 'meps':
        config = meps_config.ConfigDict()
        dataset = meps.MEPS(
            data_dir=data_dir,
            subset="test",
            batch_size=FLAGS.batch_size,
            is_training=False)
        model = meps_model.MepsNetwork(config)
        layers_names = [
          "conv_layer%d" % i
          for i in range(len(config.filter_sizes_conv_layers))
        ]
    if FLAGS.experiment_type == 'german':
        config = german_config.ConfigDict()
        dataset = german.GermanCredit(
            data_dir=data_dir,
            subset="test",
            batch_size=FLAGS.batch_size,
            is_training=False)
        model = german_model.GermanNetwork(config)
        layers_names = [
          "conv_layer%d" % i
          for i in range(len(config.filter_sizes_conv_layers))
        ]
    if FLAGS.experiment_type == 'adult':
        config = adult_config.ConfigDict()
        dataset = adult.UCI_Adult(
            data_dir=data_dir,
            subset="test",
            batch_size=FLAGS.batch_size,
            is_training=False)
        model = adult_model.AdultNetwork(config)
        layers_names = [
          "conv_layer%d" % i
          for i in range(len(config.filter_sizes_conv_layers))
        ]


    images, labels, num_examples, num_classes = (dataset.images, dataset.labels,
                                                 dataset.num_examples,
                                                 dataset.num_classes)

    logits, endpoints = model(images, is_training=False)
    layers_list = [images] + [endpoints[name] for name in layers_names]

    top1_op = tf.nn.in_top_k(logits, labels, 1)

    top1_op = tf.cast(top1_op, dtype=tf.float32)
    ops_dict["top1"] = (None, top1_op)
    accuracy_ph = tf.placeholder(tf.float32, None)
    ops_dict["top1_accuracy"] = (accuracy_ph, None)
    ops_dict["logits"] = (None, logits)
    tf.summary.scalar("top1_accuracy", accuracy_ph)

    with tf.name_scope("optimizer"):
      global_step = tf.train.get_or_create_global_step()

    # Define losses.
    l2_loss_wt = config.l2_loss_wt
    xent_loss_wt = config.xent_loss_wt
    margin_loss_wt = FLAGS.margin_loss_wt
    fair_loss_wt = FLAGS.fair_loss_wt
    gamma = config.gamma
    alpha = config.alpha
    top_k = config.top_k
    dist_norm = config.dist_norm
    robust_margin_over_epochs, fair_margin_over_epochs = [], [] 
  
    with tf.name_scope("losses"):
      xent_loss = tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=logits, labels=labels))

    unique_groups, unfav_outcome, protected_idx_vec = None, None, None
    
    # get protected feature attributes
    protected_attributes = list(config.feature_dict.keys())

    fairness_dict = get_fairness_groups(images, labels, config.feature_dict) 

    fair_margin_over_epochs = { attr  : [] for attr in protected_attributes }
    unique_groups = { attr  : [] for attr in protected_attributes }
   
    for attr in protected_attributes:
        # setup protected features dictionary 
        unique_groups[attr] = config.feature_dict[attr]

    # calculate burden and robustness    
    robustness_margin, burden_margin_dict = rf_margin.rf_margin(
        logits=logits,
        one_hot_labels=tf.one_hot(labels, num_classes),
        top_k=top_k,
        unfav_outcome = config.unfavorable_label,
        fairness_dict = fairness_dict,
        feature_dict = config.feature_dict
        )

    # large margin distance
    '''
    margin = margin_loss.large_margin(
        logits=logits,
        one_hot_labels=tf.one_hot(labels, num_classes),
        layers_list=layers_list,
        gamma=gamma,
        alpha_factor=alpha,
        top_k=top_k,
        dist_norm=dist_norm,
        epsilon=1e-6,
        layers_weights=[
            np.prod(layer.get_shape().as_list()[1:])
            for layer in layers_list] if np.isinf(dist_norm) else None
        )  
    robustness_margin = 1/margin
    '''
              
    l2_loss = 0.
    for v in tf.trainable_variables():
      tf.logging.info(v)
      l2_loss += tf.nn.l2_loss(v)

    total_loss = 0.
    total_loss += xent_loss_wt * xent_loss
    total_loss += margin_loss_wt * (1/robustness_margin)
    total_loss += l2_loss_wt * l2_loss  

    for key in burden_margin_dict:
        total_loss += fair_loss_wt * burden_margin_dict[key] 

        fair_margin_over_epochs[key].append(burden_margin_dict[key])

    robust_margin_over_epochs.append(robustness_margin)

    xent_loss_ph = tf.placeholder(tf.float32, None)
    robustness_loss_ph = tf.placeholder(tf.float32, None)
    l2_loss_ph = tf.placeholder(tf.float32, None)
    fair_loss_ph = tf.placeholder(tf.float32, None)
    total_loss_ph = tf.placeholder(tf.float32, None)

    tf.summary.scalar("xent_loss", xent_loss_ph)
    tf.summary.scalar("robustness_loss", robustness_loss_ph)
    tf.summary.scalar("l2_loss", l2_loss_ph)
    tf.summary.scalar("total_loss", total_loss_ph)

    ops_dict["losses/xent_loss"] = (xent_loss_ph, xent_loss)
    ops_dict["losses/robustness_loss"] = (robustness_loss_ph, robustness_margin)
    ops_dict["losses/l2_loss"] = (l2_loss_ph, l2_loss)
    ops_dict["losses/total_loss"] = (total_loss_ph, total_loss)

    # Prepare evaluation session.
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                           tf.get_default_graph())
    vars_to_save = tf.global_variables()
    saver = tf.train.Saver(var_list=vars_to_save)
    scaffold = tf.train.Scaffold(saver=saver)
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold, checkpoint_dir=FLAGS.checkpoint_dir)

    outcomes = _eval_once(session_creator, ops_dict, summary_writer, merged_summary,
                 global_step, num_examples, images, labels, unique_groups, fair_margin_over_epochs, FLAGS, config)
    return outcomes

def main(argv):
  del argv  # Unused.
  outcomes = evaluate()


if __name__ == "__main__":
  app.run(main)
