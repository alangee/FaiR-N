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

"""Train model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle

import warnings
warnings.filterwarnings("ignore")

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

from absl import flags



# build dictionary that contains the tensors associating protected attribute to data input 
def get_fairness_groups(input_data, labels, feature_dict):
    fairness_dict = {}

    for key, value in feature_dict.items():
        protected_attr_vec = tf.zeros_like(labels)
        for val in feature_dict[key]:
            protected_attr_vec += tf.cast(input_data[:, val], tf.int32)*val
        
        fairness_dict[key] = protected_attr_vec 

    return fairness_dict


def train(FLAGS):
  """Training function."""
  is_chief = (FLAGS.task == 0)
  data_dir = "datasets/"+FLAGS.experiment_type+"/"
  g = tf.Graph()
  with g.as_default():
    with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.ps_tasks)):
      if FLAGS.experiment_type == "meps":
        config = meps_config.ConfigDict()
        dataset = meps.MEPS(
            data_dir=data_dir,
            subset="train",
            batch_size=FLAGS.batch_size,
            is_training=True)
        model = meps_model.MepsNetwork(config)
        layers_names = [
            "conv_layer%d" % i
            for i in range(len(config.filter_sizes_conv_layers))
        ]
      if FLAGS.experiment_type == 'german':
        config = german_config.ConfigDict()
        dataset = german.GermanCredit(
            data_dir=data_dir,
            subset="train",
            batch_size=FLAGS.batch_size,
            is_training=True)
        model = german_model.GermanNetwork(config)
        layers_names = [
            "conv_layer%d" % i
            for i in range(len(config.filter_sizes_conv_layers))
        ]
      if FLAGS.experiment_type == 'adult':
        config = adult_config.ConfigDict()
        dataset = adult.UCI_Adult(
            data_dir=data_dir,
            subset="train",
            batch_size=FLAGS.batch_size,
            is_training=True)
        model = adult_model.AdultNetwork(config)
        layers_names = [
            "conv_layer%d" % i
            for i in range(len(config.filter_sizes_conv_layers))
        ]


      images, labels, num_examples, num_classes = (dataset.images,
                                                   dataset.labels,
                                                   dataset.num_examples,
                                                   dataset.num_classes)

      # Build model.
      logits, endpoints = model(images, is_training=True)
      layers_list = [images] + [endpoints[name] for name in layers_names]

      # Define losses.
      l2_loss_wt = config.l2_loss_wt
      xent_loss_wt = config.xent_loss_wt
      margin_loss_wt = FLAGS.margin_loss_wt
      fair_loss_wt = FLAGS.fair_loss_wt
      gamma = config.gamma
      alpha = config.alpha
      top_k = config.top_k
      dist_norm = config.dist_norm
      robust_margin_over_epochs = [] 
  
      # get protected feature attributes
      protected_attributes = list(config.feature_dict.keys())
      fairness_dict = get_fairness_groups(images, labels, config.feature_dict) 

      fair_margin_over_epochs = { attr  : [] for attr in protected_attributes }
      fair_metric_epoch = { attr  : {} for attr in protected_attributes }
      unique_groups = { attr  : [] for attr in protected_attributes }
      TPR_dict = { attr  : [] for attr in protected_attributes }
      FPR_dict = { attr  : [] for attr in protected_attributes }
   
 
      for attr in protected_attributes:
        # setup protected features dictionary 
        unique_groups[attr] = config.feature_dict[attr]
        fair_metric_epoch[attr] = { ii : 0 for ii in range(FLAGS.num_epochs+1) }
        # confusion_metrics
        TPR_dict[attr] = {ii : [] for ii in unique_groups[attr]}
        FPR_dict[attr] = {ii : [] for ii in unique_groups[attr]}

      with tf.name_scope("losses"):
        xent_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
    
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
        tf.summary.scalar("xent_loss", xent_loss)
        tf.summary.scalar("margin_loss", robustness_margin)
        tf.summary.scalar("l2_loss", l2_loss)
        tf.summary.scalar("total_loss", total_loss)

      # Build optimizer.
      init_lr = FLAGS.init_lr
      #init_lr = config.init_lr
      with tf.name_scope("optimizer"):
        global_step = tf.train.get_or_create_global_step()
        if FLAGS.num_replicas > 1:
          num_batches_per_epoch = num_examples // (
              FLAGS.batch_size * FLAGS.num_replicas)
        else:
          num_batches_per_epoch = num_examples // FLAGS.batch_size
        max_iters = num_batches_per_epoch * FLAGS.num_epochs

        lr = tf.train.exponential_decay(init_lr,
                                        global_step,
                                        FLAGS.decay_steps,
                                        FLAGS.decay_rate,
                                        staircase=True,
                                        name="lr_schedule")

        tf.summary.scalar("learning_rate", lr)

        var_list = tf.trainable_variables()
        grad_vars = tf.gradients(total_loss, var_list)
        tf.summary.scalar(
            "grad_norm",
            tf.reduce_mean([tf.norm(grad_var) for grad_var in grad_vars]))
        grad_vars, _ = tf.clip_by_global_norm(grad_vars, 5.0)

        opt = tf.train.RMSPropOptimizer(lr, momentum=FLAGS.momentum,
                                        epsilon=1e-2)
        if FLAGS.num_replicas > 1:
          opt = tf.train.SyncReplicasOptimizer(
              opt,
              replicas_to_aggregate=FLAGS.num_replicas,
              total_num_replicas=FLAGS.num_replicas)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          opt_op = opt.apply_gradients(
              zip(grad_vars, var_list), global_step=global_step)

      # --------------- METRICS -------------------
      # Compute accuracy.
      top1_op = tf.nn.in_top_k(logits, labels, 1)
      accuracy = tf.reduce_mean(tf.cast(top1_op, dtype=tf.float32))
      my_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int64), tf.argmax(logits,1)), tf.float32))
      overall_confusion = tf.confusion_matrix(labels, tf.argmax(logits,1))
     
      predictions = tf.argmax(logits, 1)

      conf_matrix_per_group = { attr : {} for attr in protected_attributes}
      for attr in protected_attributes:
          conf_matrix_per_group[attr] = { ii : 0 for ii in unique_groups[attr]}

          # vectorized confusion matrix
          for ii in config.feature_dict[attr]:
              print(f"unique group value: {ii}")
              bool_vec = tf.equal(fairness_dict[attr], ii)   # get boolean for a particular group
              bool_vec.set_shape([None])
          
              group_predictions = tf.boolean_mask(predictions, bool_vec)   # valid preds for group
              true_label_per_group  = tf.boolean_mask(labels,  bool_vec)    # true labels for valid protected group
             
              conf_matrix_per_group[attr][ii] = tf.confusion_matrix(true_label_per_group, group_predictions)   # confusion matrix per group


      tf.summary.scalar("top1_accuracy", accuracy)
      # Prepare optimization.
      vars_to_save = tf.global_variables()
      saver = tf.train.Saver(var_list=vars_to_save, max_to_keep=5, sharded=True)
      merged_summary = tf.summary.merge_all()

      # Hooks for optimization.
      hooks = [tf.train.StopAtStepHook(last_step=max_iters)]
      if not is_chief:
        hooks.append(
            tf.train.GlobalStepWaiterHook(
                FLAGS.task * FLAGS.startup_delay_steps))

      init_op = tf.global_variables_initializer()
      scaffold = tf.train.Scaffold(
          init_op=init_op, summary_op=merged_summary, saver=saver)

      # Run optimization.
      epoch = 0
      robust_metric_epoch = { ii : 0 for ii in range(FLAGS.num_epochs+1) }
      acc_epoch = { ii : 0 for ii in range(FLAGS.num_epochs+1) }

      with tf.train.MonitoredTrainingSession(
          is_chief=is_chief,
          checkpoint_dir=FLAGS.checkpoint_dir,
          hooks=hooks,
          save_checkpoint_secs=FLAGS.save_checkpoint_secs,
          save_summaries_secs=FLAGS.save_summaries_secs,
          scaffold=scaffold) as sess:
        while not sess.should_stop():
           _, acc, i, robust_margin_value, fair_margin_value, confusion_dict, total_confusion, my_acc, pred_outcomes, true_label= \
                  sess.run((opt_op, accuracy, global_step, robust_margin_over_epochs,
             	       fair_margin_over_epochs, conf_matrix_per_group, overall_confusion, my_accuracy, predictions, labels))
           epoch = i // num_batches_per_epoch
           robust_metric_epoch[epoch] = robust_margin_value[0]

           acc_epoch[epoch] = acc

           for attr in fair_margin_value:
               fair_metric_epoch[attr][epoch] = fair_margin_value[attr]

               for ii in confusion_dict[attr]:
                   FP = confusion_dict[attr][ii][0][1]
                   FN = confusion_dict[attr][ii][1][0]
                   TP = confusion_dict[attr][ii][1][1]
                   TN = confusion_dict[attr][ii][0][0]

                   TPR_dict[attr][ii].append(TP/(TP+FN))
                   FPR_dict[attr][ii].append(FP/(FP+TN))
          

           if (i % FLAGS.log_every_steps) == 0:
             tf.logging.info("global step %d: epoch %d:\n train accuracy %.3f, my acc %.3f" %
                            (i, epoch, acc, my_acc))
             print(f"Confusion matrix : {confusion_dict}")


      used_attrbs = "_".join(protected_attributes)

      robustness_value = list(robust_metric_epoch.values())[1:]
      acc_value = acc_epoch.values()
      acc_value = list(acc_value)[1:]

      # Save out variables
      results = {}
      results['fairness'] = fair_metric_epoch 
      results['robustness'] = list(robust_metric_epoch.values())[1:]
      results['acc'] = list(acc_epoch.values())[1:]
      results['confusion_matrix'] = confusion_dict
      results['TPR'] = TPR_dict 
      results['FPR'] = FPR_dict
      results['model_prediction'] = pred_outcomes 
      results['true_label'] = true_label

      with open(f"save/{FLAGS.experiment_type}_train_{used_attrbs}_fair_{fair_loss_wt}_robust_{margin_loss_wt}_lr_{init_lr}_iter_{FLAGS.iteration}.pkl", 'wb') as handle:
          pickle.dump(results, handle)

