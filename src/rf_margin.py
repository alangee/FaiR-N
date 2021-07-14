from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")


def abs_diff(tensor_input):
    # Calculated according to the first equation in http://mathworld.wolfram.com/GiniCoefficient.html
    pair_diff_sum = 0
    n = int(tensor_input.get_shape()[-1])

    for i in range(n):
        for j in range(i + 1, n):
            pair_diff_sum += tf.abs(tf.subtract(tensor_input[i], tensor_input[j]))

    return pair_diff_sum


def rf_margin(  # pylint: disable=invalid-name
    _sentinel=None,
    logits=None,
    one_hot_labels=None,
    top_k=1,
    fairness_dict = None,
    feature_dict = None,
    loss_collection=tf.losses,
    unfav_outcome=None):
  """
  Purpose:
    Calculate the distance to boundary for each datapoint using distance
        approximation. Calculate the burden for the protected groups and 
        the adversarial robustness of the model. 

  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    logits: Float `[batch_size, num_classes]` logits outputs of the network.
    one_hot_labels: `[batch_size, num_classes]` Target integer labels in `{0,
      1}`.
    top_k: Number of top classes to include in the margin loss.
    fairness_dict: Dictionary `Tensor` of the protected attribute group index 
        as related to the input data; size is `[batch_size, 1]` 
    feature_dict: Dictionary `Tensor` of the protected attribute (from config file) 
    unfav_outcome: Int label of the unfavorable outcome.
    loss_collection: Collection to which the loss will be added.

  Returns:
    robustness_loss: Scalar `Tensor` of the same type as `logits`
    burden_dict : Dictonary `Tensor` that holdes the burden loss for
       for each protected attribute group

  Raises:
    ValueError: If the shape of `logits` doesn't match that of
      `one_hot_labels`.  Also if `one_hot_labels` or `logits` is None.
  """
  logits = tf.convert_to_tensor(logits)

  assert top_k > 0
  assert top_k <= logits.get_shape()[1]

  with tf.name_scope("large_margin_loss"):
    class_prob = tf.nn.softmax(logits)
 
    # get max likelihood
    idx_max_prob = tf.argmax(class_prob, 1)
    shape_class_prob = tf.shape(class_prob)
    
    predicted_one_hot_label = tf.one_hot(idx_max_prob, tf.shape(class_prob)[1])
    predicted_class_prob = tf.reduce_sum(
      class_prob * predicted_one_hot_label , axis=1, keep_dims=True)   
    
    # top_k in binary case should be the class that's not chosen
    not_predicted_class_prob = class_prob * (1. - predicted_class_prob )
    
    # Pick the top k class probabilities other than the correct.
    if top_k > 1:
        top_k_not_pred_prob, _ = tf.nn.top_k(not_predicted_class_prob, k=top_k)
    else:
        top_k_not_pred_prob = tf.reduce_max(not_predicted_class_prob, axis=1, keep_dims=True)
    
    topLogit_distance_to_boundary = tf.math.log(predicted_class_prob) - tf.math.log(top_k_not_pred_prob)

    # calculate robustness 
    robustness_loss = tf.reduce_mean(tf.abs(topLogit_distance_to_boundary))
    
    # choose only the class that got an unfavorable outcome  : var unfav_outcome
    mask = tf.equal(idx_max_prob,  unfav_outcome)
    unfav_pred_mask  = tf.cast(mask, topLogit_distance_to_boundary.dtype)
    unfav_boundary_distances = tf.abs(tf.multiply(tf.squeeze(topLogit_distance_to_boundary), unfav_pred_mask)) 

    # loop through all the various protected attributes
    burden_dict = {}
    for key in feature_dict:
        protected_idx_vec = fairness_dict[key]
        unique_groups = feature_dict[key]

        burden = []

        # get burden for each group
        for ii in unique_groups:
            bool_vec = tf.equal(protected_idx_vec, ii)
        
            #tensor = unfav_boundary_distances[bool_vec]
            protected_attr_mask = tf.cast(bool_vec, unfav_boundary_distances.dtype)
            tensor = tf.multiply(unfav_boundary_distances, protected_attr_mask)
            # check that tensor isnt just an empty tensor, if it is sum is 0
            relavent_burden =  tf.cond(tf.equal(tf.reduce_sum(tensor), 0), 
        	      lambda : tf.constant(0.0), lambda : tf.reduce_sum(tensor))
            
            # denominator is sum of bool_vect, check if its not empty
            num_unfav_pred_in_group_ii = tf.logical_and(mask, bool_vec)
            
            denom_tensor = tf.reduce_sum(tf.cast(protected_attr_mask, tf.float32))

            denom_burden = tf.cond(tf.equal(denom_tensor, 0), 
        	      lambda : tf.constant(1.0), lambda : denom_tensor)
            
            burden.append(tf.divide(relavent_burden, denom_burden)) 

        # condense in tensor    	    
        tf_burden_array = tf.stack(burden)    
        burden_loss = abs_diff(tf_burden_array)
        # store in dictonary of protected groups
        burden_dict[key] = burden_loss

  return robustness_loss, burden_dict

