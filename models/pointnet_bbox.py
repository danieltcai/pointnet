import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 7))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx(Nx7) """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    num_output = num_point * 7

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform) 
    local_features = net_transformed # local features, dim (B x N x 64) e.g. (32 x 1024 x 64)
    net_transformed = tf.expand_dims(net_transformed, [2]) 

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool') 

    net = tf.reshape(net, [batch_size, -1])


    global_features = net # global features, dim (B x 1024)
    global_features = tf.expand_dims(global_features, axis=1) # (B x 1 x 1024)
    global_features = tf.tile(global_features, [1, num_point, 1]) # (B x N x 1024)

    concat_features = tf.concat(
        [local_features, global_features],
        axis=2)

    # print(concat_features.get_shape())
    assert concat_features.get_shape() == [batch_size, num_point, 64 + 1024]
    concat_features = tf.reshape(concat_features, [batch_size, -1])

    # Try to overfit to small dataset first
    # Don't use dropout for regression
    net_concat = tf_util.fully_connected(concat_features, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    # net_concat = tf_util.dropout(net_concat, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')
    net_concat = tf_util.fully_connected(net_concat, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    # net_concat = tf_util.dropout(net_concat, keep_prob=0.7, is_training=is_training,
    #                       scope='dp2')
    net_concat = tf_util.fully_connected(net_concat, num_output, activation_fn=None, scope='fc3')

    # print(net_concat.get_shape())
    assert net_concat.get_shape() == [batch_size, num_point * 7]

    return net_concat, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: Bx(Nx7),
        label: Bx(Nx7), """

    # Just predict position of grasps
    batch_size = pred.get_shape()[0].value
    num_point = pred.get_shape()[1].value / 7

    y_hat = tf.reshape(pred, [batch_size, num_point, 7]) # (B x N x 7)
    y = tf.reshape(label, [batch_size, num_point, 7])

    q_hat = y_hat[:, :, 0] # (B x N)
    q = y[:, :, 0]

    g_hat = y_hat[:, :, 1:4] # (B x N x 3)
    g = y[:, :, 1:4]

    robust_mask = tf.cast(tf.cast(q, tf.bool), tf.float32) # (B x N)

    #  L2 loss
    square_error = tf.square(g - g_hat) # (B x N x 3)
    per_point_loss = tf.reduce_sum(square_error, axis=-1) #(B x N)

    # Don't penalize the regression if there is no grasp present
    per_point_loss_coords = tf.multiply(robust_mask, per_point_loss) # (B x N)
    loss_coords = tf.reduce_sum(per_point_loss_coords, axis=-1) # (B)
    # End

    # Orig
    '''
    batch_size = pred.get_shape()[0].value
    num_point = pred.get_shape()[1].value / 7

    y_hat = tf.reshape(pred, [batch_size, num_point, 7]) # (B x N x 7)
    y = tf.reshape(label, [batch_size, num_point, 7])

    q_hat = y_hat[:, :, 0] # (B x N)
    q = y[:, :, 0]

    g_hat = y_hat[:, :, 1:] # (B x N x 6)
    g = y[:, :, 1:]

    robust_mask = tf.cast(tf.cast(q, tf.bool), tf.float32) # (B x N)

    #  L2 loss
    square_error = tf.square(g - g_hat) # (B x N x 6)
    per_point_loss = tf.reduce_sum(square_error, axis=-1) #(B x N)

    # Huber loss
    # huber_loss = tf.losses.huber_loss(
    #     g, g_hat, reduction=None) # (B x N x 6)
    # per_point_loss = tf.reduce_sum(huber_loss, axis=-1) #(B x N)
    

    # print(per_point_square_error.get_shape())
    # print(robust_mask.get_shape())

    # Don't penalize the regression if there is no grasp present
    per_point_loss_coords = tf.multiply(robust_mask, per_point_loss) # (B x N)
    loss_coords = tf.reduce_sum(per_point_loss_coords, axis=-1) # (B)
    '''
    # End Orig

    # TODO: Add other loss terms 

    regression_loss = tf.reduce_mean(loss_coords)
    tf.summary.scalar('regression loss', regression_loss)


    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    # classify_loss = tf.reduce_mean(loss)
    # tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return regression_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
