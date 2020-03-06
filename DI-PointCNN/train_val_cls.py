#!/usr/bin/python3
"""Training and Validation On Classification Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import shutil
import argparse
import importlib
import data_utils
import numpy as np
import pointfly as pf
import tensorflow as tf
from datetime import datetime
import h5py
import provider
import prepare_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s',default='models', help='Path to folder for saving check points and summary')
    parser.add_argument('--model', '-m', default='pointcnn_cls',help='Model to use')
    parser.add_argument('--setting', '-x',default='ScanObjectNN_x3_l4', help='Setting to use')
    parser.add_argument('--epochs', help='Number of training epochs (default defined in setting)', type=int)
    parser.add_argument('--batch_size', help='Batch size (default defined in setting)', type=int)
    parser.add_argument('--log', help='Log to FILE in save folder; use - for stdout (default is log.txt)', metavar='FILE', default='log.txt')#default='log.txt'输出存入文档//default='-'显示输出
    parser.add_argument('--no_timestamp_folder', help='Dont save to timestamp folder', action='store_true')
    parser.add_argument('--no_code_backup', help='Dont backup code', action='store_true',default='True')
    parser.add_argument('--err_data', type=float, default=5696)
    parser.add_argument('--weight', type=float, default=10)
    args = parser.parse_args()
    
    if not args.no_timestamp_folder:
        time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        root_folder = os.path.join(args.save_folder, '%s_%s_%s_%d' % (args.model, args.setting, time_string, os.getpid()))
    else:
        root_folder = args.save_folder
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    if args.log != '-':
        sys.stdout = open(os.path.join(root_folder, args.log), 'w')

    print('PID:', os.getpid())

    print(args)

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    num_epochs = args.epochs or setting.num_epochs
    batch_size = args.batch_size or setting.batch_size
    sample_num = setting.sample_num#1024
    step_val = setting.step_val
    rotation_range = setting.rotation_range
    rotation_range_val = setting.rotation_range_val
    scaling_range = setting.scaling_range
    scaling_range_val = setting.scaling_range_val
    jitter = setting.jitter
    jitter_val = setting.jitter_val
    pool_setting_val = None if not hasattr(setting, 'pool_setting_val') else setting.pool_setting_val
    pool_setting_train = None if not hasattr(setting, 'pool_setting_train') else setting.pool_setting_train
    ERR = args.err_data
    WEIGHT = args.weight

    
    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    if setting.data_set == 'ModelNet':
        prepare_data.make_ModelNet_data_A_B(ERR,batch_size)
    if setting.data_set == 'ScanObjectNN':
        prepare_data.make_ScanObjectNN_data_batch_A_B(ERR,batch_size)

    data_train, label_train, data_val, label_val = setting.load_fn(setting.path, setting.path_val)
    
    #data_train:[2*11416,2048,3],label_train:[11416,],data_val:[2882,2048,3],label_val:[2882,]
    if setting.balance_fn is not None:#None
        num_train_before_balance = data_train.shape[0]
        repeat_num = setting.balance_fn(label_train)
        data_train = np.repeat(data_train, repeat_num, axis=0)
        label_train = np.repeat(label_train, repeat_num, axis=0)
        data_train, label_train = data_utils.grouped_shuffle([data_train, label_train])
        num_epochs = math.floor(num_epochs * (num_train_before_balance / data_train.shape[0]))

    if setting.save_ply_fn is not None:#None
        folder = os.path.join(root_folder, 'pts')
        print('{}-Saving samples as .ply files to {}...'.format(datetime.now(), folder))
        sample_num_for_ply = min(512, data_train.shape[0])
        if setting.map_fn is None:
            data_sample = data_train[:sample_num_for_ply]
        else:
            data_sample_list = []
            for idx in range(sample_num_for_ply):
                data_sample_list.append(setting.map_fn(data_train[idx], 0)[0])
            data_sample = np.stack(data_sample_list)
        setting.save_ply_fn(data_sample, folder)

    num_train = data_train.shape[0]#11416
    point_num = data_train.shape[1]#2048
    num_val = data_val.shape[0]#2882

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, None, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    data_train_placeholder = tf.placeholder(data_train.dtype, data_train.shape, name='data_train')
    label_train_placeholder = tf.placeholder(tf.int64, label_train.shape, name='label_train')
    data_val_placeholder = tf.placeholder(data_val.dtype, data_val.shape, name='data_val')
    label_val_placeholder = tf.placeholder(tf.int64, label_val.shape, name='label_val')
    handle = tf.placeholder(tf.string, shape=[], name='handle')

    ######################################################################
    dataset_train = tf.data.Dataset.from_tensor_slices((data_train_placeholder, label_train_placeholder))
    #dataset_train = dataset_train.shuffle(buffer_size=batch_size * 4)

    if setting.map_fn is not None:#None
        dataset_train = dataset_train.map(lambda data, label:
                                          tuple(tf.py_func(setting.map_fn, [data, label], [tf.float32, label.dtype])),
                                          num_parallel_calls=setting.num_parallel_calls)

    if setting.keep_remainder:#Ture
        dataset_train = dataset_train.batch(batch_size)
        batch_num_per_epoch = math.ceil(num_train / batch_size)
    else:
        dataset_train = dataset_train.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        batch_num_per_epoch = math.floor(num_train / batch_size)
    dataset_train = dataset_train.repeat(num_epochs)
    iterator_train = dataset_train.make_initializable_iterator()
    batch_num = batch_num_per_epoch * num_epochs
    print('{}-{:d} training batches.'.format(datetime.now(), batch_num))

    dataset_val = tf.data.Dataset.from_tensor_slices((data_val_placeholder, label_val_placeholder))
    if setting.map_fn is not None:#None
        dataset_val = dataset_val.map(lambda data, label: tuple(tf.py_func(
            setting.map_fn, [data, label], [tf.float32, label.dtype])), num_parallel_calls=setting.num_parallel_calls)
    if setting.keep_remainder:#Ture
        dataset_val = dataset_val.batch(batch_size)
        batch_num_val = math.ceil(num_val / batch_size)
    else:
        dataset_val = dataset_val.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        batch_num_val = math.floor(num_val / batch_size)
    iterator_val = dataset_val.make_initializable_iterator()
    print('{}-{:d} testing batches per test.'.format(datetime.now(), batch_num_val))

    iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types)
    (pts_fts, labels) = iterator.get_next()

    pts_fts_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    features_augmented = None
    
    points_sampled = pts_fts_sampled
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)
 
    net = model.Net(points=points_augmented, features=features_augmented, is_training=is_training, setting=setting)
    logits = net.logits
    feature_A = net.feature_list_A
    feature_B = net.feature_list_B
    probs = tf.nn.softmax(logits, name='probs')
    predictions = tf.argmax(probs, axis=-1, name='predictions')

    labels_2d = tf.expand_dims(labels, axis=-1, name='labels_2d')
    labels_tile = tf.tile(labels_2d, (1, tf.shape(logits)[1]), name='labels_tile')
        
    
    # --------------------------------------------------------------------
    # compute the Loss of DINet
    different = tf.square(tf.subtract(feature_A,feature_B))
    different = tf.reduce_sum(different,1)
    
    label_A = tf.to_float(labels[0:tf.to_int32(batch_size/2)])
    label_B = tf.to_float(labels[tf.to_int32(batch_size/2):tf.to_int32(batch_size)])
    f_same = tf.multiply(tf.add(tf.sign(-tf.abs(tf.subtract(label_A,label_B))),1),different)
    f_diff = tf.divide(tf.sign(tf.abs(tf.subtract(label_A,label_B))),different)
    f_same = tf.reduce_sum(f_same,0)
    f_diff = tf.reduce_sum(f_diff,0)
    f_loss = tf.add(f_diff,f_same)
    # --------------------------------------------------------------------
    
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_tile, logits=logits) + ((f_loss/(61440*batch_size/2)) * WEIGHT)#10

    with tf.name_scope('metrics'):
        loss_mean_op, loss_mean_update_op = tf.metrics.mean(loss_op)
        t_1_acc_op, t_1_acc_update_op = tf.metrics.accuracy(labels_tile, predictions)
        t_1_per_class_acc_op, t_1_per_class_acc_update_op = tf.metrics.mean_per_class_accuracy(labels_tile,
                                                                                               predictions,
                                                                                               setting.num_class)
       
    reset_metrics_op = tf.variables_initializer([var for var in tf.local_variables()
                                                 if var.name.split('/')[0] == 'metrics'])

    _ = tf.summary.scalar('loss/train', tensor=loss_mean_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/train', tensor=t_1_acc_op, collections=['train'])
    _ = tf.summary.scalar('t_1_per_class_acc/train', tensor=t_1_per_class_acc_op, collections=['train'])

    _ = tf.summary.scalar('loss/val', tensor=loss_mean_op, collections=['val'])
    _ = tf.summary.scalar('t_1_acc/val', tensor=t_1_acc_op, collections=['val'])
    _ = tf.summary.scalar('t_1_per_class_acc/val', tensor=t_1_per_class_acc_op, collections=['val'])

    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                           setting.decay_rate, staircase=True)
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])
    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op, epsilon=setting.epsilon)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)

    # backup all code
    if not args.no_code_backup:
        code_folder = os.path.abspath(os.path.dirname(__file__))
        shutil.copytree(code_folder, os.path.join(root_folder, os.path.basename(code_folder)))

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        summaries_op = tf.summary.merge_all('train')
        summaries_val_op = tf.summary.merge_all('val')
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)

        sess.run(init_op)

        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))
        else:
            latest_ckpt = tf.train.latest_checkpoint(folder_ckpt)
            if latest_ckpt:
                print('{}-Found checkpoint {}'.format(datetime.now(), latest_ckpt))
                saver.restore(sess, latest_ckpt)
                print('{}-Checkpoint loaded from {} (Iter {})'.format(
                    datetime.now(), latest_ckpt, sess.run(global_step)))

        handle_train = sess.run(iterator_train.string_handle())
        handle_val = sess.run(iterator_val.string_handle())

        sess.run(iterator_train.initializer, feed_dict={
            data_train_placeholder: data_train,
            label_train_placeholder: label_train,
        })

        for batch_idx_train in range(batch_num):
            ######################################################################
            # Validation
            
            if (batch_idx_train % step_val == 0 and (batch_idx_train != 0 or args.load_ckpt is not None)) \
                    or batch_idx_train == batch_num - 1:
                sess.run(iterator_val.initializer, feed_dict={
                    data_val_placeholder: data_val,
                    label_val_placeholder: label_val,
                })
                filename_ckpt = os.path.join(folder_ckpt, 'iter')
                saver.save(sess, filename_ckpt, global_step=global_step)
                print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))

                sess.run(reset_metrics_op)
                total_seen_class = [0 for _ in range(setting.num_class)]
                total_correct_class = [0 for _ in range(setting.num_class)]
                for batch_idx_val in range(batch_num_val):
                    if not setting.keep_remainder \
                            or num_val % batch_size == 0 \
                            or batch_idx_val != batch_num_val - 1:
                        batch_size_val = batch_size
                    else:
                        batch_size_val = num_val % batch_size
                    xforms_np, rotations_np = pf.get_xforms(batch_size_val,
                                                            rotation_range=rotation_range_val,
                                                            scaling_range=scaling_range_val,
                                                            order=setting.rotation_order)
                    lll, ppp, _, _ = sess.run([labels, predictions, t_1_acc_update_op, t_1_per_class_acc_update_op],
                             feed_dict={
                                 handle: handle_val,
                                 indices: pf.get_indices(batch_size_val, sample_num, point_num,
                                                         ),
                                 xforms: xforms_np,
                                 rotations: rotations_np,
                                 jitter_range: np.array([jitter_val]),
                                 is_training: False,
                             })
                    ppp = np.reshape(ppp,[-1,])
                    for i in range(len(lll)):
                        l = lll[i]
                        total_seen_class[l] +=1
                        total_correct_class[l] +=(ppp[i] == l)
                    
                t_1_acc_val, t_1_per_class_acc_val, summaries_val, step = sess.run(
                    [t_1_acc_op, t_1_per_class_acc_op, summaries_val_op, global_step])
                summary_writer.add_summary(summaries_val, step)
                print('{}-[Val  ]-Average:     T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'
                      .format(datetime.now(),  t_1_acc_val, t_1_per_class_acc_val))
                print('every class acc:',np.array(total_correct_class)/np.array(total_seen_class,dtype = np.float))
                sys.stdout.flush()
                
            ######################################################################

            ######################################################################
            # Training
            if not setting.keep_remainder \
                    or num_train % batch_size == 0 \
                    or (batch_idx_train % batch_num_per_epoch) != (batch_num_per_epoch - 1):
                batch_size_train = batch_size
            else:
                batch_size_train = num_train % batch_size

            offset = int(random.gauss(0, sample_num * setting.sample_num_variance))
            offset = max(offset, -sample_num * setting.sample_num_clip)
            offset = min(offset, sample_num * setting.sample_num_clip)
            sample_num_train = sample_num + offset
            xforms_np, rotations_np = pf.get_xforms(batch_size_train,
                                                    rotation_range=rotation_range,
                                                    scaling_range=scaling_range,
                                                    order=setting.rotation_order)
            sess.run(reset_metrics_op)
            sess.run([train_op, loss_mean_update_op, t_1_acc_update_op, t_1_per_class_acc_update_op],
                     feed_dict={
                         handle: handle_train,
                         indices: pf.get_indices(batch_size_train, sample_num_train, point_num, pool_setting_train),
                         xforms: xforms_np,
                         rotations: rotations_np,
                         jitter_range: np.array([jitter]),
                         is_training: True,
                     })
            if batch_idx_train % 10 == 0:
                loss, t_1_acc, t_1_per_class_acc, summaries, step = sess.run([loss_mean_op,
                                                                        t_1_acc_op,
                                                                        t_1_per_class_acc_op,
                                                                        summaries_op,
                                                                        global_step])
                summary_writer.add_summary(summaries, step)
                print('{}-[Train]-Iter: {:06d}  Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'
                      .format(datetime.now(), step, loss, t_1_acc, t_1_per_class_acc))
                sys.stdout.flush()
            ######################################################################
        print('{}-Done!'.format(datetime.now()))

if __name__ == '__main__':
    main()
