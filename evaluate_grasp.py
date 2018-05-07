import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
# import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
# import pc_util


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_bbox', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--eval_dir', default='eval', help='eval folder path [eval]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
EVAL_DIR = FLAGS.eval_dir
if not os.path.exists(EVAL_DIR): os.mkdir(EVAL_DIR)
LOG_FOUT = open(os.path.join(EVAL_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# Set gpu
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_INDEX)

# NUM_CLASSES = 40
# SHAPE_NAMES = [line.rstrip() for line in \
#     open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

# HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
# TRAIN_FILES = provider.getDataFiles( \
#     os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/gen_data_kit_h5/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def rotation_matrix(alpha, beta, gamma):
    Rx = np.array([[1, 0, 0],
                   [0,  np.cos(alpha), -np.sin(alpha)],
                   [0,  np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    # R = Rz * Ry * Rx
    R = np.dot(np.dot(Rz, Ry), Rx)
    # R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        # Get placeholders
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # Get model and loss
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    # total_seen_class = [0 for _ in range(NUM_CLASSES)]
    # total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(EVAL_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        print("file data shape: {}".format(current_data.shape))
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        # print(file_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx
            
            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            # batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
            # batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes

        # for vote_idx in range(num_votes):
            # rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
            #                                   vote_idx/float(num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx, :, :],
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
            # batch_pred_sum += pred_val
            # batch_pred_val = np.argmax(pred_val, 1)
            # for el_idx in range(cur_batch_size):
            #     batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
                
            # pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END
            
            # correct = np.sum(pred_val == current_label[start_idx:end_idx])
            # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
            # total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            log_string('Visualize pred and label for batch: {} (start: {}, end: {})'.format(\
                batch_idx, start_idx, end_idx))

            # For each example in the batch  
            for i in range(start_idx, end_idx):
                log_string('Example num: {}'.format(i))

                point_cloud = current_data[i]
                point_labels = current_label[i]
                preds = pred_val[i - start_idx]
                preds = np.reshape(preds, point_labels.shape)

                # print("point_cloud.shape: {}".format(point_cloud.shape))
                # print("point_labels.shape: {}".format(point_labels.shape))
                # print("preds.shape: {}".format(preds.shape))


                # Visualize point cloud, label, pred
                for pt_idx, point_label in enumerate(point_labels):
                    # Calculate distances
                    pred = preds[pt_idx]
                    log_string("Point: {}".format(pt_idx))
                    gq_distance = np.linalg.norm(pred[0] - point_label[0])
                    pos_distance = np.linalg.norm(pred[1:4] - point_label[1:4])
                    ori_distance = np.linalg.norm(pred[4:] - point_label[4:])
                    log_string("GQ distance: {}".format(gq_distance))
                    log_string("Pos distance: {}".format(pos_distance))
                    log_string("Ori distance: {}".format(ori_distance))

                    log_string("Label for point: {}".format(pt_idx))
                    log_string(point_label)
                    log_string("Pred for point: {}".format(pt_idx))
                    log_string(pred)


                #     # GROUND TRUTH
                #     # Plot point cloud
                #     fig = plt.figure(1)
                #     ax = fig.add_subplot(111, projection='3d')
                #     pc_xs = point_cloud[:,0]
                #     pc_ys = point_cloud[:,1]
                #     pc_zs = point_cloud[:,2]
                #     ax.scatter(pc_xs, pc_ys, pc_zs, s=1, c='blue') # Object point cloud

                #     # Plot grasp position
                #     point = point_cloud[pt_idx]
                #     ax.scatter(point[0], point[1], point[2], s=50, c='purple') # Point being considered

                #     # Plot nearest grasp according to label. Dont forget pos is offset from point
                #     grasp_x = point_label[1] + point[0] 
                #     grasp_y = point_label[2] + point[1]
                #     grasp_z = point_label[3] + point[2]
                #     grasp_pos = np.reshape([grasp_x, grasp_y, grasp_z], [3,1])

                #     # Plot grasp axes orientation
                #     alpha = point_label[4]
                #     beta = point_label[5]
                #     gamma = point_label[6]
                #     R = rotation_matrix(alpha, beta, gamma)

                #     x_axis = np.reshape([0.01,0,0], [3,1])
                #     x_axis = np.dot(R, x_axis) + grasp_pos
                #     y_axis = np.reshape([0,0.01,0], [3,1])
                #     y_axis = np.dot(R, y_axis) + grasp_pos
                #     z_axis = np.reshape([0,0,0.01], [3,1])
                #     z_axis = np.dot(R, z_axis) + grasp_pos



                #     exists_near_robust = point_label[0]
                #     if exists_near_robust:
                #         ax.scatter(grasp_x, grasp_y, grasp_z, s=50, c='green') # Nearby robust grasp positions
                #     else: 
                #         ax.scatter(grasp_x, grasp_y, grasp_z, s=50, c='red') # Nearest but still too far robust grasp positions

                #     ax.scatter(x_axis[0], x_axis[1], x_axis[2], s=50, c='red') # Nearest but still too far robust grasp positions
                #     ax.scatter(y_axis[0], y_axis[1], y_axis[2], s=50, c='green') # Nearest but still too far robust grasp positions
                #     ax.scatter(z_axis[0], z_axis[1], z_axis[2], s=50, c='blue') # Nearest but still too far robust grasp positions
                #     plt.show()




                #     # PREDICTION
                #     # Plot point cloud
                #     fig = plt.figure(2)
                #     ax = fig.add_subplot(111, projection='3d')
                #     pc_xs = point_cloud[:,0]
                #     pc_ys = point_cloud[:,1]
                #     pc_zs = point_cloud[:,2]
                #     ax.scatter(pc_xs, pc_ys, pc_zs, s=1, c='blue') # Point being considered

                #     # Plot grasp position
                #     point = point_cloud[pt_idx]
                #     ax.scatter(point[0], point[1], point[2], s=50, c='purple') # Object point cloud

                #     # Plot nearest grasp according to label. Dont forget pos is offset from point
                #     grasp_x = pred[1] + point[0] 
                #     grasp_y = pred[2] + point[1]
                #     grasp_z = pred[3] + point[2]
                #     grasp_pos = np.reshape([grasp_x, grasp_y, grasp_z], [3,1])

                #     # Plot grasp axes orientation
                #     alpha = pred[4]
                #     beta = pred[5]
                #     gamma = pred[6]
                #     R = rotation_matrix(alpha, beta, gamma)

                #     x_axis = np.reshape([0.01,0,0], [3,1])
                #     x_axis = np.dot(R, x_axis) + grasp_pos
                #     y_axis = np.reshape([0,0.01,0], [3,1])
                #     y_axis = np.dot(R, y_axis) + grasp_pos
                #     z_axis = np.reshape([0,0,0.01], [3,1])
                #     z_axis = np.dot(R, z_axis) + grasp_pos



                #     exists_near_robust = pred[0]
                #     if exists_near_robust:
                #         ax.scatter(grasp_x, grasp_y, grasp_z, s=50, c='green') # Nearby robust grasp positions
                #     else: 
                #         ax.scatter(grasp_x, grasp_y, grasp_z, s=50, c='red') # Nearest but still too far robust grasp positions

                #     ax.scatter(x_axis[0], x_axis[1], x_axis[2], s=50, c='red') # Nearest but still too far robust grasp positions
                #     ax.scatter(y_axis[0], y_axis[1], y_axis[2], s=50, c='green') # Nearest but still too far robust grasp positions
                #     ax.scatter(z_axis[0], z_axis[1], z_axis[2], s=50, c='blue') # Nearest but still too far robust grasp positions
                #     plt.show()








                # total_seen_class[l] += 1
                # total_correct_class[l] += (pred_val[i-start_idx] == l)
                # fout.write('%d, %d\n' % (pred_val[i-start_idx], l))
                
                # if pred_val[i-start_idx] != l and FLAGS.visu: # ERROR CASE, DUMP!
                #     img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                #                                            SHAPE_NAMES[pred_val[i-start_idx]])
                #     img_filename = os.path.join(EVAL_DIR, img_filename)
                #     output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                #     scipy.misc.imsave(img_filename, output_img)
                #     error_cnt += 1
                
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    # log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    # log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    # class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    # for i, name in enumerate(SHAPE_NAMES):
    #     log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
