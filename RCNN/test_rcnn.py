# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import numpy as np
import tensorflow as tf

from utils import checkmate as cm
from utils import data_helpers as dh
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Parameters
# ==================================================

logger = dh.logger_fn("tflog", "logs/test-{0}.log".format(time.asctime()))

MODEL = input("☛ Please input the model file you want to test, it should be like(1490175368): ")

while not (MODEL.isdigit() and len(MODEL) == 10):
    MODEL = input("✘ The format of your input is illegal, it should be like(1490175368), please re-input: ")
logger.info("✔︎ The format of your input is legal, now loading to next step...")

TRAININGSET_DIR = '../data/Train.json'
VALIDATIONSET_DIR = '../data/Validation.json'
TESTSET_DIR = '../data/Test.json'
MODEL_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_MODEL_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'results/' + MODEL

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data")
tf.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")
tf.flags.DEFINE_string("best_checkpoint_dir", BEST_MODEL_DIR, "Best checkpoint directory from training run")

# Model Hyperparameters
tf.flags.DEFINE_integer("pad_seq_len", 100, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("lstm_hidden_size", 256, "Hidden size for bi-lstm layer(default: 256)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 367, "Number of labels (depends on the task)")
tf.flags.DEFINE_integer("top_num", 5, "Number of top K prediction classes (default: 5)")
tf.flags.DEFINE_float("threshold", 0.5, "Threshold for prediction classes (default: 0.5)")

# Test Parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 1)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def test_rcnn():
    """Test RCNN model."""

    # Load data
    logger.info("✔︎ Loading data...")
    logger.info("Recommended padding Sequence length is: {0}".format(FLAGS.pad_seq_len))

    logger.info("✔︎ Test data processing...")
    test_data = dh.load_data_and_labels(FLAGS.test_data_file, FLAGS.num_classes,
                                        FLAGS.embedding_dim, data_aug_flag=False)

    logger.info("✔︎ Test data padding...")
    x_test, y_test = dh.pad_data(test_data, FLAGS.pad_seq_len)
    y_test_labels = test_data.labels

    # Load rcnn model
    BEST_OR_LATEST = input("☛ Load Best or Latest Model?(B/L): ")

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("✘ The format of your input is illegal, please re-input: ")
    if BEST_OR_LATEST.upper() == 'B':
        logger.info("✔︎ Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(FLAGS.best_checkpoint_dir, select_maximum_value=True)
    else:
        logger.info("✔︎ Loading latest model...")
        checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs[0]
            loss = graph.get_operation_by_name("loss/loss").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = "output/scores"

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, "graph", "graph-rcnn-{0}.pb".format(MODEL), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(zip(x_test, y_test, y_test_labels)), FLAGS.batch_size, 1, shuffle=False)

            test_counter, test_loss = 0, 0.0

            test_pre_tk = [0.0] * FLAGS.top_num
            test_rec_tk = [0.0] * FLAGS.top_num
            test_F_tk = [0.0] * FLAGS.top_num

            # Collect the predictions here
            true_labels = []
            predicted_labels = []
            predicted_scores = []

            # Collect for calculating metrics
            true_onehot_labels = []
            predicted_onehot_scores = []
            predicted_onehot_labels_ts = []
            predicted_onehot_labels_tk = [[] for _ in range(FLAGS.top_num)]

            for batch_test in batches:
                x_batch_test, y_batch_test, y_batch_test_labels = zip(*batch_test)
                feed_dict = {
                    input_x: x_batch_test,
                    input_y: y_batch_test,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }
                batch_scores, cur_loss = sess.run([scores, loss], feed_dict)

                # Prepare for calculating metrics
                for i in y_batch_test:
                    true_onehot_labels.append(i)
                for j in batch_scores:
                    predicted_onehot_scores.append(j)

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores, threshold=FLAGS.threshold)

                # Add results to collection
                for i in y_batch_test_labels:
                    true_labels.append(i)
                for j in batch_predicted_labels_ts:
                    predicted_labels.append(j)
                for k in batch_predicted_scores_ts:
                    predicted_scores.append(k)

                # Get onehot predictions by threshold
                batch_predicted_onehot_labels_ts = \
                    dh.get_onehot_label_threshold(scores=batch_scores, threshold=FLAGS.threshold)
                for i in batch_predicted_onehot_labels_ts:
                    predicted_onehot_labels_ts.append(i)

                # Get onehot predictions by topK
                for top_num in range(FLAGS.top_num):
                    batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=batch_scores, top_num=top_num+1)

                    for i in batch_predicted_onehot_labels_tk:
                        predicted_onehot_labels_tk[top_num].append(i)

                test_loss = test_loss + cur_loss
                test_counter = test_counter + 1

            # Calculate Precision & Recall & F1 (threshold & topK)
            test_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                          y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            test_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                       y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            test_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                                 y_pred=np.array(predicted_onehot_labels_ts), average='micro')

            for top_num in range(FLAGS.top_num):
                test_pre_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                       y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                       average='micro')
                test_rec_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                    y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                    average='micro')
                test_F_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                              y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                              average='micro')

            # Calculate the average AUC
            test_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                     y_score=np.array(predicted_onehot_scores), average='micro')

            # Calculate the average PR
            test_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                               y_score=np.array(predicted_onehot_scores), average="micro")
            test_loss = float(test_loss / test_counter)

            logger.info("☛ All Test Dataset: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                        .format(test_loss, test_auc, test_prc))

            # Predict by threshold
            logger.info("☛ Predict by threshold: Precision {0:g}, Recall {1:g}, F1 {2:g}"
                        .format(test_pre_ts, test_rec_ts, test_F_ts))

            # Predict by topK
            logger.info("☛ Predict by topK:")
            for top_num in range(FLAGS.top_num):
                logger.info("Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}"
                            .format(top_num + 1, test_pre_tk[top_num], test_rec_tk[top_num], test_F_tk[top_num]))

            # Save the prediction result
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            dh.create_prediction_file(output_file=SAVE_DIR + "/predictions.json", data_id=test_data.testid,
                                      all_labels=true_labels, all_predict_labels=predicted_labels,
                                      all_predict_scores=predicted_scores)

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    test_rcnn()
