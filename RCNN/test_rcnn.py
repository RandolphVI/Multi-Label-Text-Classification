# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging
import numpy as np

sys.path.append('../')
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf

from utils import checkmate as cm
from utils import data_helpers as dh
from utils import param_parser as parser
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

args = parser.parameter_parser()
MODEL = dh.get_model_name()
logger = dh.logger_fn("tflog", "logs/Test-{0}.log".format(time.asctime()))

CPT_DIR = 'runs/' + MODEL + '/checkpoints/'
BEST_CPT_DIR = 'runs/' + MODEL + '/bestcheckpoints/'
SAVE_DIR = 'output/' + MODEL


def create_input_data(data: dict):
    return zip(data['pad_seqs'], data['onehot_labels'], data['labels'])


def test_rcnn():
    """Test RCNN model."""
    # Print parameters used for the model
    dh.tab_printer(args, logger)

    # Load word2vec model
    word2idx, embedding_matrix = dh.load_word2vec_matrix(args.word2vec_file)

    # Load data
    logger.info("Loading data...")
    logger.info("Data processing...")
    test_data = dh.load_data_and_labels(args, args.test_file, word2idx)

    # Load rcnn model
    OPTION = dh._option(pattern=1)
    if OPTION == 'B':
        logger.info("Loading best model...")
        checkpoint_file = cm.get_best_checkpoint(BEST_CPT_DIR, select_maximum_value=True)
    else:
        logger.info("Loading latest model...")
        checkpoint_file = tf.train.latest_checkpoint(CPT_DIR)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
        session_conf.gpu_options.allow_growth = args.gpu_options_allow_growth
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
            batches = dh.batch_iter(list(create_input_data(test_data)), args.batch_size, 1, shuffle=False)

            # Collect the predictions here
            test_counter, test_loss = 0, 0.0
            test_pre_tk = [0.0] * args.topK
            test_rec_tk = [0.0] * args.topK
            test_F1_tk = [0.0] * args.topK

            # Collect the predictions here
            true_labels = []
            predicted_labels = []
            predicted_scores = []

            # Collect for calculating metrics
            true_onehot_labels = []
            predicted_onehot_scores = []
            predicted_onehot_labels_ts = []
            predicted_onehot_labels_tk = [[] for _ in range(args.topK)]

            for batch_test in batches:
                x, y_onehot, y = zip(*batch_test)
                feed_dict = {
                    input_x: x,
                    input_y: y_onehot,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }

                batch_scores, cur_loss = sess.run([scores, loss], feed_dict)

                # Prepare for calculating metrics
                for i in y_onehot:
                    true_onehot_labels.append(i)
                for j in batch_scores:
                    predicted_onehot_scores.append(j)

                # Get the predicted labels by threshold
                batch_predicted_labels_ts, batch_predicted_scores_ts = \
                    dh.get_label_threshold(scores=batch_scores, threshold=args.threshold)

                # Add results to collection
                for i in y:
                    true_labels.append(i)
                for j in batch_predicted_labels_ts:
                    predicted_labels.append(j)
                for k in batch_predicted_scores_ts:
                    predicted_scores.append(k)

                # Get onehot predictions by threshold
                batch_predicted_onehot_labels_ts = \
                    dh.get_onehot_label_threshold(scores=batch_scores, threshold=args.threshold)
                for i in batch_predicted_onehot_labels_ts:
                    predicted_onehot_labels_ts.append(i)

                # Get onehot predictions by topK
                for top_num in range(args.topK):
                    batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=batch_scores, top_num=top_num+1)

                    for i in batch_predicted_onehot_labels_tk:
                        predicted_onehot_labels_tk[top_num].append(i)

                test_loss = test_loss + cur_loss
                test_counter = test_counter + 1

            # Calculate Precision & Recall & F1
            test_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                          y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            test_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                       y_pred=np.array(predicted_onehot_labels_ts), average='micro')
            test_F1_ts = f1_score(y_true=np.array(true_onehot_labels),
                                  y_pred=np.array(predicted_onehot_labels_ts), average='micro')

            for top_num in range(args.topK):
                test_pre_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                       y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                       average='micro')
                test_rec_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                    y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                    average='micro')
                test_F1_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                               y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                               average='micro')

            # Calculate the average AUC
            test_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                     y_score=np.array(predicted_onehot_scores), average='micro')

            # Calculate the average PR
            test_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                               y_score=np.array(predicted_onehot_scores), average="micro")
            test_loss = float(test_loss / test_counter)

            logger.info("All Test Dataset: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                        .format(test_loss, test_auc, test_prc))

            # Predict by threshold
            logger.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F1 {2:g}"
                        .format(test_pre_ts, test_rec_ts, test_F1_ts))

            # Predict by topK
            logger.info("Predict by topK:")
            for top_num in range(args.topK):
                logger.info("Top{0}: Precision {1:g}, Recall {2:g}, F1 {3:g}"
                            .format(top_num + 1, test_pre_tk[top_num], test_rec_tk[top_num], test_F1_tk[top_num]))

            # Save the prediction result
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            dh.create_prediction_file(output_file=SAVE_DIR + "/predictions.json", data_id=test_data['id'],
                                      true_labels=true_labels, predict_labels=predicted_labels,
                                      predict_scores=predicted_scores)

    logger.info("All Done.")


if __name__ == '__main__':
    test_rcnn()
