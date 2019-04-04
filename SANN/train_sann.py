# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf

from tensorboard.plugins import projector
from text_sann import TextSANN
from utils import checkmate as cm
from utils import data_helpers as dh
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Parameters
# ==================================================

TRAIN_OR_RESTORE = input("☛ Train or Restore?(T/R): ")

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input("✘ The format of your input is illegal, please re-input: ")
logging.info("✔︎ The format of your input is legal, now loading to next step...")

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn("tflog", "logs/training-{0}.log".format(time.asctime()))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn("tflog", "logs/restore-{0}.log".format(time.asctime()))

TRAININGSET_DIR = '../data/Train.json'
VALIDATIONSET_DIR = '../data/Validation.json'
METADATA_DIR = '../data/metadata.tsv'

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")
tf.flags.DEFINE_string("metadata_file", METADATA_DIR, "Metadata file for embedding visualization"
                                                      "(Each line is a word segment in metadata_file).")
tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate (default: 0.001)")
tf.flags.DEFINE_integer("pad_seq_len", 100, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("lstm_hidden_size", 256, "Hidden size for bi-lstm layer(default: 256)")
tf.flags.DEFINE_integer("attention_unit_size", 350, "Attention unit size(default: 350)")
tf.flags.DEFINE_integer("attention_hops_size", 30, "Attention hops size(default: 30)")
tf.flags.DEFINE_boolean("attention_penalization", True, "Use penalization or not(default: True)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 367, "Number of labels (depends on the task)")
tf.flags.DEFINE_integer("top_num", 5, "Number of top K prediction classes (default: 5)")
tf.flags.DEFINE_float("threshold", 0.5, "Threshold for prediction classes (default: 0.5)")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("num_epochs", 150, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 5000, "Evaluate model on dev set after this many steps (default: 5000)")
tf.flags.DEFINE_float("norm_ratio", 2, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate. (default: 500)")
tf.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate. (default: 0.95)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 50)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def train_sann():
    """Training RNN model."""

    # Load sentences, labels, and training parameters
    logger.info("✔︎ Loading data...")

    logger.info("✔︎ Training data processing...")
    train_data = dh.load_data_and_labels(FLAGS.training_data_file, FLAGS.num_classes,
                                         FLAGS.embedding_dim, data_aug_flag=False)

    logger.info("✔︎ Validation data processing...")
    val_data = dh.load_data_and_labels(FLAGS.validation_data_file, FLAGS.num_classes,
                                       FLAGS.embedding_dim, data_aug_flag=False)

    logger.info("Recommended padding Sequence length is: {0}".format(FLAGS.pad_seq_len))

    logger.info("✔︎ Training data padding...")
    x_train, y_train = dh.pad_data(train_data, FLAGS.pad_seq_len)

    logger.info("✔︎ Validation data padding...")
    x_val, y_val = dh.pad_data(val_data, FLAGS.pad_seq_len)

    # Build vocabulary
    VOCAB_SIZE, pretrained_word2vec_matrix = dh.load_word2vec_matrix(FLAGS.embedding_dim)

    # Build a graph and sann object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            sann = TextSANN(
                sequence_length=FLAGS.pad_seq_len,
                num_classes=FLAGS.num_classes,
                vocab_size=VOCAB_SIZE,
                lstm_hidden_size=FLAGS.lstm_hidden_size,
                attention_unit_size=FLAGS.attention_unit_size,
                attention_hops_size=FLAGS.attention_hops_size,
                fc_hidden_size=FLAGS.fc_hidden_size,
                embedding_size=FLAGS.embedding_dim,
                embedding_type=FLAGS.embedding_type,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pretrained_embedding=pretrained_word2vec_matrix)

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=sann.global_step, decay_steps=FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(sann.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=sann.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("☛ Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input("✘ The format of your input is illegal, please re-input: ")
                logger.info("✔︎ The format of your input is legal, now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("✔︎ Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("✔︎ Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", sann.loss)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            if FLAGS.train_or_restore == 'R':
                # Load sann model
                logger.info("✔︎ Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Embedding visualization config
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = "embedding"
                embedding_conf.metadata_path = FLAGS.metadata_file

                projector.visualize_embeddings(train_summary_writer, config)
                projector.visualize_embeddings(validation_summary_writer, config)

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(sann.global_step)

            def train_step(x_batch, y_batch):
                """A single training step"""
                feed_dict = {
                    sann.input_x: x_batch,
                    sann.input_y: y_batch,
                    sann.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    sann.is_training: True
                }
                _, step, summaries, loss = sess.run(
                    [train_op, sann.global_step, train_summary_op, sann.loss], feed_dict)
                logger.info("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_val, y_val, writer=None):
                """Evaluates model on a validation set"""
                batches_validation = dh.batch_iter(list(zip(x_val, y_val)), FLAGS.batch_size, 1)

                # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
                eval_counter, eval_loss = 0, 0.0

                eval_pre_tk = [0.0] * FLAGS.top_num
                eval_rec_tk = [0.0] * FLAGS.top_num
                eval_F_tk = [0.0] * FLAGS.top_num

                true_onehot_labels = []
                predicted_onehot_scores = []
                predicted_onehot_labels_ts = []
                predicted_onehot_labels_tk = [[] for _ in range(FLAGS.top_num)]

                for batch_validation in batches_validation:
                    x_batch_val, y_batch_val = zip(*batch_validation)
                    feed_dict = {
                        sann.input_x: x_batch_val,
                        sann.input_y: y_batch_val,
                        sann.dropout_keep_prob: 1.0,
                        sann.is_training: False
                    }
                    step, summaries, scores, cur_loss = sess.run(
                        [sann.global_step, validation_summary_op, sann.scores, sann.loss], feed_dict)

                    # Prepare for calculating metrics
                    for i in y_batch_val:
                        true_onehot_labels.append(i)
                    for j in scores:
                        predicted_onehot_scores.append(j)

                    # Predict by threshold
                    batch_predicted_onehot_labels_ts = \
                        dh.get_onehot_label_threshold(scores=scores, threshold=FLAGS.threshold)

                    for k in batch_predicted_onehot_labels_ts:
                        predicted_onehot_labels_ts.append(k)

                    # Predict by topK
                    for top_num in range(FLAGS.top_num):
                        batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores, top_num=top_num+1)

                        for i in batch_predicted_onehot_labels_tk:
                            predicted_onehot_labels_tk[top_num].append(i)

                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1

                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)

                # Calculate Precision & Recall & F1 (threshold & topK)
                eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                              y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                           y_pred=np.array(predicted_onehot_labels_ts), average='micro')
                eval_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                                     y_pred=np.array(predicted_onehot_labels_ts), average='micro')

                for top_num in range(FLAGS.top_num):
                    eval_pre_tk[top_num] = precision_score(y_true=np.array(true_onehot_labels),
                                                           y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                           average='micro')
                    eval_rec_tk[top_num] = recall_score(y_true=np.array(true_onehot_labels),
                                                        y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                        average='micro')
                    eval_F_tk[top_num] = f1_score(y_true=np.array(true_onehot_labels),
                                                  y_pred=np.array(predicted_onehot_labels_tk[top_num]),
                                                  average='micro')

                # Calculate the average AUC
                eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                         y_score=np.array(predicted_onehot_scores), average='micro')
                # Calculate the average PR
                eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                                   y_score=np.array(predicted_onehot_scores), average='micro')

                return eval_loss, eval_auc, eval_prc, eval_rec_ts, eval_pre_ts, eval_F_ts, \
                       eval_rec_tk, eval_pre_tk, eval_F_tk

            # Generate batches
            batches_train = dh.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            num_batches_per_epoch = int((len(x_train) - 1) / FLAGS.batch_size) + 1

            # Training loop. For each batch...
            for batch_train in batches_train:
                x_batch_train, y_batch_train = zip(*batch_train)
                train_step(x_batch_train, y_batch_train)
                current_step = tf.train.global_step(sess, sann.global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    logger.info("\nEvaluation:")
                    eval_loss, eval_auc, eval_prc, \
                    eval_rec_ts, eval_pre_ts, eval_F_ts, eval_rec_tk, eval_pre_tk, eval_F_tk = \
                        validation_step(x_val, y_val, writer=validation_summary_writer)

                    logger.info("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}"
                                .format(eval_loss, eval_auc, eval_prc))

                    # Predict by threshold
                    logger.info("☛ Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}"
                                .format(eval_pre_ts, eval_rec_ts, eval_F_ts))

                    # Predict by topK
                    logger.info("☛ Predict by topK:")
                    for top_num in range(FLAGS.top_num):
                        logger.info("Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}"
                                    .format(top_num+1, eval_pre_tk[top_num], eval_rec_tk[top_num], eval_F_tk[top_num]))
                    best_saver.handle(eval_prc, sess, current_step)
                if current_step % FLAGS.checkpoint_every == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.info("✔︎ Saved model checkpoint to {0}\n".format(path))
                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    logger.info("✔︎ Epoch {0} has finished!".format(current_epoch))

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    train_sann()
