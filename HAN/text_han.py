# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm


class TextHAN(object):
    """A HAN for text classification."""

    def __init__(
            self, sequence_length, num_classes, vocab_size, lstm_hidden_size, fc_hidden_size,
            embedding_size, embedding_type, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

        def _linear(input_, output_size, scope="SimpleLinear"):
            """
            Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
            Args:
                input_: a tensor or a list of 2D, batch x n, Tensors.
                output_size: int, second dimension of W[i].
                scope: VariableScope for the created subgraph; defaults to "SimpleLinear".
            Returns:
                A 2D Tensor with shape [batch x output_size] equal to
                sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
            Raises:
                ValueError: if some of the arguments has unspecified or wrong shape.
            """

            shape = input_.get_shape().as_list()
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
            input_size = shape[1]

            # Now the computation.
            with tf.variable_scope(scope):
                W = tf.get_variable("W", [input_size, output_size], dtype=input_.dtype)
                b = tf.get_variable("b", [output_size], dtype=input_.dtype)

            return tf.nn.xw_plus_b(input_, W, b)

        def _highway_layer(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu):
            """
            Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wy + b)
            z = t * g(Wy + b) + (1 - t) * y
            where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
            """

            for idx in range(num_layers):
                g = f(_linear(input_, size, scope=("highway_lin_{0}".format(idx))))
                t = tf.sigmoid(_linear(input_, size, scope=("highway_gate_{0}".format(idx))) + bias)
                output = t * g + (1. - t) * input_
                input_ = output

            return output

        # Embedding Layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if pretrained_embedding is None:
                self.embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], minval=-1.0, maxval=1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="embedding")
            self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input_x)

        # Bi-LSTM Layer
        with tf.name_scope("Bi-lstm"):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # forward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # backward direction cell
            if self.dropout_keep_prob is not None:
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)

            # Creates a dynamic bidirectional recurrent neural network
            # shape of `outputs`: tuple -> (outputs_fw, outputs_bw)
            # shape of `outputs_fw`: [batch_size, sequence_length, lstm_hidden_size]

            # shape of `state`: tuple -> (outputs_state_fw, output_state_bw)
            # shape of `outputs_state_fw`: tuple -> (c, h) c: memory cell; h: hidden state
            outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                             self.embedded_sentence, dtype=tf.float32)

        # Concat output
        self.lstm_concat = tf.concat(outputs, axis=2)  # [batch_size, sequence_length, lstm_hidden_size * 2]

        # Attention Layer
        with tf.name_scope("attention"):
            num_units = self.lstm_concat.get_shape().as_list()[-1]  # Get last dimension [lstm_hidden_size * 2]
            u_attention = tf.Variable(tf.truncated_normal(shape=[num_units], stddev=0.1, dtype=tf.float32),
                                      name="u_attention")
            # 1. One-Layer MLP
            # W = tf.Variable(tf.truncated_normal(shape=[lstm_hidden_size * 2, num_units],
            #                                     stddev=0.1, dtype=tf.float32), name="W")
            # b = tf.Variable(tf.constant(0.1, shape=[num_units], dtype=tf.float32), name="b")
            # shape of `u`: [batch_size, sequence_length, num_units]
            u = tf.layers.dense(self.lstm_concat, units=num_units, activation=tf.nn.tanh, use_bias=True)

            # 2. Compute weight by computing similarity of u and attention vector u_attention
            score = tf.multiply(u, u_attention)  # [batch_size, sequence_length, num_units]
            weight = tf.reduce_mean(score, axis=2, keepdims=True)  # [batch_size, sequence_length, 1]

            # 3. Weight sum
            self.attention = tf.reduce_sum(tf.multiply(u, weight), axis=1)  # [batch_size, num_units]

        # Fully Connected Layer
        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=[num_units, fc_hidden_size],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b")
            self.fc = tf.nn.xw_plus_b(self.attention, W, b)

            # Batch Normalization Layer
            self.fc_bn = batch_norm(self.fc, is_training=self.is_training, trainable=True, updates_collections=None)

            # Apply nonlinearity
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")

        # Highway Layer
        with tf.name_scope("highway"):
            self.highway = _highway_layer(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final scores
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, num_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[num_classes], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.scores = tf.sigmoid(self.logits, name="scores")

        # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_mean(tf.reduce_sum(losses, axis=1), name="sigmoid_losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name="loss")
