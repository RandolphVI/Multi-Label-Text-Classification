# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import heapq
import multiprocessing
import gensim
import logging
import json
import numpy as np

from collections import OrderedDict
from pylab import *
from gensim.models import word2vec
from tflearn.data_utils import pad_sequences

TEXT_DIR = '../data/content.txt'
METADATA_DIR = '../data/metadata.tsv'


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def create_prediction_file(output_file, data_id, all_labels, all_predict_labels, all_predict_scores):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted results provided by network
        data_id: The data record id info provided by class Data
        all_labels: The all origin labels
        all_predict_labels: The all predict labels by threshold
        all_predict_scores: The all predict scores by threshold
    Raises:
        IOError: If the prediction file is not a <.json> file
    """
    if not output_file.endswith('.json'):
        raise IOError("✘ The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(all_predict_labels)
        for i in range(data_size):
            predict_labels = [int(i) for i in all_predict_labels[i]]
            predict_scores = [round(i, 4) for i in all_predict_scores[i]]
            labels = [int(i) for i in all_labels[i]]
            data_record = OrderedDict([
                ('id', data_id[i]),
                ('labels', labels),
                ('predict_labels', predict_labels),
                ('predict_scores', predict_scores)
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def get_onehot_label_threshold(scores, threshold=0.5):
    """
    Get the predicted onehot labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        onehot_labels_list = [0] * len(score)
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                onehot_labels_list[index] = 1
                count += 1
        if count == 0:
            max_score_index = score.index(max(score))
            onehot_labels_list[max_score_index] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_onehot_label_topk(scores, top_num=1):
    """
    Get the predicted onehot labels based on the topK number.

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        predicted_onehot_labels: The predicted labels (onehot)
    """
    predicted_onehot_labels = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        onehot_labels_list = [0] * len(score)
        max_num_index_list = list(map(score.index, heapq.nlargest(top_num, score)))
        for i in max_num_index_list:
            onehot_labels_list[i] = 1
        predicted_onehot_labels.append(onehot_labels_list)
    return predicted_onehot_labels


def get_label_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict score greater than threshold, then choose the label which has the max predict score.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_labels: The predicted labels
        predicted_scores: The predicted scores
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        score_list = []
        for index, predict_score in enumerate(score):
            if predict_score >= threshold:
                index_list.append(index)
                score_list.append(predict_score)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            score_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def get_label_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK number.
    Note: Only Used in `test_model.py`

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_scores = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        score_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            score_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_scores.append(score_list)
    return predicted_labels, predicted_scores


def create_metadata_file(embedding_size, output_file=METADATA_DIR):
    """
    Create the metadata file based on the corpus file(Use for the Embedding Visualization later).

    Args:
        embedding_size: The embedding size
        output_file: The metadata file (default: 'metadata.tsv')
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist."
                      "Please use function <create_vocab_size(embedding_size)> to create it!")

    model = gensim.models.Word2Vec.load(word2vec_file)
    word2idx = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def create_word2vec_model(embedding_size, input_file=TEXT_DIR):
    """
    Create the word2vec model based on the given embedding size and the corpus file.

    Args:
        embedding_size: The embedding size
        input_file: The corpus file
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    sentences = word2vec.LineSentence(input_file)
    # sg=0 means use CBOW model(default); sg=1 means use skip-gram model.
    model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
                                   sg=0, workers=multiprocessing.cpu_count())
    model.save(word2vec_file)


def load_word2vec_matrix(embedding_size):
    """
    Return the word2vec model matrix.

    Args:
        embedding_size: The embedding size
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist. "
                      "Please use function <create_vocab_size(embedding_size)> to create it!")

    model = gensim.models.Word2Vec.load(word2vec_file)
    vocab_size = len(model.wv.vocab.items())
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    embedding_matrix = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():
        if key is not None:
            embedding_matrix[value] = model[key]
    return vocab_size, embedding_matrix


def data_word2vec(input_file, num_labels, word2vec_model):
    """
    Create the research data tokenindex based on the word2vec model file.
    Return the class Data (includes the data tokenindex and data labels).

    Args:
        input_file: The research data
        num_labels: The number of classes
        word2vec_model: The word2vec model file
    Returns:
        The class Data (includes the data tokenindex and data labels)
    Raises:
        IOError: If the input file is not the .json file
    """
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def _token_to_index(content):
        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    def _create_onehot_labels(labels_index):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label

    if not input_file.endswith('.json'):
        raise IOError("✘ The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    with open(input_file) as fin:
        testid_list = []
        content_index_list = []
        labels_list = []
        onehot_labels_list = []
        labels_num_list = []
        total_line = 0

        for eachline in fin:
            data = json.loads(eachline)
            testid = data['testid']
            features_content = data['features_content']
            labels_index = data['labels_index']
            labels_num = data['labels_num']

            testid_list.append(testid)
            content_index_list.append(_token_to_index(features_content))
            labels_list.append(labels_index)
            onehot_labels_list.append(_create_onehot_labels(labels_index))
            labels_num_list.append(labels_num)
            total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def testid(self):
            return testid_list

        @property
        def tokenindex(self):
            return content_index_list

        @property
        def labels(self):
            return labels_list

        @property
        def onehot_labels(self):
            return onehot_labels_list

        @property
        def labels_num(self):
            return labels_num_list

    return _Data()


def data_augmented(data, drop_rate=1.0):
    """
    Data augmented.

    Args:
        data: The Class Data()
        drop_rate: The drop rate
    Returns:
        aug_data
    """
    aug_num = data.number
    aug_testid = data.testid
    aug_tokenindex = data.tokenindex
    aug_labels = data.labels
    aug_onehot_labels = data.onehot_labels
    aug_labels_num = data.labels_num

    for i in range(len(data.tokenindex)):
        data_record = data.tokenindex[i]
        if len(data_record) == 1:  # 句子长度为 1，则不进行增广
            continue
        elif len(data_record) == 2:  # 句子长度为 2，则交换两个词的顺序
            data_record[0], data_record[1] = data_record[1], data_record[0]
            aug_testid.append(data.testid[i])
            aug_tokenindex.append(data_record)
            aug_labels.append(data.labels[i])
            aug_onehot_labels.append(data.onehot_labels[i])
            aug_labels_num.append(data.labels_num[i])
            aug_num += 1
        else:
            data_record = np.array(data_record)
            for num in range(len(data_record) // 10):  # 打乱词的次数，次数即生成样本的个数；次数根据句子长度而定
                # random shuffle & random drop
                data_shuffled = np.random.permutation(np.arange(int(len(data_record) * drop_rate)))
                new_data_record = data_record[data_shuffled]

                aug_testid.append(data.testid[i])
                aug_tokenindex.append(list(new_data_record))
                aug_labels.append(data.labels[i])
                aug_onehot_labels.append(data.onehot_labels[i])
                aug_labels_num.append(data.labels_num[i])
                aug_num += 1

    class _AugData:
        def __init__(self):
            pass

        @property
        def number(self):
            return aug_num

        @property
        def testid(self):
            return aug_testid

        @property
        def tokenindex(self):
            return aug_tokenindex

        @property
        def labels(self):
            return aug_labels

        @property
        def onehot_labels(self):
            return aug_onehot_labels

        @property
        def labels_num(self):
            return aug_labels_num

    return _AugData()


def load_data_and_labels(data_file, num_labels, embedding_size, data_aug_flag):
    """
    Load research data from files, splits the data into words and generates labels.
    Return split sentences, labels and the max sentence length of the research data.

    Args:
        data_file: The research data
        num_labels: The number of classes
        embedding_size: The embedding size
        data_aug_flag: The flag of data augmented
    Returns:
        The class Data
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    # Load word2vec model file
    if not os.path.isfile(word2vec_file):
        create_word2vec_model(embedding_size, TEXT_DIR)

    model = word2vec.Word2Vec.load(word2vec_file)

    # Load data from files and split by words
    data = data_word2vec(input_file=data_file, num_labels=num_labels, word2vec_model=model)
    if data_aug_flag:
        data = data_augmented(data)

    # plot_seq_len(data_file, data)

    return data


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Return the padded data and data labels.

    Args:
        data: The research data
        pad_seq_len: The max sentence length of research data
    Returns:
        pad_seq: The padded data
        labels: The data labels
    """
    pad_seq = pad_sequences(data.tokenindex, maxlen=pad_seq_len, value=0.)
    onehot_labels = data.onehot_labels
    return pad_seq, onehot_labels


def plot_seq_len(data_file, data, percentage=0.98):
    """
    Visualizing the sentence length of each data sentence.

    Args:
        data_file: The data_file
        data: The class Data (includes the data tokenindex and data labels)
        percentage: The percentage of the total data you want to show
    """
    data_analysis_dir = '../data/data_analysis/'
    if 'train' in data_file.lower():
        output_file = data_analysis_dir + 'Train Sequence Length Distribution Histogram.png'
    if 'validation' in data_file.lower():
        output_file = data_analysis_dir + 'Validation Sequence Length Distribution Histogram.png'
    if 'test' in data_file.lower():
        output_file = data_analysis_dir + 'Test Sequence Length Distribution Histogram.png'
    result = dict()
    for x in data.tokenindex:
        if len(x) not in result.keys():
            result[len(x)] = 1
        else:
            result[len(x)] += 1
    freq_seq = [(key, result[key]) for key in sorted(result.keys())]
    x = []
    y = []
    avg = 0
    count = 0
    border_index = []
    for item in freq_seq:
        x.append(item[0])
        y.append(item[1])
        avg += item[0] * item[1]
        count += item[1]
        if count > data.number * percentage:
            border_index.append(item[0])
    avg = avg / data.number
    print('The average of the data sequence length is {0}'.format(avg))
    print('The recommend of padding sequence length should more than {0}'.format(border_index[0]))
    xlim(0, 400)
    plt.bar(x, y)
    plt.savefig(output_file)
    plt.close()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data
        batch_size: The size of the data batch
        num_epochs: The number of epochs
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
