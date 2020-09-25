# Deep Learning for Multi-Label Text Classification

[![Python Version](https://img.shields.io/badge/language-python3.6-blue.svg)](https://www.python.org/downloads/) [![Build Status](https://travis-ci.org/RandolphVI/Multi-Label-Text-Classification.svg?branch=master)](https://travis-ci.org/RandolphVI/Multi-Label-Text-Classification) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/c45aac301b244316830b00b9b0985e3e)](https://www.codacy.com/app/chinawolfman/Multi-Label-Text-Classification?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=RandolphVI/Multi-Label-Text-Classification&amp;utm_campaign=Badge_Grade) [![License](https://img.shields.io/github/license/RandolphVI/Multi-Label-Text-Classification.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Issues](https://img.shields.io/github/issues/RandolphVI/Multi-Label-Text-Classification.svg)](https://github.com/RandolphVI/Multi-Label-Text-Classification/issues)

This repository is my research project, and it is also a study of TensorFlow, Deep Learning (Fasttext, CNN, LSTM, etc.).

The main objective of the project is to solve the multi-label text classification problem based on Deep Neural Networks. Thus, the format of the data label is like [0, 1, 0, ..., 1, 1] according to the characteristics of such a problem.

## Requirements

- Python 3.6
- Tensorflow 1.15.0
- Tensorboard 1.15.0
- Sklearn 0.19.1
- Numpy 1.16.2
- Gensim 3.8.3
- Tqdm 4.49.0

## Project

The project structure is below:

```text
.
├── Model
│   ├── test_model.py
│   ├── text_model.py
│   └── train_model.py
├── data
│   ├── word2vec_100.model.* [Need Download]
│   ├── Test_sample.json
│   ├── Train_sample.json
│   └── Validation_sample.json
└── utils
│   ├── checkmate.py
│   ├── data_helpers.py
│   └── param_parser.py
├── LICENSE
├── README.md
└── requirements.txt
```



## Innovation

### Data part
1. Make the data support **Chinese** and English (Can use `jieba` or `nltk` ).
2. Can use **your pre-trained word vectors** (Can use `gensim`). 
3. Add embedding visualization based on the **tensorboard** (Need to create `metadata.tsv` first).

### Model part
1. Add the correct **L2 loss** calculation operation.
2. Add **gradients clip** operation to prevent gradient explosion.
3. Add **learning rate decay** with exponential decay.
4. Add a new **Highway Layer** (Which is useful according to the model performance).
5. Add **Batch Normalization Layer**.

### Code part
1. Can choose to **train** the model directly or **restore** the model from the checkpoint in `train.py`.
2. Can predict the labels via **threshold** and **top-K** in `train.py` and `test.py`.
3. Can calculate the evaluation metrics --- **AUC** & **AUPRC**.
4. Can create the prediction file which including the predicted values and predicted labels of the Testset data in `test.py`.
5. Add other useful data preprocess functions in `data_helpers.py`.
6. Use `logging` for helping to record the whole info (including **parameters display**, **model training info**, etc.).
7. Provide the ability to save the best n checkpoints in `checkmate.py`, whereas the `tf.train.Saver` can only save the last n checkpoints.

## Data

See data format in `/data` folder which including the data sample files. For example:

```json
{"testid": "3935745", "features_content": ["pore", "water", "pressure", "metering", "device", "incorporating", "pressure", "meter", "force", "meter", "influenced", "pressure", "meter", "device", "includes", "power", "member", "arranged", "control", "pressure", "exerted", "pressure", "meter", "force", "meter", "applying", "overriding", "force", "pressure", "meter", "stop", "influence", "force", "meter", "removing", "overriding", "force", "pressure", "meter", "influence", "force", "meter", "resumed"], "labels_index": [526, 534, 411], "labels_num": 3}
```

- **"testid"**: just the id.
- **"features_content"**: the word segment (after removing the stopwords)
- **"labels_index"**: The label index of the data records.
- **"labels_num"**: The number of labels.

### Text Segment

1. You can use `nltk` package if you are going to deal with the English text data.

2. You can use `jieba` package if you are going to deal with the Chinese text data.

### Data Format

This repository can be used in other datasets (text classification) in two ways:
1. Modify your datasets into the same format of [the sample](https://github.com/RandolphVI/Multi-Label-Text-Classification/blob/master/data).
2. Modify the data preprocessing code in `data_helpers.py`.

Anyway, it should depend on what your data and task are.

**🤔Before you open the new issue about the data format, please check the `data_sample.json` and read the other open issues first, because someone maybe ask me the same question already. For example:**

- [输入文件的格式是什么样子的？](https://github.com/RandolphVI/Multi-Label-Text-Classification/issues/1)
- [Where is the dataset for training?](https://github.com/RandolphVI/Multi-Label-Text-Classification/issues/7)
- [在 data_helpers.py 中的 content.txt 与 metadata.tsv 是什么，具体格式是什么，能否提供一个样例？](https://github.com/RandolphVI/Multi-Label-Text-Classification/issues/12)

### Pre-trained Word Vectors

**You can download the [Word2vec model file](https://drive.google.com/file/d/1S33iejwuQOIaNQfXW7fA_6zBwHHClT--/view?usp=sharing) (dim=100). Make sure they are unzipped and under the `/data` folder.**

You can pre-training your word vectors (based on your corpus) in many ways:
- Use `gensim` package to pre-train data.
- Use `glove` tools to pre-train data.
- Even can use a **fasttext** network to pre-train data.

## Usage

See [Usage](https://github.com/RandolphVI/Multi-Label-Text-Classification/blob/master/Usage.md).

## Network Structure

### FastText

![](https://farm2.staticflickr.com/1917/45609842012_30f370a0ee_o.png)

References:

- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

---

### TextANN

![](https://farm2.staticflickr.com/1965/44745949305_50f831a579_o.png)

References:

- **Personal ideas 🙃**

---

### TextCNN

![](https://farm2.staticflickr.com/1927/44935475604_1d6b8f71a3_o.png)

References:

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

---

### TextRNN

**Warning: Model can use but not finished yet 🤪!**

![](https://farm2.staticflickr.com/1925/30719666177_6665038ea2_o.png)

#### TODO
1. Add BN-LSTM cell unit.
2. Add attention.

References:

- [Recurrent Neural Network for Text Classification with Multi-Task Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

---

### TextCRNN

![](https://farm2.staticflickr.com/1915/43842346360_e4660c5921_o.png)

References:

- **Personal ideas 🙃**

---

### TextRCNN

![](https://farm2.staticflickr.com/1950/31788031648_b5cba7bbf0_o.png)

References:

- **Personal ideas 🙃**

---

### TextHAN

References:

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

---

### TextSANN

**Warning: Model can use but not finished yet 🤪!**

#### TODO
1. Add attention penalization loss.
2. Add visualization.

References:

- [A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130.pdf)

---

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Ph.D.

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
