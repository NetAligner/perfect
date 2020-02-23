import numpy as np
from scipy.special import jn

from six import string_types, iteritems
from six.moves import range

import sys
from collections import defaultdict
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL, \
    uint32, seterr, array, uint8, vstack, fromstring, sqrt, \
    empty, sum as np_sum, ones, logaddexp
from types import GeneratorType
import itertools
from math import sinh, cosh
import math
import argparse
import time

MAX_EXP = 9
EXP_TABLE_SIZE = 1000
MAX_SENTENCE_LENGTH = 80

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, data


class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class KeyedVectors(object):
    """
    Base class to contain vectors and vocab for any set of vectors which are each associated with a key.
    """

    def __init__(self):
        self.syn0 = []
        self.vocab = {}
        self.index2word = []
        self.vector_size = None
        self.syn0norm = None

    def wv(self):
        return self

    def save(self, *args, **kwargs):
        # don't bother storing the cached normalized vectors
        kwargs['ignore'] = kwargs.get('ignore', ['syn0norm'])
        self.save(*args, **kwargs)


# def save_word2vec_format(self, fname, fvocab=None, binary=False, total_vec=None):

MAX_WORDS_IN_BATCH = 1000
"""迭代读取句子"""


class read_word(object):
    """Iterate over sentences from the Brown corpus (part of NLTK data)."""

    def __init__(self, source, max_sentence_length=MAX_WORDS_IN_BATCH, limit=None):
        self.source = source
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        with open(self.source, 'r') as fin:
            for line in itertools.islice(fin, self.limit):
                line = line.split()
                # print(line)
                i = 0
                while i < len(line):
                    yield line[i: i + self.max_sentence_length]
                    i += self.max_sentence_length


class Word2Vec(object):

    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
                 max_vocab_size=None, sample=1e-3, seed=1,
                 negative=5, hashfxn=hash, iter=100, null_word=0,
                 trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH):
        self.initialize_word_vectors()

        # self.sg = int(sg)
        self.cum_table = None  # for negative sampling
        self.vector_size = int(size)
        self.layer1_size = int(size)
        self.alpha = float(alpha)
        self.min_alpha_yet_reached = float(alpha)  # To warn user if alpha increases
        self.window = int(window)
        self.max_vocab_size = max_vocab_size
        self.seed = seed
        self.random = random.RandomState(seed)
        self.min_count = min_count
        self.sample = sample
        # self.workers = int(workers)
        # self.min_alpha = float(min_alpha)
        # self.hs = hs
        self.negative = negative
        # self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.iter = iter
        self.null_word = null_word
        self.train_count = 1
        self.total_train_time = 0
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words
        self.model_trimmed_post_training = False
        # self.compute_loss = compute_loss
        self.running_training_loss = 0
        if sentences is not None:
            if isinstance(sentences, GeneratorType):
                raise TypeError("You can't pass a generator as the sentences argument. Try an iterator.")
            self.build_vocab(sentences, trim_rule=trim_rule)
            '''self.train(
                sentences, total_examples=self.corpus_count, epochs=self.iter,
                start_alpha=self.alpha
            )
            '''

    def initialize_word_vectors(self):
        self.wv = KeyedVectors()

    def make_cum_table(self, power=0.75, domain=2 ** 31 - 1):
        vocab_size = len(self.wv.index2word)
        self.cum_table = zeros(vocab_size, dtype=uint32)
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0
        for word_index in range(vocab_size):
            train_words_pow += self.wv.vocab[self.wv.index2word[word_index]].count ** power
        cumulative = 0.0
        i = 0
        for word_index in range(vocab_size):
            self.cum_table[word_index] = i
            if word_index / float(len(self.wv.index2word)) > train_words_pow:
                i = i + 1
                train_words_pow += pow(self.wv.vocab[self.wv.index2word[word_index]].count, power) / float(
                    train_words_pow)
            if i >= vocab_size:
                i = vocab_size - 1
        # if len(self.cum_table) > 0:
        # assert self.cum_table[-1] == domain

    def build_vocab(self, sentences, keep_raw_vocab=False, trim_rule=None, progress_per=10000, update=False):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of unicode strings.
        """
        print("build------------------")
        self.scan_vocab(sentences, progress_per=progress_per, trim_rule=trim_rule)  # initial survey
        # trim by min_count & precalculate downsampling
        self.scale_vocab(trim_rule=trim_rule, update=update)
        self.finalize_vocab(update=update)

    """对单词进行初始化"""

    def scan_vocab(self, sentences, progress_per=10000, trim_rule=None):
        """Do an initial scan of all words appearing in sentences."""
        sentence_no = -1
        total_words = 0
        min_reduce = 1
        vocab = defaultdict(int)
        checked_string_types = 0
        for sentence_no, sentence in enumerate(sentences):
            '''if not checked_string_types:
                if isinstance(sentence, string_types):
                     logger.warning(
                        "Each 'sentences' item should be a list of words (usually unicode strings). "
                        "First item here is instead plain %s.",
                        type(sentence)
                    )
                checked_string_types += 1'''

            for word in sentence:
                # print(word)
                vocab[word] += 1
                # print(vocab)
            total_words += len(sentence)

            '''if self.max_vocab_size and len(vocab) > self.max_vocab_size:
                    utils.prune_vocab(vocab, min_reduce, trim_rule=trim_rule)  # prune_vocab？？？？
                    min_reduce += 1'''
        self.corpus_count = sentence_no + 1
        self.raw_vocab = vocab
        return total_words

    def scale_vocab(self, min_count=1, sample=None,
                    trim_rule=None, update=False):
        min_count = self.min_count
        print("min_count ----------->", min_count)
        sample = sample or self.sample
        drop_total = drop_unique = 0
        if not update:
            retain_total, retain_words = 0, []
            self.wv.index2word = []
            # make stored settings match these applied settings
            self.min_count = min_count
            self.sample = sample
            self.wv.vocab = {}
            for word, v in iteritems(self.raw_vocab):
                # print(self.raw_vocab)
                if self.raw_vocab[word] > min_count:
                    retain_words.append(word)
                    retain_total += v
                    self.wv.vocab[word] = Vocab(count=v, index=len(self.wv.index2word))
                    self.wv.index2word.append(word)
        else:
            new_total = pre_exist_total = 0
            new_words = pre_exist_words = []
            for word, v in iteritems(self.raw_vocab):
                if self.wv.vocab[word].count > min_count:
                    if word in self.wv.vocab:
                        pre_exist_words.append(word)
                        pre_exist_total += v
                        self.wv.vocab[word].count += v
                    else:
                        new_words.append(word)
                        new_total += v
                        self.wv.vocab[word] = Vocab(count=v, index=len(self.wv.index2word))
                        self.wv.index2word.append(word)

    def finalize_vocab(self, update=False):
        if not self.wv.index2word:
            self.scale_vocab()
        if self.negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()
        if not update:
            self.reset_weights()
        else:
            self.update_weights()

    def sort_vocab(self):
        """Sort the vocabulary so the most frequent words have the lowest indexes."""
        if len(self.wv.syn0):
            raise RuntimeError("cannot sort vocabulary after model weights already initialized.")
        self.wv.index2word.sort(key=lambda word: self.wv.vocab[word].count, reverse=True)
        for i, word in enumerate(self.wv.index2word):
            self.wv.vocab[word].index = i

        """初始化训练向量"""

    def reset_weights(self):
        """Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary."""
        self.wv.syn0 = empty((len(self.wv.vocab), self.vector_size), dtype=REAL)
        # randomize weights vector by vector, rather than materializing a huge random matrix in RAM at once
        for i in range(len(self.wv.vocab)):
            r = (np.random.random(1) - 0.5) / self.vector_size
            theta = np.random.random(self.vector_size - 1) * 2 * math.pi
            rand_vector = []
            for j in range(self.vector_size):
                x = 1
                for k in range(j):
                    x *= math.sin(theta[k])
                if j != self.vector_size - 1:
                    x *= math.cos(theta[j])
                rand_vector.append(x)
            self.wv.syn0[i] = rand_vector

        '''for i in range(len(self.wv.vocab)):
            # construct deterministic seed from word AND seed argument
            self.wv.syn0[i] = self.seeded_vector(self.wv.index2word[i] + str(self.seed))'''
        # if self.hs:
        #   self.syn1 = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
        if self.negative:
            self.syn1neg = zeros((len(self.wv.vocab), self.layer1_size), dtype=REAL)
        self.wv.syn0norm = None

        self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)  # zeros suppress learning

    def update_weights(self):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        gained_vocab = len(self.wv.vocab) - len(self.wv.syn0)
        newsyn0 = empty((gained_vocab, self.vector_size), dtype=REAL)

        # randomize the remaining words
        for i in range(len(self.wv.syn0), len(self.wv.vocab)):
            # construct deterministic seed from word AND seed argument
            newsyn0[i - len(self.wv.syn0)] = self.seeded_vector(self.wv.index2word[i] + str(self.seed))

        # Raise an error if an online update is run before initial training on a corpus
        if not len(self.wv.syn0):
            raise RuntimeError(
                "You cannot do an online vocabulary-update of a model which has no prior vocabulary. "
                "First build the vocabulary of your model with a corpus before doing an online update."
            )

        self.wv.syn0 = vstack([self.wv.syn0, newsyn0])

        if self.hs:
            self.syn1 = vstack([self.syn1, zeros((gained_vocab, self.layer1_size), dtype=REAL)])
        if self.negative:
            self.syn1neg = vstack([self.syn1neg, zeros((gained_vocab, self.layer1_size), dtype=REAL)])
        self.wv.syn0norm = None

        # do not suppress learning for already learned words
        self.syn0_lockf = ones(len(self.wv.vocab), dtype=REAL)  # zeros suppress learning

    def seeded_vector(self, seed_string):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = random.RandomState(self.hashfxn(seed_string) & 0xffffffff)
        return (once.rand(self.vector_size) - 0.5) / self.vector_size

    def train(self, model_, Gpar, Z, sentences, G, link, word_num=0, total_examples=None, total_words=None,
              epochs=iter, start_alpha=None, end_alpha=None, word_count=0,
              queue_factor=2, report_delay=1.0, compute_loss=None, negative=10):
        print("Training...")
        starting_alpha = start_alpha or self.alpha
        word = 0
        sentence_length = 0
        sentence_position = 0
        word_count = 0
        last_word_count = 0
        sen = []
        deriv_syn0_tmp_arr = [0] * self.layer1_size
        deriv_syn1neg_tmp_arr = [0] * self.layer1_size
        deriv_syn0_poincare = [0] * self.layer1_size
        next_random = 1
        word_count_actual = 0
        local_iter = epochs
        line = []
        t = word_num
        for sentence in sentences:
            for w in sentence:
                line.append(w)

        G = G
        if t == 0:
            numpy2ri.activate()
            MixGHD = importr("MixGHD")
            B = np.array(self.wv.syn0)
            model = MixGHD.MGHD(data=B, G=G)
            gpar = ro.r["slot"](model, "gpar")
            z = ro.r["slot"](model, "z")
        else:
            gpar = Gpar
            z = Z

        while True:
            if word_count - last_word_count > 10000:
                word_count_actual += word_count - last_word_count
                last_word_count = word_count
                alpha = starting_alpha * (1 - word_count_actual / float(self.iter * self.train_count + 1))
                if alpha < starting_alpha * 0.0001:
                    alpha = starting_alpha * 0.0001

            if local_iter == 0:
                break

            if sentence_length == 0:
                print("local_iter--------------->", local_iter)
                sen = []
                while True:
                    v = line[t]
                    t = (t + 1) % len(line)
                    i = self.wv.vocab.get(v, -1)

                    if i == -1:
                        continue
                    else:
                        word_count += 1
                        word = self.wv.vocab[v].index

                    if self.sample > 0:
                        ran = float((sqrt(self.wv.vocab[v].count / (self.sample * self.train_count)) + 1) * (
                                self.sample * self.train_count) /
                                    self.wv.vocab[v].count)
                        next_random = next_random * 25214903917 + 11
                        if ran < (next_random & 0xFFFF) / 65536:
                            continue
                        else:
                            sen.append(int(word))
                            sentence_length += 1
                    if sentence_length >= MAX_SENTENCE_LENGTH:
                        break
                sentence_position = 0
            word = sen[sentence_position]
            self.train_count += 1
            neu1 = []
            neu1e = []
            for c in range(0, self.layer1_size):
                neu1.append(0)
            for c in range(0, self.layer1_size):
                neu1e.append(0)
            next_random = next_random * 25214903917 + 11
            b = 0
            for a in range(b, self.window * 2 + 1 - b):
                if a != self.window:
                    c = sentence_position - self.window + a
                    if c < 0:
                        continue
                    if c > len(sen) - 1:
                        continue
                    last_word = int(sen[c])
                    if last_word == -1:
                        print("last_word=-1")
                        continue
                    l1 = last_word
                    for c in range(0, self.layer1_size):  neu1e[c] = 0
                    if negative > 0:
                        for d in range(0, negative):
                            if d == 0:
                                target = word
                                label = 1
                            else:
                                next_random = next_random * 25214903917 + 11
                                target = self.cum_table[np.random.randint(0, len(self.wv.index2word))]
                                if target == 0: target = next_random % (len(self.wv.index2word) - 1) + 1  # ??
                                if target == word:
                                    continue
                                label = 0
                            l2 = target

                            hybolic_sub_norm = 0
                            hybolic_syn0_norm = dotp(self.wv.syn0[l1], self.wv.syn0[l1])
                            hybolic_alpha = 1 - hybolic_syn0_norm
                            if hybolic_alpha < 0.0001:
                                hybolic_alpha = 0.01
                            if hybolic_syn0_norm >= 1.01:
                                print("error: 2 - syn0_norm is %lf\n", hybolic_syn0_norm)
                                for c in range(0, self.layer1_size): print("%f ", self.wv.syn0[c + l1])
                                exit(1)

                            hybolic_syn1neg_norm = dotp(self.syn1neg[l2], self.syn1neg[l2])
                            if hybolic_syn1neg_norm >= 1.01:
                                print("error: 2 - syn1neg_norm is %lf\n", hybolic_syn1neg_norm)
                                exit(1)
                            hybolic_beta = 1 - hybolic_syn1neg_norm
                            if hybolic_beta < 0.0001:
                                hybolic_beta = 0.01
                            for c in range(0, self.layer1_size): hybolic_sub_norm += pow(
                                (self.wv.syn0[l1][c] - self.syn1neg[l2][c]), 2)
                            hybolic_gamma = 1 + 2 * hybolic_sub_norm / hybolic_alpha / hybolic_beta
                            hybolic_distance = log(hybolic_gamma + sqrt(hybolic_gamma * hybolic_gamma - 1))
                            if 0 - hybolic_distance > MAX_EXP:
                                g = (label - 1) * self.alpha
                            elif 0 - hybolic_distance < -MAX_EXP:
                                g = (label - 0) * self.alpha
                            else:
                                g = (label - expTable[
                                    int(((0 - hybolic_distance + MAX_EXP) * (
                                            EXP_TABLE_SIZE / MAX_EXP / 2)))]) * self.alpha
                            deriv_syn0_tmp_cof = dotp(self.wv.syn0[l1], self.syn1neg[l2])
                            deriv_syn0_tmp_cof = (
                                                         hybolic_syn1neg_norm - 2 * deriv_syn0_tmp_cof + 1) / hybolic_alpha / hybolic_alpha
                            for c in range(0, self.layer1_size): deriv_syn0_tmp_arr[c] = self.wv.syn0[l1][
                                                                                             c] * deriv_syn0_tmp_cof
                            for c in range(0, self.layer1_size): deriv_syn1neg_tmp_arr[c] = self.syn1neg[l2][
                                                                                                c] / hybolic_alpha
                            if hybolic_gamma * hybolic_gamma - 1 < 0.00001:
                                hybolic_gamma += 0.001
                            deriv_syn0_tmp_cof = g * 4 / hybolic_beta / sqrt(hybolic_gamma * hybolic_gamma - 1)
                            for c in range(0, self.layer1_size):
                                deriv_syn0_poincare[c] = deriv_syn0_tmp_cof * (
                                        deriv_syn1neg_tmp_arr[c] - deriv_syn0_tmp_arr[c])
                            if deriv_syn0_poincare[0] == 0:
                                for c in range(0, self.layer1_size):
                                    if deriv_syn0_poincare[c] == 0:
                                        deriv_syn0_poincare[c] = 0.00001

                            deriv_syn0_tmp_cof = hybolic_alpha * hybolic_alpha / 4
                            for c in range(0, self.layer1_size): deriv_syn0_poincare[c] = deriv_syn0_tmp_cof * \
                                                                                          deriv_syn0_poincare[c]
                            for c in range(0, self.layer1_size): neu1e[c] += deriv_syn0_poincare[c]
                            deriv_syn0_tmp_cof = 0

                    i = link.get(self.wv.index2word[l1], -1)
                    if i != -1:
                        i_ = model_.wv.vocab.get(i, -1)
                        if i_ != -1:
                            l3 = i_.index
                            map_tmp_arr = model_.wv.syn0[l3]
                            map_arr = exponential_map(map_tmp_arr, neu1e, model_.layer1_size)
                            model_.wv.syn0[l3] = map_arr
                    for k in range(0, G):
                        neu1e += z[k * len(self.wv.vocab) + l1] * dot(gpar[k][0], gpar[k][3])
                    map_tmp_arr = self.wv.syn0[l1]
                    map_arr = exponential_map(map_tmp_arr, neu1e, self.layer1_size)
                    self.wv.syn0[l1] = map_arr

            sentence_position = sentence_position + 1

            if sentence_position >= sentence_length - 1:
                local_iter -= 1
                if local_iter == 0:
                    print('model start')
                    numpy2ri.activate()
                    MixGHD = importr("MixGHD")
                    B = np.array(self.wv.syn0)
                    model = MixGHD.MGHD(data=B, G=G)
                    """返回的需要的list"""
                    gpar = ro.r["slot"](model, "gpar")
                    map0 = ro.r["slot"](model, "map")
                    z = ro.r["slot"](model, "z")
                    print("model end")
                sentence_length = 0

                continue

        return self.wv.syn0, gpar, z, t


def exponential_map(x, v, layer1_size):
    x_tmp_norm = 0
    v_tmp_norm = 0
    tmp_x = [0] * layer1_size
    tmp_v = [0] * layer1_size
    map_vec = [0] * layer1_size
    for c in range(0, layer1_size): tmp_x[c] = float(x[c]);tmp_v[c] = float(v[c])
    for c in range(0, layer1_size): x_tmp_norm += float(x[c] * x[c]); v_tmp_norm += float(v[c] * v[c])
    x_tmp_norm = float(sqrt(x_tmp_norm))
    v_tmp_norm = float(sqrt(v_tmp_norm))
    a = 0.98
    sqrt_a = float(sqrt(a))
    if x_tmp_norm > a:
        for c in range(0, layer1_size): x[c] = sqrt_a * x[c] / x_tmp_norm
    if v_tmp_norm > a:
        for c in range(0, layer1_size): v[c] = sqrt_a * v[c] / v_tmp_norm
    lambda_x = float(0)
    tmp_cof = float(0)
    v_norm = float(0)
    xv_dot = float(0)
    for c in range(0, layer1_size): lambda_x += x[c] * x[c]
    lambda_x = 2 / (1 - lambda_x)
    for c in range(0, layer1_size): v_norm += v[c] * v[c]
    if v_norm < 0.00000000001:
        # print("v_norm = 0\n")
        v_norm = 1.0
    v_norm = sqrt(v_norm)
    for c in range(0, layer1_size): xv_dot += x[c] * v[c] / v_norm
    tmp_cof = lambda_x * (cosh(lambda_x * v_norm) + xv_dot * sinh(lambda_x * v_norm))  # 第一个分子前半部分
    for c in range(0, layer1_size): map_vec[c] = x[c] * tmp_cof
    tmp_cof = sinh(lambda_x * v_norm) / v_norm
    for c in range(0, layer1_size): map_vec[c] += v[c] * tmp_cof
    tmp_cof = 1 + (lambda_x - 1) * cosh(lambda_x * v_norm) + lambda_x * xv_dot * sinh(lambda_x * v_norm)  # 分母
    for c in range(0, layer1_size): map_vec[c] = map_vec[c] / tmp_cof
    tmp_cof = 0
    for c in range(0, layer1_size): tmp_cof += map_vec[c] * map_vec[c];
    tmp_cof = float(sqrt(tmp_cof))
    if tmp_cof >= 1 and tmp_cof < 1.01:
        for c in range(0, layer1_size):
            map_vec[c] = sqrt_a * map_vec[c] / tmp_cof
    return map_vec


random_radius = float(1)


def dotp(v1, v2):
    return float(dot(v1, v2))


def read_file(file):
    fp = open(file, "r")

    sentences = []

    # sentence= fp.readlines()
    for sentence in fp.readlines():
        temp = sentence.strip().strip('[]').split()
        # print("temp----------->",temp)
        sentences.append(temp)
    return sentences


def build_link(file):
    link_arr = read_file(file)
    link = {}
    link1 = {}
    for l in link_arr:
        link[l[0]] = l[1]
        link1[l[1]] = l[0]
    return link, link1


def gradient(syn, index, gpar, d, f):
    C = []
    for i in range(len(syn)):
        theta = np.mat(syn[i]).T
        brackets = []
        pis = gpar[-1]
        for g in range(d):
            para = gpar[g]
            mu = np.mat(para[0]).T
            beta = np.mat(para[1]).T
            delta = np.mat(para[3])
            omega = int(para[2][0])
            r = int(para[2][1])
            zeta = r - d / 2
            delta1 = (theta - mu).T.dot(delta.I).dot(theta - mu)
            delta2 = delta1[0, 0] + omega

            part1_ = beta + zeta / delta2 * theta
            part1 = delta.I.dot(part1_)

            v = omega + beta.T.dot(delta.I).dot(beta)
            v = v[0, 0]
            bessel1 = jn(zeta, np.sqrt(v * delta2))
            bessel2 = jn(zeta - 1, np.sqrt(v * delta2))
            part2_1 = zeta / delta2
            part2_2 = np.sqrt(v / delta2) * (bessel1 / bessel2)
            part2 = (part2_1 + part2_2) * delta.I * theta
            brackets.append(part1 - part2)

        c = np.mat(np.zeros(np.shape(theta)))
        for j in range(d):
            c -= pis[j] * brackets[j]
        C.append(c)
        c = c[0, 0]

    if f == 0:
        file = open("C.txt", 'w')
    else:
        file = open("C1.txt", 'w')
    i = 0
    for c in C:
        (m, n) = np.shape(c)
        file.write(str(index[i]) + '\t')
        for j in range(m):
            file.write(str(c[j, 0]) + '\t')
        file.write('\n')
        i += 1


def process(args):
    starttime = time.time()
    file = args.train
    file1 = args.train1
    link_file = args.link
    alpha = args.alpha
    window = args.window
    negative = args.negative
    iteration = args.iter
    output_file = args.output
    output_file1 = args.output1
    save_vocab_file = args.save_vacab
    layer1_size = args.size
    sample = args.sample
    min_count = args.mini_count

    sentences = read_file(file)
    sentences1 = read_file(file1)

    iter = 60
    link, link1 = build_link(link_file)
    model = Word2Vec(sentences, size=layer1_size, alpha=alpha, window=window, min_count=min_count,
                     max_vocab_size=100000, sample=1e-3, seed=1,
                     negative=negative, hashfxn=hash, iter=iteration, null_word=0,
                     trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH)
    model1 = Word2Vec(sentences1, size=layer1_size, alpha=alpha, window=window, min_count=min_count,
                      max_vocab_size=100000, sample=1e-3, seed=1,
                      negative=negative, hashfxn=hash, iter=iteration, null_word=0,
                      trim_rule=None, sorted_vocab=1, batch_words=MAX_WORDS_IN_BATCH)
    gpar = 0
    z = 0
    gpar1 = 0
    z1 = 0
    G = 60
    G1 = 61
    t = 0
    t1 = 0

    syn01, gpar1, z1, t1 = model1.train(model_=model, sentences=sentences1, Gpar=gpar1, G=G1, link=link1, Z=z,
                                        epochs=iter,
                                        word_num=t1, start_alpha=alpha)
    syn0, gpar, z, t = model.train(model_=model1, sentences=sentences, Gpar=gpar, Z=z, G=G, link=link, epochs=iter,
                                   word_num=t,
                                   start_alpha=alpha)

    for n in range(1, iteration):
        syn0, gpar, map0, z, t = model.train(model_=model1, sentences=sentences, Gpar=gpar, Z=z, G=G, link=link,
                                             epochs=iter, word_num=t, start_alpha=alpha)
        syn01, gpar1, map1, z1, t1 = model1.train(model_=model, sentences=sentences1, Gpar=gpar1, G=G1, link=link1,
                                                  Z=z1, epochs=iter, word_num=t1, start_alpha=alpha)

    index = model.wv.index2word
    index1 = model1.wv.index2word
    gradient(syn0, index, gpar, G, 0)
    gradient(syn01, index1, gpar1, G1, 1)

    fp = open(output_file, 'w')
    fp.write(str(len(model.wv.index2word)) + "  " + str(model.layer1_size) + '\n')
    for a in range(len(model.wv.index2word)):
        fp.write(str(model.wv.index2word[a]) + '\t')
        for b in range(0, model.layer1_size):
            fp.write(str(model.wv.syn0[a][b]) + "      	")
        fp.write('\n')
    fp.close()

    fp = open(output_file1, 'w')
    fp.write(str(len(model1.wv.index2word)) + "  " + str(model1.layer1_size) + '\n')
    for a in range(len(model1.wv.index2word)):
        fp.write(str(model1.wv.index2word[a]) + '\t')
        for b in range(0, model1.layer1_size):
            fp.write(str(model1.wv.syn0[a][b]) + "      	")
        fp.write('\n')
    fp.close()
    endtime = time.time()
    dtime = endtime - starttime
    print("Program running time：%.8s s" % dtime)


# print(sentence)
# word=Word2Vec(sentence)
# word.build_vocab(sentence)
# print(word.wv.index2word)

expTable = []


def distance(syn0, syn1neg):
    hybolic_syn1neg_norm = 0
    hybolic_sub_norm = 0
    hybolic_syn0_norm = dotp(syn0, syn0)
    hybolic_alpha = 1 - hybolic_syn0_norm
    if hybolic_alpha < 0.0001:
        hybolic_alpha = 0.01
    if hybolic_syn0_norm >= 1.01:
        print("error: 2 - syn0_norm is %lf\n", hybolic_syn0_norm)
        for c in range(0, self.layer1_size): print("%f ", syn0)
        exit(1)

        hybolic_syn1neg_norm = dotp(syn1neg * syn1neg)
    if hybolic_syn1neg_norm >= 1.01:
        print("error: 2 - syn1neg_norm is %lf\n", hybolic_syn1neg_norm)
        exit(1)
    hybolic_beta = 1 - hybolic_syn1neg_norm
    if hybolic_beta < 0.0001:
        hybolic_beta = 0.01
    for c in range(0, len(syn0)): hybolic_sub_norm += pow((syn0[c] - syn1neg[c]), 2)
    hybolic_gamma = 1 + 2 * hybolic_sub_norm / hybolic_alpha / hybolic_beta
    distance = log(hybolic_gamma + sqrt(hybolic_gamma * hybolic_gamma - 1))
    return distance


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", dest="debug", default=False,
                        help="drop a debugger if an exception is raised.")

    parser.add_argument('--size', default=2, type=int,
                        help='out size')
    parser.add_argument('--train', required=True,
                        help='Input file')
    parser.add_argument('--train1', required=True,
                        help='Input file')
    parser.add_argument('--link', required=True,
                        help='link file')
    parser.add_argument('--save_vacab',
                        help='Save vocab file')
    parser.add_argument('--read_vocab',
                        help='Read vocab file')
    parser.add_argument('--alpha', default=0.03, type=float,
                        help='learning rate')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--output1', required=True,
                        help='Output representation file')
    parser.add_argument('--window', default=5, type=int,
                        help='Window size of  model.')
    parser.add_argument('--sample', default=1e-5, type=int,
                        help='???')
    parser.add_argument('--negative', default=10, type=int,
                        help='negative')
    parser.add_argument('--iter', default=10, type=int,
                        help='iteration')
    parser.add_argument('--mini_count', default=5, type=int,
                        help='learning rate')

    parser.add_argument('--walk_length', default=40, type=int,
                        help='Length of the random walk started at each node')

    args = parser.parse_args()

    for i in range(0, EXP_TABLE_SIZE):
        temp = exp((i / float(EXP_TABLE_SIZE * 2 - 1)) * MAX_EXP)
        expTable.append(temp / (temp + 1))
    process(args)


if __name__ == "__main__":
    sys.exit(main())
