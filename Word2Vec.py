from collections import defaultdict
from numpy import exp, log, dot, zeros, random, float32 as REAL, uint32, vstack, sqrt, empty, ones
from types import GeneratorType
from six import iteritems
import math
import numpy as np

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr, data

from Vocab import Vocab, KeyedVectors
from utils import dotp, exponential_map


MAX_WORDS_IN_BATCH = 1000
MAX_EXP = 9
EXP_TABLE_SIZE = 1000
MAX_SENTENCE_LENGTH = 80


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

        expTable = []
        for i in range(0, EXP_TABLE_SIZE):
            temp = exp((i / float(EXP_TABLE_SIZE * 2 - 1)) * MAX_EXP)
            expTable.append(temp / (temp + 1))

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
                    gpar = ro.r["slot"](model, "gpar")
                    map0 = ro.r["slot"](model, "map")
                    z = ro.r["slot"](model, "z")
                    print("model end")
                sentence_length = 0

                continue

        return self.wv.syn0, gpar, z, t
