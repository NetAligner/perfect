from six.moves import range
import sys
import argparse
import time

from Word2Vec import Word2Vec
from utils import read_file, build_link, gradient


MAX_EXP = 9
EXP_TABLE_SIZE = 1000
MAX_WORDS_IN_BATCH = 1000


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

    process(args)


if __name__ == "__main__":
    sys.exit(main())
