from math import sinh, cosh
from numpy import dot, sqrt
import numpy as np
from scipy.special import jn


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
