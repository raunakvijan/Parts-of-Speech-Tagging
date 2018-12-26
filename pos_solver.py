###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math
from collections import defaultdict
import copy
from collections import Counter
from collections import deque
import sys


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:

    initial_tag = defaultdict(lambda :0)
    transition_prob = defaultdict(lambda :0)
    emission_prob = defaultdict(lambda :0)
    word_count = 0
    tag_count = defaultdict(lambda :0)

    long_trans = defaultdict(lambda: 0)
    trans_complex = defaultdict(lambda: 0)

    def posterior(self, model, sentence, label):
        if model == "Simple":
            return -999
        elif model == "Complex":
            return -999
        elif model == "HMM":
            return -999
        else:
            print("Unknown algo!")

    def get_long_trans(self, data):

        for i in range(len(data)):
            for j in range(len(data[i][1])):
                if j<len(data[i][1])-1:
                    self.trans_complex[(data[i][1][j + 1], data[i][1][j])] += 1
                    if j<len(data[i][1])-2:
                        self.long_trans[(data[i][1][j + 2], data[i][1][j])] += 1
        '''
        for t1 in self.long_trans:
            self.long_trans[t1] /= self.tag_count[t1[1]]
        for t1 in self.trans_complex:
            self.trans_complex[t1] /= self.tag_count[t1[1]]
        '''
        tags = list(self.tag_count.keys())
        for tag in tags:
            sum = 0
            for t1 in self.trans_complex:
                if t1[1] == tag:
                    sum += self.trans_complex[t1]

            for t2 in self.long_trans:
                if t2[1] == tag:
                    sum += self.long_trans[t2]

            for t2 in self.long_trans:
                if t2[1] == tag:
                    self.long_trans[t2] /= sum
            for t1 in self.trans_complex:
                if t1[1] == tag:
                    self.trans_complex[t1] /= sum
        '''
        sum = 0
        for t1 in self.trans_complex:
            if t1[1] == 'noun':
                sum += self.trans_complex[t1]
        for t2 in self.long_trans:
            if t2[1] == 'noun':
                sum += self.long_trans[t2]
        print(sum)
        '''

    def get_all_prob(self, data):
        vocab = set()
        for i in range(len(data)):
            self.initial_tag[data[i][1][0]] += 1
            for j in range(len(data[i][1])):
                vocab.add(data[i][0][j])
                self.tag_count[data[i][1][j]] += 1
                if j < len(data[i][1])-1:
                    self.transition_prob[(data[i][1][j + 1], data[i][1][j])] += 1
                self.emission_prob[(data[i][0][j], data[i][1][j])] += 1
        #print(self.transition_prob)
        self.word_count = len(vocab)
        for tuple in self.emission_prob:
            self.emission_prob[tuple] /= self.tag_count[tuple[1]]
        for tuple in self.transition_prob:
            self.transition_prob[tuple] /= self.tag_count[tuple[1]]
        t_i = len(data)
        for pos in self.initial_tag:
            self.initial_tag[pos] = self.initial_tag[pos]/t_i

    def train(self, data):
        self.get_all_prob(data)
        self.get_long_trans(data)

    def simplified(self, sentence):
        total = sum(self.tag_count.values())
        tags = []
        for w in sentence:
            maxProb = -1 * math.inf
            state = str()
            for s in self.initial_tag:
                ratio = (float(self.tag_count[s]) / total)
                if (w, s) in self.emission_prob:
                    p = self.emission_prob[(w, s)] * ratio
                else:
                    p = 1e-20 * ratio

                if p > maxProb:
                    maxProb = p
                    state = s
            tags.append(state)
        return tags

    def generate_sample(self, sentence, sample):
        sentence_len = len(sentence)
        tags = list(self.tag_count.keys())
        for index in range(sentence_len):
            word = sentence[index]
            probabilities = [0] * len(self.tag_count)

            s_1 = sample[index - 1] if index > 0 else " "
            s_3 = sample[index + 1] if index < sentence_len - 1 else " "

            for j in range(len(self.tag_count)):  # try by assigning every tag
                s_2 = tags[j]
                ep = self.emission_prob[(word, s_2)] if (word, s_2) in self.emission_prob else 0.00000000001
                j_k = self.trans_complex[(s_3, s_2)] if (s_3, s_2) in self.trans_complex else 0.00000000001
                i_j = self.trans_complex[(s_2,s_1)] if (s_2,s_1) in self.trans_complex else 0.00000000001
                i_k = self.long_trans[(s_3, s_1)] if (s_3,s_1) in self.long_trans else 0.00000000001

                if index == 0:
                    probabilities[j] = j_k * ep * self.tag_count[s_2]/self.word_count
                elif index == sentence_len - 1:
                    probabilities[j] = i_j * ep * self.tag_count[s_1]/self.word_count
                else:
                    probabilities[j] = i_j * j_k * i_k * ep * self.tag_count[s_2]/self.word_count

            s = sum(probabilities)
            probabilities = [x / s for x in probabilities]
            rand = random.random()
            p_sum = 0
            for i in range(len(probabilities)):
                p = probabilities[i]
                p_sum += p
                if rand < p_sum:
                    sample[index] = tags[i]
                    break

        return sample

    def mcmc(self, sentence, sample_count):
        sample = self.hmm_viterbi(sentence)
        #sample = ["noun"] * len(sentence)
        samples=list()
        samples.append(sample)
        for p in range(sample_count+10):
            sample = self.generate_sample(sentence, sample)
            samples.append(sample)
        return samples[10:]

    def max_marginal(self, sentence):
        sample_count = 100
        samples = self.mcmc(sentence, sample_count)
        final_sample = []

        for i in range(len(sentence)):
            tag_count = dict.fromkeys(self.tag_count.keys(), 0)
            for sample in samples:
                tag_count[sample[i]] += 1
            final_sample.append(max(tag_count, key=tag_count.get))
        return final_sample

    def hmm_viterbi(self, sentence):
        v = list()
        viterbi = defaultdict(lambda: list())
        for tag in self.initial_tag:
            if (sentence[0], tag) in self.emission_prob:
                viterbi[tag] = [self.initial_tag[tag] * self.emission_prob[(sentence[0], tag)], tag]
            else:
                viterbi[tag] = [1e-20, tag]
        v.append(viterbi)

        for i in range(1, len(sentence)):
            viterbi = defaultdict(lambda: list())
            for j in self.initial_tag:
                max = -9999999999
                max_tag = j
                for k in self.initial_tag:
                    p = 1e-20
                    if (j, k) in self.transition_prob:
                        p = v[i - 1][k][0] * self.transition_prob[(j, k)]
                    if max < p and p != 1e-20:
                        max = p
                        max_tag = k
                if max == -9999999999:
                    max = 1e-20

                if (sentence[i], j) in self.emission_prob:
                    viterbi[j] = [max * self.emission_prob[(sentence[i], j)], max_tag]
                else:
                    viterbi[j] = [(max * 1e-20), max_tag]
            v.append(viterbi)

        result = deque()
        max = -99999999999
        max_tag = str()
        last_tag = str()
        for i in v[-1]:
            p, tag = v[-1][i]
            if max < p:
                max, max_tag = v[-1][i]
                last_tag = i
        if len(sentence) > 1:
            result.appendleft(last_tag)
        result.appendleft(max_tag)

        for i in range(len(v) - 2, 0, -1):
            result.appendleft(v[i][max_tag][1])
            max_tag = v[i][max_tag][1]
        return result

    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.max_marginal(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")