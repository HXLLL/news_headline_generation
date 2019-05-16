import os
import math
import operator
from functools import reduce

def clip_count(cand_d, ref_ds):
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count

def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        ref_counts = []
        ref_lengths = []
        for reference in references:
            words = reference[si]
            ngram_d = {}
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            for i in range(limits):
                ngram = ' '.join(words[i:i + n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        words = candidate[si]
        cand_dict = {}
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    return pr

def best_length_match(ref_l, cand_l):
    least_diff = abs(cand_l - ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l - ref) < least_diff:
            least_diff = abs(cand_l - ref)
            best = ref
    return best

def bp(candidate,references):
    r = 0
    c = 0
    for si in range(len(candidate)):
        ref_lengths = []
        for reference in references:
            words = reference[si]
            ref_lengths.append(len(words))
        words = candidate[si]
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if c>r: return 1
    else: return math.exp(1-(float(r)/c))

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(candidate, references):
    precisions = []
    for i in range(4):
        pr = count_ngram(candidate, references, i + 1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp(candidate,references)
    return bleu

def calc(pred,truth,IDeos):
    A=[ ]
    B=[ ]
    for i in range(len(pred)):
        A.append([ ])
        for j in range(len(pred[i])):
            if(pred[i][j]!=IDeos): A[i].append(str(pred[i][j]))
            else: break
    for i in range(len(truth)):
        B.append([ ])
        for j in range(len(truth[i])):
            if(truth[i][j]!=IDeos): B[i].append(str(truth[i][j]))
            else: break
    candidate = A
    references = [B]
    return BLEU(candidate,references)
