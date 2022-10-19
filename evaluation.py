
import spacy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import numpy as np
from scipy.special import softmax
from gpt2 import GPT2Scorer
from nltk import metrics
import nltk
import difflib
import segeval
import json
from spacy.tokens import Doc
from spacy.vocab import Vocab
import pandas as pd
Doc.set_extension("topics", default=None, force=True)
Doc.set_extension("overlap", default=None, force=True)
from transitions import MC, MCClusters
import time
import os
CACHE_DIR = "/cs/snapless/oabend/eitan.wagner/cache/"
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
os.environ['HF_METRICS_CACHE'] = CACHE_DIR
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import FNetTokenizer, FNetForNextSentencePrediction
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments

import joblib
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import load_metric
import logging

import lengths
from lengths import LengthEstimators

#abc

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    logging.info("Running on the GPU")
else:
    dev = torch.device("cpu")
    logging.info("Running on the CPU")



# ********************* levenshtein distance - from nltk with changes *******************

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i           # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j           # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2, transpositions=False, cor_matrix=None):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    if cor_matrix is None:

        cor = 1
    else:
        cor = (1 - cor_matrix[c1, c2]) / 2  # to be between 0 and 1
        # cor = (1 - cor_matrix[c1, c2])  # to be between 0 and 2. the average is around 1

    # substitution
    # c = lev[i - 1][j - 1] + (c1 != c2)
    c = lev[i - 1][j - 1] + cor * (c1 != c2)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    if cor_matrix is None:
        lev[i][j] = min(a, b, c, d)
    else:
        lev[i][j] = min(c, d)


def edit_distance(s1, s2, transpositions=False, cor_matrix=None):
    """
    This was modified to take correlations into account!

    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type transpositions: bool
    :rtype int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(lev, i + 1, j + 1, s1, s2, transpositions=transpositions, cor_matrix=cor_matrix)
    return lev[len1][len2]


def gestalt_diff(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()

# ********************* calculate scores *******************

def get_per_sent(doc):
    """
    Get list of sentence segment-end labels.
    :param doc:  a spaCy doc
    :return: list with 1 for last in segment and 0 o.w., for each sentence
    """
    # converts a spacy doc with segments into a list of 0s (for no boundary after sent) and 1s (for last in segment).
    ends = [segment.end for segment in doc.spans["segments"]]
    logging.info(f"num_ends: {len(ends)}")
    # logging.info([i for i, s in enumerate(doc.spans["sents"]) if s.end in ends])
    if doc.spans.get("sents", None) is None:
        return [1 if s.end in ends else 0 for s in doc.sents]
    return [1 if s.end in ends else 0 for s in doc.spans["sents"]]

def accu_scores(pred_doc, gold_doc):
    """
    Calculate accuracy scores for the segmentation.
    :param pred_doc: spacy doc after the model's segmentation
    :param gold_doc: the reference doc
    :return: tuple of (precision, recall, f)
    """
    # assert that sentences are the same
    # logging.info(sum([(s1.start, s1.end) == (s2.start, s2.end) for s1, s2 in zip(pred_doc.spans['sents'], gold_doc.spans['sents'])]))

    y_true, y_pred = get_per_sent(gold_doc), get_per_sent(pred_doc)
    # logging.info(str(y_true))
    # logging.info(str(y_pred))
    logging.info("len true" + str(len(gold_doc.text)) + " " + str(len(gold_doc.spans['sents']))
                 + " " + str(len(gold_doc.spans["segments"])) + " " + str(sum(y_true)))
    logging.info("len pred" + str(len(pred_doc.text)) + " " + str(len(pred_doc.spans['sents']))
                 + " " + str(len(pred_doc.spans["segments"])) + " " + str(sum(y_pred)))
    # logging.info("len pred", len(pred_doc.text), len(list(pred_doc.sents)), len(pred_doc.spans["segments"]))
    # logging.info(y_true)
    # logging.info(y_pred)
    return precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='binary')[:3]

def windowdiff(pred_doc, gold_doc, k=0):
    """
    Calculate the windowdiff cost.
    :param pred_doc: spacy doc after the model's segmentation
    :param gold_doc: the reference doc
    :return: the windowdiff difference (lower is better).
    """
    # k should be half the average segment length
    if k == 0:  # not specified
        k = len(list(gold_doc.sents)) // (2 * len(gold_doc.spans["segments"]))  # use gold_doc average segment len
    y_true, y_pred = "".join([str(_y) for _y in get_per_sent(gold_doc)]), "".join([str(_y) for _y in get_per_sent(pred_doc)]),
    # logging.info(y_true)
    # logging.info(y_pred)
    return nltk.windowdiff(y_true, y_pred, k)

# put in class?
# encoder_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'+'/models/xlnet-large-cased/'
encoder_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'+'models/distilroberta/'
encoder = joblib.load(encoder_path + "label_encoder.pkl")
# with open('/cs/snapless/oabend/eitan.wagner/segmentation/' + 'data/topics5.json', 'r') as infile:
#     topics = json.load(infile)
# encoder = LabelEncoder().fit(topics)
# encoder.classes_ = [c[:-1].strip() if c[-1] == '\xa0' else c.strip() for c in encoder.classes_]

def topics_score(pred_topics, gold_topics, method="gestalt", path='/cs/snapless/oabend/eitan.wagner/segmentation/models/xlnet-large-cased'):
    """
    Calculate cost for the topic list based on the edit distance.
    :param pred_doc: spacy doc after the model's segmentation
    :param gold_doc: the reference doc
    :return: edit distance. lower is better
    """
    # return nltk.edit_distance(encoder.transform(pred_topics), encoder.transform(gold_topics))
    # return nltk.edit_distance(pred_topics, gold_topics)
    if method == "edit":
        # cor_matrix = np.load(path + "/correlation_matrix.npy")
        return edit_distance(pred_topics, gold_topics, transpositions=True, cor_matrix=None)
        # return edit_distance(pred_topics, gold_topics, transpositions=True, cor_matrix=cor_matrix)
    return gestalt_diff(pred_topics, gold_topics)

# ************************ baselines for segmentation **************************

class UniformSegmentor:
    """
    Segmentor for uniform-length segmentation
    """
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')

    def segment(self, text, num_segments):
        """
        Segment a given text
        :param text:
        :param num_segments:
        :return: self
        """
        doc = self.nlp(text)
        doc.spans["sents"] = list(doc.sents)
        return self.segment_doc(doc, num_segments)

    def segment_doc(self, doc, num_segments):
        """
        Segment a given spaCy doc
        :param doc:
        :param num_segments:
        :return: self
        """
        sents = doc.spans["sents"]
        # logging.info("lens: ", len(list(doc.sents)), num_segments)
        sents_arr = np.arange(len(sents))
        # sents = [0 for sent in doc.sents]
        segments = np.array_split(sents_arr, num_segments)
        doc.spans["segments"] = [doc[sents[seg[0]].start:sents[seg[-1]].end] for seg in segments]
        return doc


def dynamic_segmentation(doc, diffs, num_segments, window, alpha=0.2, len_estimator=None):
    # diff is actually a score (i.e. large means connect)
    sents = list(doc.spans["sents"])
    n = len(sents)
    k = num_segments
    # diffs = [0.] * window + [d[0] for d in diffs] + [0.] * window  # large dif is good so used -inf - No! large difference means not to divide!
    diffs = [-np.inf] * window + [d[0] for d in diffs] + [-np.inf] * window  # large diff is good so used -inf - No! large difference means not to divide!
    # diffs = [np.inf] * window + [d[0] for d in diffs] + [np.inf] * window  # large diff is good so used -inf - No! large difference means not to divide!


    bins = 10
    bin_cuts = [(13, 2946),
                (19, 1458),
                (17, 1799),
                (25, 1519),
                (61, 8772),
                (33, 1823),
                (75, 4042),
                (30, 2685),
                (30, 1160),
                (33, 2829)]

    if len_estimator is None:
        print("USING LINEAR PENALTY")
        # print("USING ONLY BIN CUTS!!")
        # print("HERE alpha DOESNT MATTER")
    else:
        print("using bin estimators")
    bin_cuts2 = [(63, 2946),
                 (19, 1215),
                 (69, 1799),
                 (57, 901),
                 (79, 4449),
                 (33, 1229),
                 (75, 1756),
                 (36, 2239),
                 (44, 1160),
                 (71, 2829)]

    L = int(n / k)
    prevs = np.zeros((n, k - 1), dtype=int)

    if len_estimator is not None:
        sent_lens = [0]+[len(s) for s in sents]
        _sent_lens = np.cumsum(sent_lens)
        lens = _sent_lens[:n+1, None] - _sent_lens[:n]
        # len_costs = - np.array([len_estimator.predict(l, 0) for l in lens])
        len_costs = np.zeros_like(lens, dtype=float)
        for i in range(len_costs.shape[0]):
            for j in range(len_costs.shape[1]):
                loc = 0.5 * (i + j) / len_costs.shape[0]
                bin = int(loc*bins)
                if not (bin_cuts2[bin][0] <= lens[i,j] <= bin_cuts2[bin][1]):
                    len_costs[i,j] = np.inf
                    # len_costs[i,j] = 1.
                else:
                    len_costs[i, j] = - len_estimator.predict(lens[i,j], bin)
        # len_costs = - np.log(len_costs)

        # len_costs[len_costs < 150] = 0
        # len_costs[len_costs > 10] = 0

    # costs = np.full((n, k - 1), np.inf)
    # costs[0, 0] = 0.
    costs = np.zeros((n, k - 1))
    costs[0, 1:] = np.inf
    for _n in range(1, n):
        if len_estimator is not None:
            l_costs = len_costs[_n, :_n]
        for _k in range(1, k - 1):
            if len_estimator is None:
                arr = costs[:_n, _k - 1] + alpha * abs((_n - np.arange(_n)) - L) / L  # not  like in the paper!!!
            else:
                arr = costs[:_n, _k - 1] + alpha * l_costs[:_n]  # not  like in the paper!!!
            m = np.argmin(arr)
            costs[_n, _k] = arr[m] - (1-alpha) * diffs[_n]
            prevs[_n, _k] = int(m)

    if len_estimator is None:
        arr = costs[:n, k - 2] + alpha * abs(n - np.arange(n) - L) / L
    else:
        # lens = _sent_lens[n] - _sent_lens[:n]
        # len_costs = - len_estimator.predict(lens, 0)
        # len_costs = np.full_like(lens, np.inf)
        # len_costs[len_costs < 150] = 0
        # len_costs[len_costs > 10] = 0

        arr = costs[:n, k - 2] + alpha * len_costs[n]
    m = np.argmin(arr)

    i = int(m)  # best break for last. These are the beginnings
    doc.spans["segments"] = []
    j = k - 2
    assignment = [i]
    while j > 0:
        i = prevs[i, j]
        j -= 1
        assignment.insert(0, i)
    assignment.insert(0, 0)
    assignment.append(n)  # we should get k+1 in assignment

    for i, j in enumerate(assignment[:-1]):
        doc.spans["segments"].append(doc[sents[j].start:sents[assignment[i + 1] - 1].end])


class Gpt2Segmentor:
    """
    Segmentor based on gpt2 scores
    """
    def __init__(self, with_topics=False, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/',
                 model_name='distilroberta', run_id=None, large=False, xlarge=False, use_nb=False, from_path=None, nb_topics=True):
        from transformer_classification import TransformerClassifier
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')
        gpt_j = True if (not large and xlarge) else False
        self.scorer = GPT2Scorer(large=large, xlarge=xlarge, from_path=from_path, gpt_j=gpt_j)
        self.with_topics = with_topics
        self.run_id = run_id
        self.use_nb = use_nb
        self.nb_topics = nb_topics
        self.len_estimators = None

        if with_topics:
            self.model = TransformerClassifier(base_path=base_path, model_name=model_name, full_lm_scores=False)
            logging.info("Loaded classifier")
            self.cache_path = base_path + "models/" + model_name + "/pred_cache/"
            self.cache = None
            self.cache_id = None
            self.avg_lens = np.array([242.42, 468.88, 282.28, 332.4, 561.43, 428.92, 297.3, 607.48, 392.81, 123.0, 315.0, 377.2,
                          212.4, 403.0, 331.41, 433.0, 197.5, 535.63, 290.8, 674.0,404.16, 175.6, 428.92, 332.69,
                          494.0, 130.0, 332.3, 307.0, 419.6, 1586.0])
        if use_nb:
            if nb_topics:
                print("Using topics nb")
                self.len_estimators = joblib.load(base_path + 'models/' + model_name
                                                  + f'/t_length_estimator{len(self.model.topics)}.pkl')
            else:
                print("Using bin nbs")
                # self.len_estimators = joblib.load(base_path + 'models/' + model_name
                #                                   + '/length_estimator1.pkl')
                self.len_estimators = joblib.load(base_path + 'models/' + model_name
                                                  + '/length_estimator10.pkl')

    def segment(self, text, num_segments, window=1):
        """
        Segment a given text
        :param window: window for gpt2-scorer
        :param text:
        :param num_segments:
        :return: self
        """
        doc = self.nlp(text)
        doc.spans["sents"] = list(doc.sents)
        return self.segment_doc(doc, num_segments, window)

    def _make_segments(self, doc, diffs, num_segments, window=3):
        sents = list(doc.spans["sents"])
        # diffs.sort(reverse=False)  # TODO: should be reversed
        diffs.sort(reverse=True)  #should be reversed since we use the loss
        last_js = sorted([d[1] for d in diffs[:num_segments-1]])

        doc.spans["segments"] = [doc[:sents[last_js[0]-1].end]]
        for i, j in enumerate(last_js[:-1]):
            doc.spans["segments"].append(doc[sents[j].start:sents[last_js[i+1]-1].end:])
        doc.spans["segments"].append(doc[sents[last_js[-1]].start:])

    def _make_segments_dynamic(self, doc, diffs, num_segments, window, alpha):
        sents = list(doc.spans["sents"])
        k = num_segments
        diffs = [0.] * window + [d[0] for d in diffs] + [0.] * window  # large diff is good so used -inf - No! large difference means not to divide!
        n = len(sents)
        L = int(n / k)
        prevs = np.zeros((n, k-1), dtype=int)
        costs = np.zeros((n, k-1))
        costs[0, 1:] = np.inf
        for _n in range(1, n):
            for _k in range(1, k-1):
                arr = costs[:_n, _k-1] + (1 - alpha) * abs((_n - np.arange(_n)) - L) / L  # not like in the paper!!!
                m = np.argmin(arr)
                costs[_n, _k] = arr[m] - alpha * (diffs[_n]/50)  # ??
                prevs[_n, _k] = int(m)

        arr = costs[:n, k-2] + (1 - alpha) * abs(n - np.arange(n) - L) / L
        m = np.argmin(arr)

        i = int(m)  # best break for last. These are the beginnings
        doc.spans["segments"] = []
        j = k - 2
        assignment = [i]
        while j > 0:
            i = prevs[i, j]
            j -= 1
            assignment.insert(0, i)
        assignment.insert(0, 0)
        assignment.append(n)  # we should get k+1 in assignment
        # print("LEN ASSIGNMENT: ", len(assignment))

        for i, j in enumerate(assignment[:-1]):
            doc.spans["segments"].append(doc[sents[j].start:sents[assignment[i+1]-1].end])
        # print("LEN ASSIGNMENT: ", len(doc.spans["segments"]))

    def load_cache(self, t):
        try:
            self.cache = np.load(self.cache_path + f"{str(t)}_r{self.run_id}.npy")
            self.cache_id = t
        except IOError as err:
            # except:
            pass

    def _get_classification_costs(self, sents, t=None):
        if self.cache_id is not None:
            return self.cache
        else:  # calculate probs
            logging.info("Calculating classification scores")
            costs = np.full((len(sents), len(sents)+1, len(encoder.classes_)), np.inf)
            for i, s1 in enumerate(sents):
                # logging.info(i)
                # print(i)
                for j, s2 in enumerate(sents[i:]):
                    # if j < i:
                    #     continue
                    span = s1.doc[s1.start: s2.end]
                    bins = 10
                    loc = 0.5 * (span.start + span.end) / len(span.doc)
                    if len(span) > 1500 or len(span) < 15:
                        continue
                    else:
                        costs[i, i + j, :] = - self.model.predict_raw(span.text + " [SEP] " + str(int(bins * loc)))  # TODO:
            self.cache_id = t
            np.save(self.cache_path + f"{str(self.cache_id)}_r{self.run_id}.npy", costs)
            return costs


    def _make_segments_dynamic_topics(self, doc, diffs, num_segments, window, alpha=0.016, beta=1e-3, t=None):
        """

        :param doc:
        :param diffs:
        :param num_segments:
        :param window:
        :param alpha:
        :param beta: parameter for the classification scores
        :return:
        """
        # classes = self.model.encoder.classes_
        classes = encoder.classes_
        c = len(classes)
        # sents = list(doc.spans["sents"])[:100]
        sents = list(doc.spans["sents"])
        # k = num_segments //10
        k = num_segments + 1
        # diffs = [0.] * window + [d[0] for d in diffs[:100-2*window]] + [0.] * window  # large difference means not to divide!
        # diffs = [0.] * window + [d[0] for d in diffs] + [0.] * window  # large difference means not to divide!
        diffs = [-np.inf] * window + [d[0] for d in diffs] + [-np.inf] * window  # large difference means not to divide!
        n = len(sents)
        # L = int(n / k)

        sent_lens = [0]+[len(s) for s in sents]
        _sent_lens = np.cumsum(sent_lens)
        # _Ls = len(doc) / self.avg_lens
        _Ls = self.avg_lens

        # TODO: do we actually make one less?
        prevs = np.zeros((n, c, k-1), dtype=int)
        prev_topic = np.zeros((n, c, k-1), dtype=int)
        # costs = np.zeros((n, c, k-1))  # the score for having the k-th break at the n-th sentence with topic c
        costs = np.full((n, c, k-1), np.inf)  # costs[n, c, k] the score for having the k+1-th (starting from 1)  break *before* the n-th sentence with topic c
        costs[0, :, 0] = 0.

        lens = _sent_lens[:n, None] - _sent_lens[:n]  # len_costs[r, c] contains the distance from *before* sent r to before sent c
        if not self.use_nb:
            print("using linear length penalties (by topic average length)")
            len_costs = np.broadcast_to(lens[..., None], lens.shape + (c,))
            len_costs = alpha * abs(len_costs - _Ls) / _Ls
            # len_costs = (1 - alpha) * abs(len_costs - _Ls) / _Ls
        else:
            len_costs = - alpha * \
                        np.stack([[self.len_estimators.predict(l, _c) for l in lens] for _c in range(c)], axis=-1)


        # classification_cost[s1, s2, c] represents the classification cost for a segment from s1 to s2 with topic c
        c_costs = self._get_classification_costs(sents, t=t)  # this *includes* the second sentence
        # _c_costs = np.zeros_like(c_costs)
        # idx = c_costs.argmin(axis=-1)
        # _c_costs[np.arange(c_costs.shape[0])[:, None], np.arange(c_costs.shape[1]), idx] = 1
        c_costs = beta * c_costs  # this *includes* the second sentence
        # c_costs = -beta * np.log(_c_costs)  # this *includes* the second sentence

        # _c_costs = (1-beta) * np.broadcast_to(classification_cost[..., None], classification_cost.shape + (c,))

        for _n in range(1, n):  # current considered breaking point. i.e. at _n=1 we consider a breakpoint *before* the second sentence
            _cost = len_costs[_n, :_n, :]
            for _k in range(1, min(k-1, _n+1)):  # number of breaking points until now (not included)
                for _c in range(c):  # current topic
                    # if _n==17:
                    #     print(_n)
                    # here we consider a topic _c and compare to all previous options
                    # arr = costs[:_n, :, _k-1] + _cost + _c_costs[:_n, _n, _c, :]  # prev costs + cost for new segments
                    arr = costs[:_n, :, _k-1] + _cost + c_costs[:_n, _n-1, _c, None]  # prev costs + cost for new segments
                    arr[:_n, _c] = np.inf  # can't have two consecutive of the same
                    m, t = np.unravel_index(np.argmin(arr), arr.shape)
                    costs[_n, _c, _k] = arr[m, t] - (1-alpha-beta) * diffs[_n]  # ??
                    prevs[_n, _c, _k] = int(m)
                    prev_topic[_n, _c, _k] = int(t)

        # _cost = (1 - alpha) * abs((n - np.arange(n)) - L) / L
        # _cost = (1 - alpha) * abs((_sent_lens[_n] - _sent_lens[:_n]) - _Ls) / _Ls
        # arr = costs[:n, :, k-2] + np.broadcast_to(_cost[..., None], _cost.shape+(c,))
        # _cost = _sent_lens[n] - _sent_lens[:n]
        # _cost = np.broadcast_to(_cost[..., None], _cost.shape+(c,))
        # _cost = len_costs
        # _cost = len_costs[n, :n, :]
        _cost = _sent_lens[n] - _sent_lens[:n]
        if not self.use_nb:
            _cost = np.broadcast_to(_cost[..., None], _cost.shape + (c,))
            _cost = alpha * abs(_cost - _Ls) / _Ls
        else:
            _cost = - alpha * \
                        np.stack([[self.len_estimators.predict(l, _c) for l in _cost] for _c in range(c)], axis=-1)

        last_cost = np.inf
        for _c in range(c):
            # _cost1 = np.copy(_cost)
            # _cost1[:, _c] = np.inf
            arr = costs[:n, :, k-2] + _cost + c_costs[:n, n-1, _c, None]  # why not -1??
            arr[:n, _c] = np.inf
            _m, _t = np.unravel_index(np.argmin(arr), arr.shape)
            if arr[_m, _t] <= last_cost:
                last_cost = arr[_m, _t]
                i, t, t2 = int(_m), int(_t), _c
        # i, t = int(m), int(t)  # best break for last. These are the beginnings
        j = k - 2
        topics = [t, t2]
        assignment = [i, n]
        while j > 0:
            i = prevs[i, t, j]
            t = prev_topic[i, t, j]
            j -= 1
            assignment.insert(0, i)
            topics.insert(0, t)
        assignment.insert(0, 0)
        # assignment.append(n)  # we should get k+1 in assignment
        # print("LEN ASSIGNMENT: ", len(assignment))

        doc.spans["segments"] = []
        doc._.topics = topics
        for i, j in enumerate(assignment[:-1]):
            doc.spans["segments"].append(doc[sents[j].start:sents[assignment[i+1]-1].end])
        # print("LEN ASSIGNMENT: ", len(doc.spans["segments"]))

    def segment_doc(self, doc, num_segments, window, dynamic=False, alpha=0.2, beta=1e-3, t=None, diff_scale=1.):
        """
        Segment a given spacy doc
        :param t: testimony number - for caching
        :param dynamic: wheter to use the dynamic spacing method
        :param alpha: weight for the dynamic method. If 0 then almost like uniform, and if 1 then like without the dynamic
        :param window: window for gpt2-scorer
        :param doc:
        :param num_segments:
        :return: self
        """
        if t is not None:
            self.scorer.load_cache(t)
            pass
        sents = list(doc.spans["sents"])

        diffs = []
        for j, s in enumerate(sents):
            if j < window or j + window > len(sents):
                continue
            # since the scores are the loss, so this is all minused. So a large diff means we want to put a boundary
            # this always represents the loss for putting a boundary *before*
            if self.scorer.gpt_j and t is None:
                time.sleep(0.5)
            gpt2_p1 = self.scorer.sentence_score(doc[sents[j-window].start:sents[j+window-1].end].text)
            if self.scorer.gpt_j and t is None:
                time.sleep(0.5)
            gpt2_p2 = self.scorer.sentence_score(doc[sents[j-window].start:sents[j].start].text) \
                      + self.scorer.sentence_score(doc[sents[j].start:sents[j+window-1].end].text)
            diffs.append(((gpt2_p1 - gpt2_p2)/diff_scale, j))  # j is the beginning?
        if t is not None:
            self.scorer.save_cache()
            pass

        # js = sorted([d[1] for d in diffs[:-num_segments+1]], reverse=True)  # this assumes num_segments>1

        if not dynamic:
            self._make_segments(doc, diffs, num_segments, window=window)
        else:
            if self.with_topics:
                self._make_segments_dynamic_topics(doc, diffs, num_segments, window, alpha=alpha, beta=beta, t=t)
            else:
                # self._make_segments_dynamic(doc, diffs, num_segments, window, alpha)
                if self.nb_topics:
                    dynamic_segmentation(doc, diffs, num_segments, window, alpha)
                else:
                    dynamic_segmentation(doc, diffs, num_segments, window, alpha, len_estimator=self.len_estimators)
        return doc

    def segment_sents(self, doc, sents, num_segments, window=3, dynamic=True, alpha=0.8):
        """
        Segments on a list of sentences for a document
        :param doc:
        :param sents:
        :param num_segments:
        :param window:
        :param dynamic:
        :param alpha:
        :return:
        """
        diffs = []
        for j, s in enumerate(sents):
            if j < window or j + window >= len(sents):
                continue
            gpt2_p1 = self.scorer.sentence_score(doc[sents[j-window].start:sents[j+window].start].text)
            gpt2_p2 = self.scorer.sentence_score(doc[sents[j-window].start:sents[j].start].text) \
                      + self.scorer.sentence_score(doc[sents[j].start:sents[j+window].start].text)
            diffs.append((gpt2_p1 - gpt2_p2, j))  # j is the beginning?
        if dynamic:
            k = num_segments
            diffs = [0.] * window + [d[0] for d in diffs] + [0.] * window  # large diff is good so used -inf - No! large difference means not to divide!

            n = len(sents)
            L = int(n / k)
            prevs = np.zeros((n, k-1), dtype=int)
            costs = np.zeros((n, k-1))
            costs[0, 1:] = np.inf
            for _n in range(1, n):
                for _k in range(1, k-1):
                    arr = costs[:_n, _k-1] + (1 - alpha) * abs((_n - np.arange(_n)) - L) / L  # not  like in the paper!!!
                    m = np.argmin(arr)
                    costs[_n, _k] = arr[m] - alpha * (diffs[_n]/50)
                    prevs[_n, _k] = int(m)

            arr = costs[:n, k-2] + (1 - alpha) * abs(n - np.arange(n) - L) / L
            m = np.argmin(arr)

            i = int(m)  # best break for last. These are the beginnings
            j = k - 2
            assignment = [i]
            while j > 0:
                i = prevs[i, j]
                j -= 1
                assignment.insert(0, i)
            assignment.insert(0, 0)
            assignment.append(n)  # we should get k+1 in assignment

            segments = []
            for i, j in enumerate(assignment[:-1]):
                segments.append(doc[sents[j].start:sents[assignment[i+1]-1].end])
            # print("LEN ASSIGNMENT: ", len(doc.spans["segments"]))
        return segments


class NSPSegmentor:
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/'):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
        # self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir='/cs/snapless/oabend/eitan.wagner/cache/')
        # logging.info("Using Bert")
        self.tokenizer = FNetTokenizer.from_pretrained("google/fnet-base")
        self.model = FNetForNextSentencePrediction.from_pretrained("google/fnet-base")
        self.model.eval()
        logging.info("Using fnet")
        # self.model = None
        # self.tokenizer = None

        self.base_path = base_path
        self.cache = {}
        self.cache_id = None

    def save_cache(self):
        np.save(self.base_path + f"/nsp_cache{str(self.cache_id)}.npy", self.cache)

    def load_cache(self, i):
        self.cache_id = i
        try:
            self.cache = np.load(self.base_path + f"/nsp_cache{str(i)}.npy",
                                 allow_pickle='TRUE').item()
        except IOError as err:
            # except:
            pass

    def fit(self):
        return self

    def _NSP_score(self, sent1, sent2):
        """
        Calculates the probability that sent2 is the continuation of sent1
        :param sent1:
        :param sent2:
        :return:
        """
        encoding = self.tokenizer(sent1, sent2, return_tensors='pt').to(dev)
        outputs = self.model(**encoding, labels=torch.LongTensor([1]).to(dev))
        logits = outputs.logits.detach().cpu()
        return logits[0, 0]

    def segment_doc(self, doc, num_segments, window, t=None, dynamic=False, alpha=0.2):
        """
        Segment a given spacy doc
        :param dynamic: whether to use the dynamic spacing method
        :param window: window for NSP-scoring
        :param doc:
        :param num_segments:
        :param t:
        :return: self
        """
        if t is not None:
            self.load_cache(t)
            pass
        # print("1 it seems the problem is here!!")
        sents = doc.spans["sents"]
        self.model.to(dev)

        diffs = []
        for j, s in enumerate(sents):
            # print(j)
            if j < window or j + window > len(sents):
                continue

            # print(doc[sents[j-window].start:sents[j].start].text, doc[sents[j].start:sents[j+window].start].text)
            score = self._NSP_score(doc[sents[j-window].start:sents[j].start].text,
                                    doc[sents[j].start:sents[j+window-1].end].text)
            diffs.append((-score, j))
            # the score is high probability, so the cost of putting a boundary here is high. But in the algorithm we take the minus


        self.model.cpu()
        if t is not None:
            # self.save_cache()
            pass
        if not dynamic:
            diffs.sort(reverse=True)  # This is only correct with the minus
            last_js = sorted([d[1] for d in diffs[:num_segments-1]])

            print(3)
            doc.spans["segments"] = [doc[:sents[last_js[0]-1].end]]
            for i, j in enumerate(last_js[:-1]):
                doc.spans["segments"].append(doc[sents[j].start:sents[last_js[i+1]-1].end:])
            doc.spans["segments"].append(doc[sents[last_js[-1]].start:])

        else:
            dynamic_segmentation(doc, diffs, num_segments, window, alpha)
        return doc


def create_dynamic(doc, c_probs):
    # assumes the frequencies are a two dimensional and in log
    k = len(doc.spans["segments"])
    c = len(encoder.classes_)
    prevs = np.zeros((k, c))
    scores = np.zeros((k, c))  # score table. scores[k, c] is the score for the k-th assignment being c
    scores[0, :] = c_probs[0, :]  # frequencies for first segments
    probs = np.ones((c, c))  # uniform transitions (up to staying)
    probs[np.arange(k), np.arange(k)] = 0
    probs = np.log(probs / np.sum(probs, axis=0, keepdims=True))
    for _k in range(1, k):
        for _c in range(c):
            arr = scores[_k-1, :] + c_probs[_k, :] + probs[:, _c]
            m = np.argmax(arr)
            scores[_k, _c] = arr[m]
            prevs[_k, _c] = m

    prev_c = int(np.argmax(scores[k-1, :]))
    assigmnent = [prev_c]
    for _k in range(k-1, 1, -1):
        prev_c = prevs[_k, prev_c]
        assigmnent.insert(0, prev_c)
    return assigmnent

def make_topics(doc, allow_doubles=True):
    """
    Makes a topic assignment for the given doc
    :param self:
    :param doc:
    :return:
    """
    from transformer_classification import TransformerClassifier
    # model = TransformerClassifier(base_path='/cs/snapless/oabend/eitan.wagner/segmentation/', model_name='xlnet-large-cased', mc=None, full_lm_scores=False)
    model = TransformerClassifier(base_path='/cs/snapless/oabend/eitan.wagner/segmentation/', model_name='distilroberta', mc=None, full_lm_scores=False)

    if allow_doubles:
        topics = []
        for span in doc.spans["segments"]:
            m = np.argmax(model.predict_raw(span.text))
            topics.append(int(m))
        doc._.topics = topics
        return
    else:
        bins = 10
        k = len(doc.spans["segments"])
        c = len(encoder.classes_)
        c_probs = np.zeros((k, c))
        for i, span in enumerate(doc.spans["segments"]):
            loc = 0.5 * (span.start + span.end) / len(span.doc)
            c_probs[i, :] = model.predict_raw(span.text + " [SEP] " + str(int(bins * loc)))
        doc._.topics = create_dynamic(doc, c_probs)

# ************ bert IOB method  - not implemented yet ******************

def make_data(docs, context=1):
    for doc in docs:
        sents = list(doc.sents)
        iob = get_per_sent(doc)
        data = []
        for i, b in enumerate(iob):
            if i < context or i >= len(iob) - context:
                continue
            data.append((sents[i], " ".join(sents[i-context:i+context]), b))  # tuple of sent, context, label
    return data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class BertIOBSegmentor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')
        self.model = None
        self.tokenizer = None

    def load(self):
        pass

    def fit(self, docs, context=1):
        """

        :param docs: list of reference pre-segmented spacy docs
        :return:
        """
        data = make_data(docs, context=context)

        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large', cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")

        encodings = np.array([self.tokenizer(text[0], text[1], truncation=True, padding=True) for text in data])
        labels = np.array(list(zip(*data))[2])
        train_encodings,val_encodings, train_labels, val_labels = train_test_split(encodings, labels, test_size=.2)
        logging.info("made encodings")

        train_dataset = Dataset(train_encodings, train_labels)
        val_dataset = Dataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=3,              # total number of training epochs
            learning_rate=5e-5 * 1/16,
            per_device_train_batch_size=1,  # batch size per device during training
            per_device_eval_batch_size=1,   # batch size for evaluation
            # learning_rate=5e-5,
            # per_device_train_batch_size=16,  # batch size per device during training
            # per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            label_smoothing_factor=0.,
            # report_to=None,
        )

        model = RobertaForSequenceClassification.from_pretrained('roberta-large',
                                                                 cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                                 num_labels=2)
        model.to(dev)

        logging.info("Training")
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,             # evaluation dataset
            compute_metrics=compute_metrics,
        )
        trainer.train()

        out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/iob-roberta'
        model.save_pretrained(out_path)
        self.model = model
        return self

    def segment(self, text, num_segments, window=1):
        doc = self.nlp(text)
        sents = list(doc.sents)
        threshold = num_segments / len(sents)

        doc_data = make_data([self.nlp(doc)], window=window)
        # preds = np.zeros(len(sents))
        boundaries = []

        for i, d in enumerate(doc_data):
            inputs = self.tokenizer(d[0], d[1], return_tensors="pt")
            labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
            outputs = self.model(**inputs, labels=labels)
            logits = outputs.logits
            if logits[0][1] >= threshold:
                # preds[i + window] = 1
                boundaries.append(i+window)

        doc.spans["segments"] = [doc[:boundaries[0]]]
        for i, j in enumerate(boundaries[:-1]):
            doc.spans["segments"].append(doc[j:boundaries[i+1]:])
        doc.spans["segments"].append(doc[boundaries[-1]:])

        return doc


# ************************ baselines for topic assignment **************************

class TopicAssigner:
    """
    Assigns a random topic assignment (by markov chain or frequency)
    """
    def __init__(self, markov_chain=False, frequencies=None, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/',
                 name="models/transitions/mc.json", from_classifier=False):
        """
        :param markov_chain: whether the assigner is based on MC probabilities. Otherwise by given probabilities
        :param frequencies: probilities for the topic. if None then uses uniform
        """
        if frequencies is None:
            frequencies = np.ones(len(encoder.classes_)) / len(encoder.classes_)
        self.frequencies = frequencies
        self.markov_chain = markov_chain
        if markov_chain:
            if name[-4:] == "json":
                self.mc = MC(base_path=base_path, name=name)
            elif name[-3:] == "pkl":
                with open(base_path + 'models/transitions/mcc5_iner5_iter15.pkl', 'rb') as infile:
                    self.mc = joblib.load(infile)
        if from_classifier:
            from transformer_classification import TransformerClassifier
            # model = TransformerClassifier(base_path='/cs/snapless/oabend/eitan.wagner/segmentation/', model_name='xlnet-large-cased', mc=None, full_lm_scores=False)
            self.model = TransformerClassifier(base_path='/cs/snapless/oabend/eitan.wagner/segmentation/',
                                          model_name='distilroberta', mc=None, full_lm_scores=False)

    def create(self, doc, avoid_doubles=False):
        """
        Creates an assignment for a segmented doc
        :param doc:
        :return: list of topics
        """
        k = len(doc.spans["segments"])
        if self.markov_chain:
            return list(self.mc.sample(k))
        else:
            if avoid_doubles:
                prev = -1
                topics = []
                for _ in range(k):
                    probs = np.array(self.frequencies)
                    if prev != -1:
                        probs[prev] = 0
                        probs = probs / probs.sum()
                    prev = int(np.random.choice(len(encoder.classes_), size=1, p=probs))
                    topics.append(prev)
                return topics
            return list(np.random.choice(len(encoder.classes_), size=k, p=self.frequencies))

    def create_dynamic(self, doc):
        k = len(doc.spans["segments"])
        c = len(encoder.classes_)
        c_probs = np.zeros((k, c))
        for i, span in enumerate(doc.spans["segments"]):
            c_probs[i, :] = self.model.predict_raw(span.text)
        # assumes the frequencies are two dimensional and in log
        prevs = np.zeros((k, c), dtype=int)
        scores = np.zeros((k, c))  # score table. scores[k, c] is the score for the k-th assignment being c
        scores[0, :] = c_probs[0, :]  # frequencies for first segments
        probs = np.ones((c, c))  # uniform transitions (up to staying)
        probs[np.arange(c), np.arange(c)] = 0
        probs = np.log(probs / np.sum(probs, axis=0, keepdims=True))
        for _k in range(1, k):
            for _c in range(c):
                arr = scores[_k-1, :] + c_probs[_k, _c] + probs[:, _c]
                m = np.argmax(arr)
                scores[_k, _c] = arr[m]
                prevs[_k, _c] = int(m)

        prev_c = int(np.argmax(scores[k-1, :]))
        assigmnent = [prev_c]
        for _k in range(k-1, 0, -1):
            # print(prev_c)
            # sys.stdout.flush()
            prev_c = prevs[_k, prev_c]
            assigmnent.insert(0, prev_c)
        return assigmnent


# ******************** reference data ******************

def get_topic_list(data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/", use_raw=False):
    """
    Obtains list of topics (after conversion) for the reference data
    :param data_path:
    :return: list of all topics in the reference data
    """
    with open(data_path + 'title_w_segments.json', 'r') as infile:
        all_topics = [tw[0] for tw in json.load(infile)]
    # logging.info([e for e in enumerate(all_topics)])
    # logging.info(f"Count: {len(all_topics)}")

    topic2newtopic = pd.read_csv(data_path + 'noam_old2noam_newtopic.csv', header=None, index_col=0, squeeze=True).to_dict()
    newtopic2num = pd.read_csv(data_path + 'noam_newtopic2num.csv', header=None, index_col=0, squeeze=True).to_dict()
    num2newtopic = pd.read_csv(data_path + 'num2newtopic.csv', header=None, index_col=0, squeeze=True).to_dict()
    if use_raw:
        new_topics = [topic2newtopic.get(t, None) for t in all_topics]
    else:
        new_topics = [num2newtopic.get(newtopic2num.get(topic2newtopic.get(t, None), None), None) for t in all_topics]
    # logging.info(f"None count: {sum([1 for t in new_topics if t is None])}")

    return new_topics
    # new_words2topics = {w: num2newtopic.get(newtopic2num.get(t, None), None) for w, t in words2topics.items()}


def save_doc(doc, doc_num, path="/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/"):
    """
    Save the given doc
    :param doc:
    :param doc_num:
    :param path:
    :return:
    """
    doc.to_disk(path + "doc_" + str(doc_num))


def make_gold_xlsx(data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_xlsx/", save=True, name="", r=None):
    """
    Make spacy docs from annotated documents in xlsx format (with multiple sheets)
    :param data_path:
    :param save:
    :return:
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.disable_pipes('ner')

    sheets = pd.read_excel(data_path + f"sf_annotation - {name}.xlsx", sheet_name=None)
    for t, df in sheets.items():
        # if t == "sf_25639":
        #     pass
        if r is not None and t not in r:
            continue
        sents = df['Text'].astype("string").fillna('').to_list()
        overlap = df['Overlap?'].to_list()
        for i, s in enumerate(sents):
            if s == "":
                overlap.pop(i)
        sent_words = [s.split() for s in sents]
        sent_starts = [[True if j == 0 else False for j, w in enumerate(s_w)] for s_w in sent_words]
        # lens = np.array([len(s) + 1 for s in sents])  # +1 for the following whitespace
        # char_ends = np.cumsum([len(_s) + 1 for _s in sents])
        # char_starts = [0] + char_ends.tolist()[:-1]
        topics = df['Topic'].to_numpy()

        starts = np.insert(np.nonzero(topics[1:] != topics[:-1])[0] + 1, 0, 0)
        ends = np.append(starts[1:], len(topics))
        # seg_starts = [[True if i in starts and  else False for _ in s_w] for i, s_w in enumerate(sent_words)]

        segs_w_topics = [[" ".join(sents[s:e]), topics[s] if topics[s] != "None" else "NO_TOPIC"] for s, e in zip(starts, ends)]
        topics = list(list(zip(*segs_w_topics))[1])

        _docs = []
        # sents = []
        num_tokens = 0

        for s, e in zip(starts, ends):
            _docs.append(Doc(nlp.vocab, words=[w for s_w in sent_words[s:e] for w in s_w],
                             sent_starts=[_s for s_s in sent_starts[s:e] for _s in s_s]))

        # for j, s in enumerate(segs_w_topics):
        #     # logging.info(j)
        #     _docs.append(nlp(s[0]))
        #     sents = sents + [(s.start + num_tokens, s.end+num_tokens) for s in _docs[-1].sents]
        #     num_tokens += len(_docs[-1])

        doc = Doc.from_docs(list(_docs), ensure_whitespace=True)
        doc._.overlap = overlap
        ends = np.cumsum([len(_d) for _d in _docs])  # does this count the whitespaces?? they shouldn't count
        starts = [0] + ends.tolist()[:-1]
        doc.spans["segments"] = [doc[s:e] for s, e in zip(starts, ends)]
        doc.spans["sents"] = list(doc.sents)
        doc._.topics = topics

        logging.info(f"#segments ({t}): " + str(len(doc.spans["segments"])))
        logging.info("#sents: " + str(len(doc.spans["sents"])))
        if save:
            save_doc(doc, doc_num=t + "_" + name[:1])


def make_gold_csv(data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/", save=True):
    """
    Make spacy docs from annotated documents in csv format
    :param data_path:
    :return:
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.disable_pipes('ner')

    if not save:
        all_topics = encoder.classes_
        topic_w_segments = {t: [] for t in set(all_topics)}
    for i in range(111, 121):
        # logging.info(i)
        testimony = pd.read_csv(data_path + f'gold csv/testimony_{i}.csv', header=None)
        segments = list(testimony[:][0])
        topics = list(testimony[:][1])

        _docs = []
        sents = []
        num_tokens = 0
        for j, s in enumerate(segments):
            # logging.info(j)
            _docs.append(nlp(s))
            sents = sents + [(s.start + num_tokens, s.end+num_tokens) for s in _docs[-1].sents]
            num_tokens += len(_docs[-1])

        doc = Doc.from_docs(list(_docs), ensure_whitespace=True)
        ends = np.cumsum([len(_d) for _d in _docs])  # does this count the whitespaces?? they shouldn't count
        starts = [0] + ends.tolist()[:-1]
        doc.spans["segments"] = [doc[s:e] for s, e in zip(starts, ends)]
        doc.spans["sents"] = [doc[s[0]:s[1]] for s in sents]
        doc._.topics = topics

        logging.info(f"#segments ({i}): " + str(len(doc.spans["segments"])))
        logging.info("#sents: " + str(len(doc.spans["sents"])))
        if save:
            save_doc(doc, doc_num=i)
        else:
            for t, s in zip(topics, segments):
                if t == 'ghetto':
                    t = "Ghetto"
                if t == 'Discussions/desicions':
                    t = 'Discussions/decisions'
                topic_w_segments[t].append(s)

    if not save:
        with open(data_path + 'topic_w_segments2.json', 'w+') as outfile:
            json.dump(topic_w_segments, outfile)
    return


def make_gold_docs(data_path="/cs/snapless/oabend/eitan.wagner/segmentation/data/", save=True):
    """
    Makes gold doc as spacy docs with segments, and saves in new folder
    :param data_path:
    :return:
    """
    segment = ""
    _docs = []
    docs = {}
    t_num = -1
    sents = []
    num_tokens = 0

    # nlp = spacy.load("en_core_web_trf")
    nlp = spacy.load("en_core_web_sm")
    # nlp.disable_pipes('transformer', 'ner')
    nlp.disable_pipes('ner')

    with open(data_path + 'segmented.txt', encoding="utf-8") as infile:
        for l in infile:
            l = l.replace("\u2019", "'").replace("\u201c","\"").replace("\u201d","\"")
            if l[:3] == "***":  # segment boundary
                if segment != "":
                    _docs.append(nlp(segment.rstrip()))
                    sents = sents + [(s.start + num_tokens, s.end+num_tokens) for s in _docs[-1].sents]
                    num_tokens += len(_docs[-1])
                    # char_lens.append(len(segment.rstrip()))
                    segment = ""
                i = l.find("Testimony")
                if i != -1:
                    if t_num == -1:
                        t_num = int(l[i + len("Testimony "):i + len("Testimony ")+3])
                    if len(_docs) > 0:
                        docs[t_num] = Doc.from_docs(list(_docs), ensure_whitespace=True)
                        ends = np.cumsum([len(_d) for _d in _docs])  # does this count the whitespaces?? they shouldn't count
                        starts = [0] + ends.tolist()[:-1]
                        docs[t_num].spans["segments"] = [docs[t_num][s:e] for s, e in zip(starts, ends)]
                        docs[t_num].spans["sents"] = [docs[t_num][s[0]:s[1]] for s in sents]
                    _docs = []
                    sents = []
                    num_tokens = 0
                    t_num = int(l[i + len("Testimony "):i + len("Testimony ")+3])

            elif l.strip() != "":
                segment = segment + l.rstrip('\n') + " "

    # docs = []
    # for t in range(101, 110):
    #     doc = nlp(texts[str(t)])
    #     doc.spans["segments"] = []
    #     logging.info(len(doc.text))
    #     while start_char < len(doc.text):
    #         # for l in char_lens:
    #         l = char_lens.pop(0)
    #         logging.info(t, start_char, l)
    #         logging.info(doc.text[start_char:start_char+l])
    #         if start_char + l > len(doc.text):
    #             l = len(doc.text) - start_char
    #         doc.spans["segments"].append(doc.char_span(start_char, start_char+l, alignment_mode="contract"))
    #         # start_char = start_char + len(doc.spans["segments"][-1])
    #         start_char = start_char + l
    #     save_doc(doc, doc_num=t)
    #     start_char = 0

    # logging.info("Segment list:")
    # for segment in doc.spans["segments"]:
    #     logging.info(segment)
    # docs.append(doc)

    # all_topics = get_topic_list(data_path)
    all_topics = get_topic_list(data_path, use_raw=True)

    logging.info(["None!!!"+str(e[0]) for e in enumerate(all_topics) if e[1]==None])
    logging.info(len(all_topics))

    lens_notopic = []
    lens = []
    t_count = 0

    topic_w_segments = {t: [] for t in all_topics}
    for i, doc in docs.items():
        # for s in doc.spans["segments"]:
        #     logging.info(s)
        logging.info(f"Old doc lengths (in tokens), {i}: ")
        logging.info(len(doc))
        logging.info(sum([len(s) for s in doc.spans["segments"]]))
        logging.info(sum([len(s) for s in doc.spans["sents"]]))
        doc._.topics = all_topics[t_count:t_count+len(doc.spans["segments"])]

        # doc2 = nlp(doc.text)
        # doc2.spans["segments"] = [doc2[s.start:s.end] for s in doc.spans["segments"]]
        # logging.info(f"segments:" + str(len(list(doc2.spans["segments"]))))
        # logging.info(f"sents:" + str(len(list(doc2.sents))))
        logging.info(f"sents_old generator:" + str(len(list(doc.sents))))
        logging.info(f"sents_old:" + str(len(doc.spans["sents"])))
        # logging.info(sum(get_per_sent(doc2)))
        logging.info(sum(get_per_sent(doc)))
        # save_doc(doc2, doc_num=i)

        # logging.info(all_topics[t_count:t_count+len(doc.spans["segments"])])
        logging.info(["Empty segment!!!!!" for s in doc.spans["segments"] if len(s) == 0])
        logging.info(len(all_topics[t_count:t_count+len(doc.spans["segments"])]))
        t_count += len(doc.spans["segments"])
        logging.info(f"Count: {t_count}")

        # merge same topics
        segments = [s for s in doc.spans["segments"]]
        topics = []
        for j, t in enumerate(doc._.topics):
            if j+1 < len(doc._.topics) and t == doc._.topics[j+1]:
                # segments[j:j+2] = [doc[doc.spans["segments"][j].start:doc.spans["segments"][j+1].end]]
                segments[j+1] = doc[segments[j].start:segments[j+1].end]
                segments[j] = None
            else:
                topics.append(t)
                # topics.pop(j)
        segments = [s for s in segments if s is not None]
        doc.spans["segments"] = segments
        doc._.topics = topics

        if save:
            save_doc(doc, doc_num=i)
        else:
            for t, s in zip(topics, segments):
                topic_w_segments[t].append(s.text)


        # lens_notopic.append([len(s) for j, s in enumerate(doc.spans["segments"]) if doc._.topics[j] == "NO_TOPIC"])
        lens_notopic = lens_notopic + [len(s) for j, s in enumerate(doc.spans["segments"]) if doc._.topics[j] == "NO_TOPIC"]
        # lens.append([len(s) for j, s in enumerate(doc.spans["segments"]) if doc._.topics[j] != "NO_TOPIC"])
        # lens = lens + [len(s) for j, s in enumerate(doc.spans["segments"]) if doc._.topics[j] != "NO_TOPIC"]
        lens = lens + [len(s) for j, s in enumerate(doc.spans["segments"])]
    logging.info(f"lens NO_TOPIC: {lens_notopic}")
    logging.info(f"lens other: {lens}")
    logging.info(f"avg len NO_TOPIC: {np.mean(lens_notopic)}")
    logging.info(f"std len NO_TOPIC: {np.std(lens_notopic)}")
    logging.info(f"avg len other: {np.mean(lens)}")
    logging.info(f"std len other: {np.std(lens)}")
    logging.info(f"Count: {t_count}")

    if not save:
        with open(data_path + 'topic_w_segments.json', 'w+') as outfile:
            json.dump(topic_w_segments, outfile)


def evaluate(doc, gold_doc, method="dynamic", t=None):
    """
    Evaluate doc segmentation against the gold_doc
    :param doc:
    :param gold_doc:
    :param return_all:
    :param only_dynamic: whether to evaluate the given (dynamic) segmentation
    :return: accuracy and windowdiff scores, if return_all then also for uniform and gpt2 baselines
    """
    estimated_segments = int(len(gold_doc) / 256.29)

    if method == "uniform":
        us = UniformSegmentor()
        doc2 = us.segment_doc(doc, estimated_segments)
        logging.info(f"Uniform segmentation scores:")

    if method == "dynamic":
        logging.info(f"Dynamic segmentation scores:")
        doc2 = doc

    if method == "gpt2_dynamic":
        alpha = 0.8
        gs = Gpt2Segmentor()
        # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
        # doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=True, alpha=alpha, t=t)
        doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=True, alpha=alpha)
        logging.info(f"GPT2 dynamic segmentation scores, alpha {alpha}: ")

    if method == "gpt2":
        gs = Gpt2Segmentor()
        # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
        # doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=False, t=t)
        doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=False)
        logging.info(f"GPT2 (only) segmentation scores: ")

    if method == "nsp":
        nsps = NSPSegmentor()
        doc2 = nsps.segment_doc(doc, estimated_segments, window=3)
        # doc2 = nsps.segment_doc(doc, estimated_segments, window=3, t=t)
        logging.info(f"NSP segmentation scores t_{t}: ")

    a_s = accu_scores(doc2, gold_doc)
    w_d = windowdiff(doc2, gold_doc, k=int(0.5 * len(list(gold_doc.sents)) / len(gold_doc.spans["segments"])))
    logging.info(a_s)
    logging.info(w_d)
    return a_s, w_d

    #
    # if not only_dynamic:
    #     us = UniformSegmentor()
    #     estimated_segments = int(len(gold_doc) / 256.29)
    #     doc2 = us.segment(gold_doc.text, estimated_segments)
    #     logging.info(f"Uniform segmentation scores:")
    #     uni_accu = accu_scores(doc2, gold_doc)
    #     uni_wd = windowdiff(doc2, gold_doc, k=int(0.5 * len(list(gold_doc.sents)) / len(gold_doc.spans["segments"])))
    #     logging.info(uni_accu)
    #     logging.info(uni_wd)
    #
    #     gs = Gpt2Segmentor()
    #     # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
    #     doc3 = gs.segment_doc(gold_doc, estimated_segments, window=3, dynamic=True)
    #     logging.info(f"GPT2 segmentation scores: ")
    #     gpt2_accu = accu_scores(doc3, gold_doc)
    #     gpt2_wd = windowdiff(doc3, gold_doc, k=int(0.5 * len(list(gold_doc.sents)) / len(gold_doc.spans["segments"])))
    #     logging.info(gpt2_accu)
    #     logging.info(gpt2_wd)
    #
    # logging.info(f"Dynamic segmentation scores:")
    # a_s, w_d = accu_scores(doc, gold_doc), windowdiff(doc, gold_doc, k=int(0.5 * len(list(gold_doc.sents)) / len(gold_doc.spans["segments"])))
    # logging.info(a_s)
    # logging.info(w_d)
    # if return_all:
    #     return a_s, w_d, uni_accu, uni_wd, gpt2_accu, gpt2_wd
    # return a_s, w_d


def combine_close(diffs, topics, threshold=0.4, path='/cs/snapless/oabend/eitan.wagner/segmentation/models/xlnet-large-cased'):
    """
    Combine the diffs if the correlation between the topics is high
    :param diffs:
    :param topics:
    :return:
    """
    cor_matrix = np.load(path + "/correlation_matrix.npy")
    is_close = [True if cor_matrix[topics[i], topics[i+1]] > threshold else False for i in range(len(topics) - 1)]
    for i in range(len(diffs) - 1, 0, -1):
        if is_close[i-1]:
            diffs[i-1] += diffs[i]
            diffs[i] = 0
    return [d for d in diffs if d > 0]


def _make_length_dict(doc, doc_dynamic=None, gold_doc=None, gold_doc2=None, t=None, fixed_segments=False, segmentors=None,
                      out_path="/cs/snapless/oabend/eitan.wagner/segmentation/out_docs_max", avg=None, use_close=False,
                      threshold=0.4, eval_gpt2_topics=True, from_dict=False):
    """
    Makes lengths for one testimony.
    :param doc_dynamic:
    :param fixed_segments:
    :param segmentors:
    :param avg:
    :param use_close:
    :param threshold:
    :param eval_gpt2_topics:
    :param doc:
    :param gold_doc:
    :param gold_doc2:
    :param t:
    :param out_path:
    :return:
    """
    def add_to_dict(dict, doc, method, use_close=False, use_encoder=False):
        ends = np.array(get_per_sent(doc))
        # idxs = ends != 0
        idxs = [-1] + np.nonzero(ends)[0].tolist()  # the ends are included so the "previous end" is -1
        diffs = []
        for i in range(1, len(idxs)):
            diffs.append(idxs[i] - idxs[i-1])
        if use_close:
            if use_encoder:  # this is for the gold doc
                diffs = combine_close(diffs, encoder.transform(doc._.topics), threshold=threshold)
            else:
                diffs = combine_close(diffs, doc._.topics, threshold=threshold)
        dict[method] = diffs

    dict = {}
    if from_dict:
        for m, d in doc.items():
            add_to_dict(dict, d, method=m, use_encoder=(True if m[:4] == "gold" else False))
        return dict

    logging.info("Adding gold and dynamic")
    logging.info(f"Using closeness: {use_close}, thereshold: {threshold}")
    # gold_doc._.topics = encoder.transform(gold_doc._.topics)

    add_to_dict(dict, gold_doc, method="gold", use_close=use_close, use_encoder=True)
    if gold_doc2 is not None:
        add_to_dict(dict, gold_doc2, method="gold2", use_close=use_close, use_encoder=True)
    if doc_dynamic is not None:
        add_to_dict(dict, doc_dynamic, method="dynamic", use_close=use_close)

    if not fixed_segments:
        if avg is not None:
            estimated_segments = int(len(gold_doc) / avg)
        else:
            estimated_segments = int(len(gold_doc) / 256.29)
    else:
        # estimated_segments = len(doc_dynamic.spans["segments"])
        # estimated_segments = len(dict["gold"])
        estimated_segments = len(dict["dynamic"])

    if segmentors is None:
        segmentor = UniformSegmentor()
    else:
        segmentor = segmentors["uniform"]
    us = segmentor
    doc2 = us.segment_doc(doc, estimated_segments)
    logging.info("Adding uniform")
    add_to_dict(dict, doc2, method="uniform")

    if segmentors is None:
        segmentor = Gpt2Segmentor()
    else:
        segmentor = segmentors["gpt2"]
    gs = segmentor
    # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
    doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=False)
    # doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=False, t=t)
    logging.info("Adding gpt2")
    add_to_dict(dict, doc2, method="gpt2")

    alpha = 0.8
    # gs = Gpt2Segmentor()
    # doc3 = gs.segment(gold_doc.text, estimated_segments, window=3)
    # doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=True, alpha=alpha, t=t)
    doc2 = gs.segment_doc(doc, estimated_segments, window=3, dynamic=True, alpha=alpha)
    logging.info("Adding gpt2_dynamic")
    add_to_dict(dict, doc2, method="gpt2_dynamic")

    if eval_gpt2_topics:
        make_topics(doc2)
        logging.info("Gestalt scores (gpt2): ")
        evaluate_topics(doc2, gold_doc, evaluate_dynamic=True, method="gestalt")
        logging.info("Edit scores (gpt2): ")
        evaluate_topics(doc2, gold_doc, evaluate_dynamic=True, method="edit")

    if segmentors is None:
        segmentor = NSPSegmentor()
    else:
        segmentor = segmentors["nsp"]
    nsps = segmentor
    # doc2 = nsps.segment_doc(doc, estimated_segments, window=3, t=t)
    doc2 = nsps.segment_doc(doc, estimated_segments, window=3)
    logging.info("Adding nsp")
    add_to_dict(dict, doc2, method="nsp")

    return dict

def make_len_dict(path="/cs/snapless/oabend/eitan.wagner/segmentation/", ratio=None, fixed_segments=False,
                  segmentors=None, r=None, method="max", avg=None, use_close=False, suffix="", annotators=None,
                  with_data=True, bin=False, from_dict=None, run_id=None):
    """
    Makes dictionary of segment lengths for each method, for using segeval.
    Saves the dict in path.
    :param path:
    :return:
    """
    if r is None:
        r = range(101, 111)
    if annotators is None:
        annotators = [""]
    if bin:
        bin = "_b"
    else:
        bin = ""
    dict = {}
    if from_dict is None:
        for i in r:
            logging.info(f"i: {i}")
            # if i == 108:
            #     continue

            gold_doc = Doc(Vocab()).from_disk(path + 'data/gold_docs/doc_' + str(i) + annotators[0])
            doc = Doc(Vocab()).from_disk(path + 'data/gold_docs/doc_' + str(i) + annotators[0])
            if len(annotators) > 1:
                gold_doc2 = Doc(Vocab()).from_disk(path + 'data/gold_docs/doc_' + str(i) + annotators[1])
            else:
                gold_doc2 = None

            if not with_data:
                doc_dynamic = Doc(Vocab()).from_disk(path + 'data/gold_docs/doc_' + str(i) + annotators[0]+bin)
            elif ratio is None:
                doc_dynamic = Doc(Vocab()).from_disk(path + f'out_docs_{method}/doc_' + str(i) + annotators[0]+bin)
            else:
                doc_dynamic = Doc(Vocab()).from_disk(path + f'out_docs_{method}/doc_' + str(i) + annotators[0] + "_" + str(ratio)+bin)
            dict[str(i)] = _make_length_dict(doc, doc_dynamic, gold_doc, gold_doc2=gold_doc2, t=i, fixed_segments=fixed_segments, segmentors=segmentors, avg=avg, use_close=use_close)
    else:
        dict[r] = _make_length_dict(doc=from_dict, from_dict=True)

    # reverse the order of testimonies and methods
    # _dict = {k: {} for k in dict["101"].keys()}  # for each method
    _dict = {k: {} for k in list(dict.values())[0].keys()}  # for each method
    for i, d in dict.items():
        for method, l in d.items():
            _dict[method][str(i)] = l

    full_dict = {"items": _dict, "segmentation_type": "linear"}
    if run_id is None:
        run_id = ""
    else:
        run_id = "_" + run_id
    with open(path + f'segmentation_lens{suffix}{bin}{run_id}.json', 'w+') as outfile:
        json.dump(full_dict, outfile)


def seg_eval(path="/cs/snapless/oabend/eitan.wagner/segmentation/", gold2=False, r=None, suffix="", bin=False, run_id=None):
    """
    Evaluate with segeval package.
    :param path:
    :return:
    """
    if r is None:
        r = range(101, 111)

    if run_id is None:
        run_id = ""
    else:
        run_id = "_" + run_id

    if gold2:
        gold = "_g2"
        gold = ""
    else:
        gold = ""

    def f_measure(seg1, gold):
        conf_matrix = segeval.boundary_confusion_matrix(seg1, gold)
        return segeval.fmeasure(conf_matrix)
    def precision(seg1, gold):
        conf_matrix = segeval.boundary_confusion_matrix(seg1, gold)
        return segeval.precision(conf_matrix)
    if bin:
        dataset = segeval.input_linear_mass_json(path + f'segmentation_lens{suffix}_b{run_id}{gold}.json')
    else:
        dataset = segeval.input_linear_mass_json(path + f'segmentation_lens{suffix}{run_id}{gold}.json')
        print(dataset)

    sims = {}
    golds = ["gold"]
    if gold2:
        sims2 = {}
        golds.append("gold2")
        methods = []
    # methods = ["gpt2", "gpt2_dynamic", "nsp", "uniform", "dynamic"]
    else:
        methods = ["gpt2", "gpt2-topics", "uniform", "nsp"]
    # methods = ["gpt2", "gpt2_dynamic", "uniform", "dynamic", "gold2"]

    for g in golds:
        g_sims = sims2 if gold2 else sims
        for func in [segeval.boundary_similarity, segeval.segmentation_similarity, segeval.window_diff,
                     segeval.pk, f_measure, precision]:
            avg = {}
            for m in methods:
                # print(m)
                _sims = []
                # for i in range(101, 111):
                for i in r:
                    # if i == 108:
                    #     continue
                    seg1 = dataset[m][str(i)]
                    gold = dataset["gold"][str(i)]
                    # sims.append(segeval.boundary_similarity(seg1, gold))
                    _sims.append(func(seg1, gold))
                avg[m] = float(np.mean(_sims))
            # g_sims[func.__name__] = avg
            # logging.info(avg)
            if gold2:
                _sims = []
                # for i in range(101, 111):
                for i in r:
                    # if i == 108:
                    #     continue
                    seg1 = dataset["gold2"][str(i)]
                    gold = dataset["gold"][str(i)]
                    # sims.append(segeval.boundary_similarity(seg1, gold))
                    _sims.append(func(seg1, gold))
                avg['gold2'] = float(np.mean(_sims))
            g_sims[func.__name__] = avg
                # g_sims["agreement"] = float(np.mean([segeval.boundary_similarity(dataset["gold"][str(i)],
                #                                                                  dataset["gold2"][str(i)]) for i in r]))
    # TODO what about fleiss kappa? how do I take only the gold in the dataset?
    logging.info(g_sims)
    # print(g_sims)
    # with open(path + 'gsims.json', 'w+') as outfile:
    #     json.dump(g_sims, outfile)
    return g_sims


def evaluate_topics(doc, gold_doc, evaluate_dynamic=False, evaluate_gold=False, with_classifier=True, method="gestalt"):
    """
    Evaluate using edit distance, for the given doc and for a random (markov) assignment, both compared to the gold_doc
    :param doc:
    :param gold_doc:
    :param evaluate_dynamic: whether to evaluate with the dynamic method also
    :return:
    """
    if evaluate_dynamic:
        # logging.info(f"Dynamic topic scores:")
        # logging.info(gold_doc._.topics)
        # logging.info(doc._.topics)
        dynamic_score = topics_score(doc._.topics, encoder.transform(gold_doc._.topics), method=method)
        logging.info(dynamic_score)
    if evaluate_gold:
        c_score = topics_score(encoder.transform(doc._.topics), encoder.transform(gold_doc._.topics), method=method)
        logging.info(c_score)

    if with_classifier:
        ta = TopicAssigner(from_classifier=True)
        logging.info(f"Classifier topic scores:")
        # logging.info(f"Markov Chain topic scores:")
        c_score = topics_score(ta.create_dynamic(doc), encoder.transform(gold_doc._.topics), method=method)
        logging.info(c_score)
        print(f"c_score: {c_score}")
    else:
        ta = TopicAssigner(name="mcc5_iner5_iter15.pkl")
        logging.info(f"Markov Chain (mixture) topic scores:")
        # logging.info(f"Markov Chain topic scores:")
        mc_score = topics_score(ta.create(doc), encoder.transform(gold_doc._.topics), method=method)
        logging.info(mc_score)

    ta2 = TopicAssigner(markov_chain=False)
    logging.info(f"Uniform topic scores:")
    print("Avoiding doubles!!")
    uni_score = topics_score(ta2.create(doc, avoid_doubles=True), encoder.transform(gold_doc._.topics), method=method)
    logging.info(uni_score)
    print(f"uni_score: {uni_score}")
    if with_classifier or evaluate_gold:
        return c_score, uni_score

    if evaluate_dynamic:
        return dynamic_score, mc_score, uni_score
    else:
        return mc_score


def with_segeval(ratio=0.75, r=None, method="max", avg=None, use_close=False, suffix="", annotators=None, bin=False):
    # segmentors = {"uniform": UniformSegmentor(),
    #               "gpt2": Gpt2Segmentor(),
    #               "nsp": NSPSegmentor()}
    # segmentors = {"uniform": UniformSegmentor()}
    segmentors = None
    logging.info("\nMade segmentors")

    gold2 = False
    logging.info(f"Ratio: {ratio}")
    logging.info("With estimated segment length: ")
    make_len_dict(ratio=ratio, fixed_segments=False, segmentors=segmentors, r=r, method=method, avg=avg,
                  use_close=use_close, suffix=suffix+"_e", annotators=annotators, bin=bin)
    seg_eval(r=r, suffix=suffix+"_e", bin=bin)
    logging.info("With fixed segment length: ")
    make_len_dict(ratio=ratio, fixed_segments=True, segmentors=segmentors, r=r, method=method,
                  use_close=use_close, suffix=suffix+"_f", annotators=annotators, bin=bin)
    # gold2 = True if annotators is not None else False
    seg_eval(r=r, suffix=suffix+"_f", gold2=gold2, bin=bin)


def main():
    r = ["sf_43019", "sf_38929", "sf_32788", "sf_38936", "sf_20505", "sf_23579", "sf_48155", "sf_35869", "sf_30751", "sf_30753",
         "sf_45091", "sf_25639", "sf_46120", "sf_32809", "sf_34857", "sf_46122", "sf_30765", "sf_24622", "sf_21550", "sf_26672"]
    # r = ["sf_43019", "sf_38929", "sf_32788", "sf_38936", "sf_20505"]
    # make_gold_xlsx(save=True, name="Yelena", r=r)
    # make_gold_xlsx(save=False, name="Yelena", r=r)
    make_gold_xlsx(save=True, name="Nicole", r=r[:7])
    # # diffs = combine_close([1,2,3], [12,2,35])
    # with_segeval(r=r, annotators=["_Y", "_N"])

def main2(alpha=0.8, beta=1e-3, use_cache=False, run_id=None, gpt2_large=False, use_nb=False, test=False, window=3,
          diff_scale=1., gpt2_path=None, num_bins=None, topic_nb=False, bin_nb=False, xlarge=False,
          use_all=False):
    if not test:
        r = ["sf_43019_Y", "sf_38929_Y", "sf_32788_Y", "sf_38936_Y", "sf_20505_Y"]
        # r = ["sf_43019_Y"]
    else:
        r = ["sf_23579", "sf_48155", "sf_35869", "sf_30751", "sf_30753", "sf_45091", "sf_25639", "sf_46120", "sf_32809",
             "sf_34857", "sf_46122", "sf_30765", "sf_24622", "sf_21550", "sf_26672"]

        r = r[:]
        make_gold_xlsx(save=True, name="Yelena", r=r)
        r = [_r + "_Y" for _r in r]
        if use_all:
            print("using all testimonies")
            r = ["sf_43019_Y", "sf_38929_Y", "sf_32788_Y", "sf_38936_Y", "sf_20505_Y"] + r
            r= r[:]

    s_es = []
    ids = []
    lengths = []
    b_s = []  # for dynamic/topics/finetuned
    b_s0 = []
    b_s_uni = []
    b_s_nsp = []
    sm = []
    sm0 = [None] * len(r)  # gpt2
    sm_uni_t = [None] * len(r)  # uniform topics (not uniform segmentation)
    sm_uni = [None] * len(r)  # uniform segments (not uniform topics)
    sm_nsp = [None] * len(r)  # nsp segments (and dynamic topics)
    sm_gold = [None] * len(r)  # gold segments (and dynamic topics)
    ed = []
    ed0 = [None] * len(r)
    ed_uni_t = [None] * len(r)
    ed_uni = [None] * len(r)
    ed_nsp = [None] * len(r)
    ed_gold = [None] * len(r)
    print(f"Scores, {alpha}, {beta}")

    for i, t in enumerate(r):
        # logging.info(t)
        print(t)
        sys.stdout.flush()
        ids.append(t)
        gold_doc = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}")
        doc2 = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}")
        doc3 = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}")
        doc4 = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}")
        doc5 = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}")
        lengths.append(len(gold_doc.spans['segments']))

        from_dict = {}
        from_dict["gold"] = gold_doc
        sm_gold[i], _ = evaluate_topics(doc2, gold_doc, method="gestalt")
        ed_gold[i], _ = evaluate_topics(doc2, gold_doc, method="edit")

        us = UniformSegmentor()
        estimated_segments = int(len(gold_doc) / 440)
        doc2 = us.segment_doc(doc2, estimated_segments)
        logging.info(f"Made uniform segmentation ")
        sys.stdout.flush()
        from_dict["uniform"] = doc2
        sm_uni[i], _ = evaluate_topics(doc2, gold_doc, method="gestalt")
        ed_uni[i], _ = evaluate_topics(doc2, gold_doc, method="edit")

        # #
        gs = Gpt2Segmentor(large=gpt2_large, xlarge=xlarge)
        print("NOT USING DYNAMIC GPT2")
        # doc3 = gs.segment_doc(doc3, estimated_segments, window=window, dynamic=True, t=t[:-2], diff_scale=diff_scale)
        doc3 = gs.segment_doc(doc3, estimated_segments, window=window, dynamic=False, t=t[:-2], diff_scale=diff_scale,
                              alpha=1-alpha)
        logging.info(f"Made GPT2 segmentation (no topics)")
        sys.stdout.flush()
        from_dict["gpt2"] = doc3
        sm0[i], sm_uni_t[i] = evaluate_topics(doc3, gold_doc, method="gestalt")
        ed0[i], ed_uni_t[i] = evaluate_topics(doc3, gold_doc, method="edit")

        if gpt2_path is None:
            if num_bins != 1 and (topic_nb or bin_nb):
                gs = Gpt2Segmentor(with_topics=True, model_name='distilroberta', run_id=run_id, large=gpt2_large,
                                   use_nb=use_nb, from_path=gpt2_path, nb_topics=topic_nb, xlarge=xlarge)
                if use_cache:
                    gs.load_cache(t=t[:-2])
                doc4 = gs.segment_doc(doc4, estimated_segments, window=window, dynamic=True, t=t[:-2], alpha=alpha, beta=beta, diff_scale=diff_scale)
                logging.info(f"Made GPT2 segmentation (with topics) ")
                from_dict["gpt2-topics"] = doc4
                sm.append(evaluate_topics(doc4, gold_doc, method="gestalt")[0])
                ed.append(evaluate_topics(doc4, gold_doc, method="edit")[0])
                # evaluate_topics(doc4, gold_doc, method="gestalt")
                # evaluate_topics(doc4, gold_doc, method="edit")
            else:
                gs = Gpt2Segmentor(large=gpt2_large, nb_topics=False, use_nb=use_nb, xlarge=xlarge)
                print("USING DYNAMIC GPT2")
                doc4 = gs.segment_doc(doc4, estimated_segments, window=window, dynamic=True, t=t[:-2],
                                      diff_scale=diff_scale, alpha=1-alpha)
                # doc3 = gs.segment_doc(doc3, estimated_segments, window=window, dynamic=False, t=t[:-2], diff_scale=diff_scale)
                logging.info(f"Made GPT2 segmentation (no topics)")
                from_dict["gpt2-topics"] = doc4
                sm.append(evaluate_topics(doc4, gold_doc, method="gestalt")[0])
                ed.append(evaluate_topics(doc4, gold_doc, method="edit")[0])
                # evaluate_topics(doc4, gold_doc, method="gestalt")
                # evaluate_topics(doc4, gold_doc, method="edit")
        else:
            gs = Gpt2Segmentor(large=gpt2_large, from_path=gpt2_path, xlarge=xlarge)
            print("gpt2-topics is from finetuning - NOT WITH TOPICS")
            # not using cache yet!!
            _t = t[:-2] if gpt2_path == "" else None
            _dynamic = True if gpt2_path == "" else False
            print(f"_dynamic: {_dynamic}")
            doc4 = gs.segment_doc(doc4, estimated_segments, window=window, dynamic=_dynamic, t=_t,
                                  diff_scale=diff_scale, alpha=1-alpha)
            logging.info(f"Made GPT2 segmentation (no topics)")
            from_dict["gpt2-topics"] = doc4
            # evaluate_topics(doc4, gold_doc, method="gestalt")
            # evaluate_topics(doc4, gold_doc, method="edit")
            sm.append(evaluate_topics(doc4, gold_doc, method="gestalt")[0])
            ed.append(evaluate_topics(doc4, gold_doc, method="edit")[0])

        nsps = NSPSegmentor()
        # doc5 = nsps.segment_doc(doc5, estimated_segments, window=window, t=t[:-2], dynamic=False)
        doc5 = nsps.segment_doc(doc5, estimated_segments, window=window, t=t[:-2], dynamic=True, alpha=1-alpha)
        logging.info(f"Made NSP segmentation - NOT dynamic")
        from_dict["nsp"] = doc5
        sm_nsp[i], _ = evaluate_topics(doc5, gold_doc, method="gestalt")
        ed_nsp[i], _ = evaluate_topics(doc5, gold_doc, method="edit")

        # logging.info(accu_scores(doc5, gold_doc))
        # logging.info(windowdiff(doc5, gold_doc, k=int(0.5*len(list(gold_doc.sents))/len(gold_doc.spans["segments"]))))
        make_len_dict(from_dict=from_dict, suffix="_topic", r=t, run_id=run_id)
        s_e = seg_eval(r=[t], suffix="_topic", run_id=run_id)
        # print(s_e['boundary_similarity']['gpt2-topics'])
        # print(logging.info(s_e['boundary_similarity']['gpt2-topics']))
        print(s_e)
        s_es.append(s_e)
        # logging.info(s_e)
        b_s.append(s_e['boundary_similarity']['gpt2-topics'])
        b_s0.append(s_e['boundary_similarity']['gpt2'])
        b_s_uni.append(s_e['boundary_similarity']['uniform'])
        b_s_nsp.append(s_e['boundary_similarity']['nsp'])
        sys.stdout.flush()
    print(f"Alpha: {alpha}, Beta: {beta}")
    print(f"\n\"ids\": {ids}")
    print(f"\n\"Lengths\": {lengths}")
    print(f"\n\"Boundary scores (dynamic)\": {b_s}")
    print(f",\"Boundary scores (gpt)\": {b_s0}")
    print(f",\"Boundary scores (nsp)\": {b_s_nsp}")
    print(f",\"Boundary scores (uniform segments)\": {b_s_uni}")
    print(f",\"Sequence Matching (dynamic)\": {sm}")
    print(f",\"Sequence Matching (gpt)\": {sm0}")
    print(f",\"Sequence Matching (gpt2 segments, uniform topics)\": {sm_uni_t}")
    print(f",\"Sequence Matching (uniform segments, dynamic topics)\": {sm_uni}")
    print(f",\"Sequence Matching (nsp segments, dynamic topics)\": {sm_nsp}")
    print(f",\"Sequence Matching (gold segments, dynamic topics)\": {sm_gold}")
    print(f",\"Edit distance (dynamic)\": {ed}")
    print(f",\"Edit distance (gpt)\": {ed0}")
    print(f",\"Edit distance (gpt2 segments, uniform topics)\": {ed_uni_t}")
    print(f",\"Edit distance (uniform segments, dynamic topics)\": {ed_uni}")
    print(f",\"Edit distance (nsp segments, dynamic topics)\": {ed_nsp}")
    print(f",\"Edit distance (gold segments, dynamic topics)\": {ed_gold}")
    # logging.info(b_s)
    # logging.info(b_s0)
    print(s_es)
    print("\n")


def inter_annotator(rs=7, run_id=None):
    r = ["sf_43019", "sf_38929", "sf_32788", "sf_38936", "sf_20505", "sf_23579", "sf_48155"][:rs]

    b_s = []
    sm = [None] * len(r)
    ed = [None] * len(r)
    from_dict = {}
    for i, t in enumerate(r):
        gold = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}_Y")
        gold2 = Doc(Vocab()).from_disk(f"/cs/snapless/oabend/eitan.wagner/segmentation/data/gold_docs/doc_{t}_N")
        from_dict["gold"] = gold
        from_dict["gold2"] = gold2
        make_len_dict(from_dict=from_dict, suffix="_golds", r=t, run_id=run_id)
        s_e = seg_eval(r=[t], suffix="_golds", run_id=run_id, gold2=True)
        b_s.append(s_e['boundary_similarity']['gold2'])
        sm[i], _ = evaluate_topics(gold2, gold, method="gestalt", evaluate_gold=True)
        ed[i], _ = evaluate_topics(gold2, gold, method="edit", evaluate_gold=True)

    print(f"\n\"Boundary scores\": {b_s}")
    print(f"\n\"Sequence Matching\": {sm}")
    print(f"\n\"Edit distance\": {ed}")

if __name__ == '__main__':
    # import logging
    logging.basicConfig(level=logging.INFO)

    # main()
    # inter_annotator(rs=7, run_id=str(0))


    import sys
    run_id = str(sys.argv[1])
    test = False
    window = 3
    gpt2_path = None
    sizes = ['base', 'large', 'xl', 'j']
    # sizes = ['j']
    if len(sys.argv) > 2:
        if sys.argv[2] == "test":
            test = True
            # print("Testing only 10")
            print("Testing 15")
        else:
            window = int(sys.argv[2])
        if len(sys.argv) > 3:
            gpt2_path = sys.argv[3]
            if gpt2_path in sizes:
                sizes = [gpt2_path]
                gpt2_path = None

    for size in sizes:
        # for use_nb in [True]:
        for use_nb in [False]:
    # for size in ['base']:
            if size == 'base':
                large = False
                xlarge = False
            elif size == 'large':
                large = True
                xlarge = False
            elif size == 'xl':
                large = True
                xlarge = True
            elif size == 'j':
                large = False
                xlarge = True
            # use_nb = False
            # use_nb = True
            # topic_nb = False
            topic_nb = True
            # topic_nb = use_nb
            use_all = True
            # gpt2_path = "/cs/snapless/oabend/eitan.wagner/segmentation/models/gpt2/checkpoint-5000"
            print("Reweighted alpha and beta")
            print("Using probs by bin")

            num_bins = 10
            print(f"test: {test}")
            print(f"\n***********size: {size}****************")
            # print(f"***********xlarge: {xlarge}**************")
            print(f"************use_nb: {use_nb}***************\n")
            print(f"use_all: {use_all}")
            print(f"topic_nb (i.e. do we use topics and nb): {topic_nb}")
            print(f"num_bins: {num_bins}")
            print(f"window: {window}")
            print(gpt2_path)
            print(run_id)
            sys.stdout.flush()
            diff_scale = 1.
            if use_nb:
                if topic_nb:
                    lengths.train(by_topic=True, smooth=0.9, num_docs=5)
                else:
                    lengths.train(by_topic=False, smooth=0.9, num_docs=5, num_bins=10)
            # for i, (alpha, beta) in enumerate([(0.25, 0.25), (0.01, 0.01), (0.49, 0.49), (0.15, 0.15), (0.35, 0.35)]):
            # for i, (alpha, beta) in enumerate([(0.5-1e-2, 1e-2), (0.4, 0.1), (0.3, 0.2), (0.2, 0.3), (0.1, 0.4), (1e-2, 0.5-1e-2)
            #                                       , (0.25, 0.25)]):
            # for i, (alpha, beta) in enumerate([(0.01, 0.01), (0.5, 0.1), (0.8, 0.05)]):
            # for i, (alpha, beta) in enumerate([(0.1, 0.3), (0.1, 0.5), (0.1, 0.7), (0.1, 0.8)]):
            for i, (alpha, beta) in enumerate([(0.2, 0.2)]):
                #     for diff_scale in [0.1, 0.50, 1e-2, 1e-3]:
                # , (0.5-1e-2, 1e-2), (0.4, 0.1),
                # (0.3, 0.2), (0.2, 0.3), (0.1, 0.4), (1e-2, 0.5-1e-2)]):
                # for alpha in [0.4, 0.3, 0.2, 1e-2, 0.5-1e-5]:
                # for alpha in [0.5]:
                #     for beta in [0.1, 0.2, 0.3, 0.4, 0.5 - 1e-5, 1e-2]:
                # for beta in [1e-2]:
                #     logging.info(f"Alpha: {alpha}, Beta: {beta}")
                print(f"Alpha: {alpha}, Beta: {beta}")
                print(f"diff_scale: {diff_scale}")
                sys.stdout.flush()
                # if i == 0:
                #     main2(alpha=alpha, beta=beta, use_cache=False, run_id=run_id, gpt2_large=large, use_nb=use_nb)
                # else:
                #     main2(alpha=alpha, beta=beta, use_cache=True, run_id=run_id, gpt2_large=large, use_nb=use_nb)
                main2(alpha=alpha, beta=beta, use_cache=True, run_id=run_id, gpt2_large=large, use_nb=use_nb, test=test,
                      window=window, diff_scale=diff_scale, gpt2_path=gpt2_path, num_bins=num_bins, topic_nb=topic_nb,
                      xlarge=xlarge, use_all=use_all)
                print("\n")
                sys.stdout.flush()
