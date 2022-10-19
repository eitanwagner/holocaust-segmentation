
import numpy as np
import json
import pickle
import joblib

from shortest_path import k_shortest
from parse_sf import TestimonyParser
from parse_sf import remove_extensions
from textcat import SVMTextcat
from textcat import Vectorizer
from transformer_classification import TransformerClassifier
from topics import LDAScorer
from transitions import MC, MCClusters
from gpt2 import GPT2Scorer
import sys
import spacy
import logging
from summarize import Summarizer
from evaluation import evaluate
from evaluation import evaluate_topics
import pandas as pd
# from parse_sf import CR, SRL

from spacy.vocab import Vocab
from spacy.tokens import Doc
Doc.set_extension("topics", default=None, force=True)
Doc.set_extension("ps", default=None, force=True)
Doc.set_extension("summaries", default=None, force=True)
# we will also use doc.span["segments"]


class Segmentor:
    """
    Class for segmentation and topic assignment model
    """
    def __init__(self, i, text=None, spacy_doc=None, model=None, method='marginal', summarizer=True, window_lm=False):
        """

        :param i: testimony number
        :param text: text of the testimony. not used if spacy_doc is not None
        :param model: classifier model
        :param method: can be 'max', 'marginal', 'graph' or 'lda'
        :param summarizer: whether to do summarization
        :param spacy_doc: a Doc object to segment
        """

        # maybe we should put in the text separately

        self.i = i
        self.segments = [0]  # list of segment starts
        self.segment_spans = []  # list of segment spacy spans
        self.max_srls = None
        self.summaries = []
        self.ps = None  # list of probabilities for the segments
        self.topic_assignments = []  # list of sampled assignments
        self.method = method
        self.window_lm = window_lm

        self.gpt2scorer = GPT2Scorer()
        # self.gpt2scorer.load_cache(i)
        self.gpt2ratio = 0

        if model is not None:
            self.cats = model.topics
        else:
            self.cats = None
        self.model = model
        # self.model.load_cache(i)

        self.nlp = spacy.load("en_core_web_trf")

        if spacy_doc is not None:
            self.parser = None
        else:
            self.parser = TestimonyParser(self.nlp)
        # self.summarizer = Summarizer(self.model, use_bert_summarizer=False)# !!!
        if summarizer:
            self.summarizer = Summarizer(self.model, use_bert_summarizer=True)# !!!
        # logging.info("Using bert summarizer")
        if spacy_doc is not None:
            self.doc = spacy_doc
            self.doc.spans["srls"] = []
            self.doc.spans["clusters"] = []
            self.sents = [s for s in self.doc.spans["sents"]]
        else:
            self.doc = self.parser.parse_testimony(text)  # does CR but not srl - cancelled!!
            self.sents = list(self.doc.sents)  # these are spans!!!
            for s in self.sents:
                # self.parser.srler.add_to_Span(s, self.parser.srler.parse(s.text))
                self.parser.add_srls_to_s(s)  # only adds if SRL=True

    def combine_sents(self, window=1, ratio=.5, dynamic=False):
        """
        Combines sentences using gpt2 scores
        :param window: number of sentences to look at (before and after)
        :param ratio: combining ratio (higher means less sentences)
        :param dynamic:
        :return:
        """
        # combines the top sentences. higher ratio means combining more
        # calculates score by GPT2 and given window size
        # scorer = GPT2Scorer()
        self.gpt2ratio = ratio
        scorer = self.gpt2scorer
        diffs = []
        for j, s in enumerate(self.sents):
            if j < window or j + window >= len(self.sents):
                continue
            # gpt2_p1 = scorer.sentence_score(" ".join(self.sents[j-window:j+window]))
            gpt2_p1 = scorer.sentence_score(self.doc[self.sents[j-window].start:self.sents[j+window].start].text)
            # gpt2_p2 = scorer.sentence_score(" ".join(self.sents[j-window:j])) \
            #           + scorer.sentence_score(" ".join(self.sents[j:j+window]))
            gpt2_p2 = scorer.sentence_score(self.doc[self.sents[j-window].start:self.sents[j].start].text) \
                      + scorer.sentence_score(self.doc[self.sents[j].start:self.sents[j+window].start].text)
            diffs.append((gpt2_p1 - gpt2_p2, j))
        self.gpt2scorer.save_cache()

        # the loss is the minus (log) probability!!! so we want to merge where the diff is small (more negative)
        # diffs.sort(reverse=True)
        diffs.sort(reverse=False)
        js = sorted([d[1] for d in diffs[:int(len(diffs) * ratio)]], reverse=True)
        for j in js:
            # self.sents[j-1] = self.sents[j-1] + ' ' + self.sents[j]
            # self.sents[j-1] = self.doc[self.sents[j-1].start:self.sents[j].start]
            self.sents[j-1] = self.doc[self.sents[j-1].start:self.sents[j].end]
            self.sents[j] = None
        self.sents = [s for s in self.sents if s is not None]
        logging.info("Combined sentences")
        # logging.info(str([0 if i in js else -1 for i in range(len(self.doc.spans["sents"]))]))


    def second_pass(self, threshold=0.3, path='/cs/snapless/oabend/eitan.wagner/segmentation/models/deberta-large'):
        """
        Perform a second gpt2 pass after the dynamic and topic sampling, for places where the correlation between the topics is high
        :return:
        """
        def best_break(gs, start, end):
            sent0 = [s.start for s in self.sents].index(start)
            sent1 = [s.end for s in self.sents].index(end)
            return gs.segment_sents(self.doc, self.sents[sent0:sent1+1], num_segments=2)

        from evaluation import Gpt2Segmentor
        gs = Gpt2Segmentor()

        cor_matrix = np.load(path + "/correlation_matrix.npy")
        for i, s in enumerate(self.segment_spans[:-1]):
            if cor_matrix[self.doc._.topics[i], self.doc._.topics[i+1]] > threshold:
                start, end = self.doc.spans["segments"][i].start, self.doc.spans["segments"][i+1].end
                bb = best_break(gs, start, end)
                self.segment_spans[i] = bb[0]
                self.segment_spans[i+1] = bb[1]
                self.doc.spans["segments"] = self.segment_spans
        return


    def from_list(self, l):
        """
        Add a segmentation from a list. This can be done after combining sents
        :param l: list of (segment_end, topic) tuples
        :return:
        """
        _l = list(zip(*l))
        topics = list(_l[1])
        ends = list(_l[0])
        starts = [0] + ends[:-1]
        # logging.info(starts)
        # logging.info(ends)
        # logging.info(topics)
        # logging.info(len(self.sents))
        # logging.info(self.sents)

        segment_spans = [self.doc[self.sents[s].start:self.sents[e-1].end] for s, e in zip(starts, ends)]

        ps = []
        for t in topics:
            # p = np.full(len(self.cats), -np.inf)  # zeros for all
            # p[t] = np.log(1)  # 1 for the correct topic
            p = np.zeros(len(self.cats))  # zeros for all
            p[t] = 1.  # 1 for the correct topic
            ps.append(p)

        self.doc.user_data = {}
        self.segment_spans, self.ps = segment_spans, ps
        self.doc.spans["segments"], self.doc._.ps = segment_spans, ps
        return segment_spans, ps

    def segment_score(self, start, end, prev_topic=None):
        """
        calculate the log-probability for this segment. Uses the method of the class ("max", "marginal" or "graph")
        :param start: first sentence id (included)
        :param end: last sentence id (not included)
        :param prev_topic: topic of previous segment
        :return: for "max" - tuple of probability of max class, probability vector, and max class id
            for "marginal" - tuple of marginal probability and the classification probability vector
            for "graph" - list of probabilities of len num_topics
        """
        # create span

        span = self.doc[self.sents[start].start:self.sents[end-1].end]  # we don't want the end sentence too
        spans = None
        if self.window_lm:
            spans = self.sents[start:end]  # in this case it's spans and not a span
        # get features - not used!
        if self.parser is not None:
            span._.feature_vector = self.parser.make_new_features(span, bin=int(5 * (start + end) / len(self.sents)))  # the bin is by the middle
        # get probability by model
        if self.method == 'max':
            # try:
                return self.model.predict_max(span, prev_topic=prev_topic, spans=spans)
            # except RuntimeError:  # too much memory used
            #     logging.info(f"Sequence too long. start: {start}, end: {end}, len: {len(span)}")
            #     return -np.inf, [], 0
            # except IndexError:
            #     logging.info(f"Index Error. len: {len(span)}")
            #     return None
        elif self.method == 'graph':
            # try:
                return self.model.predict_all(span, prev_topic=prev_topic)
            # except RuntimeError:  # too much memory used
            #     logging.info(f"Sequence too long. start: {start}, end: {end}, len: {len(span)}")
            #     return []
            # except IndexError:
            #     logging.info(f"Index Error. len: {len(span)}")
            #     return None

        # for 'marginal' and 'lda'
        return self.model.predict(span)

    def find_segments(self, use_heuristic=True, with_topics=True):
        """
        Segments the document
        :param with_topics:
        :param use_heuristic:
        :param sents: list of sentences
        :return: list of firsts in segments (always starting from 0), and list of states
        """
        prev_v = [0]  # a list of previous vertices. so 0 will also have a predecessor
        prev_states = [None]  # a list of previous vertices. so 0 will also have a predecessor
        probs = [None]  # a list of probability vectors for each topic with the optimal prev_v
        prev_score = [0]  # a list of scores with the optimal previous
        last = 0  # for heuristic. maybe relax to use previous last

        logging.info(f"len: {len(self.sents)}")
        for i in range(1, len(self.sents)+1):
            if i % 100 == 0 or len(self.sents) - i < 2:
                logging.info(f"i: {i}")
                self.model.save_cache()

            js = range(last, i)
            # find previous nodes with respective scores
            if self.method == 'max':
                prevs = [(prev_score[j], ) + self.segment_score(j, i, prev_topic=prev_states[j]) for j in js]
            else:
                prevs = [(prev_score[j], ) + self.segment_score(j, i) for j in js]
            p_costs = [p[0] + p[1] for p in prevs]  # these are actually positive scores so we want the max

            # find best prev
            # prevent same segment??
            m = int(np.argmax(p_costs))
            prev_score.append(p_costs[m])
            # print("prev_state:", prev_state)
            if with_topics:
                probs.append(prevs[m][2])  # this should be a vector. the log probabilities
            if self.method == 'max':
                prev_states.append(prevs[m][3])  # this should be an integer
                if with_topics:
                    probs[-1] = -np.inf * np.ones(len(self.cats))
                    probs[-1][prevs[m][3]] = 0.

            prev_v.append(m + last)
            if use_heuristic:
                last += m

        # backward pass - change this for method='max' !!!!!
        # for now we will just use a categorical distribution
        logging.info(len(prev_v))
        # segments = []
        segment_spans = []
        i = len(prev_v) - 1  # why -1??

        if with_topics:
            p = np.exp(probs[-1])  # this is the probability vector
            ps = [p]
        while i > 0:
            segment_spans = [self.doc[self.sents[prev_v[i]].start:self.sents[i-1].end]] + segment_spans
            i = prev_v[i]
            # segments = [i] + segments
            if i > 0 and with_topics:
                p = np.exp(probs[i])
                ps = [p] + ps

        # self.segments, self.ps = segments, ps

        # removing all user data
        self.doc.user_data = {}
        self.segment_spans = segment_spans
        if with_topics:
            self.ps = ps
            self.doc._.ps = ps
        self.doc.spans["segments"] = segment_spans

        # return segments, ps
        # logging.info("Probabilities: ")
        # logging.info(ps)
        if with_topics:
            return segment_spans, ps
        return segment_spans

    def make_probs(self, window=15, from_file=False, name=''):
        """
        Makes all probabilities for the find_segments_k function.
        The first index is included and the last is not (i.e. (0, -1), (5, 5) means from the first to the fifth with topic 5)
        :return:
        """
        if from_file:
            with open(name, 'r') as infile:
                return json.load(infile)

        logging.info("Window: " + str(window))
        probs = []
        n = len(self.sents)
        for i in range(n):
            logging.info(f"i: {i}")
            self.model.save_cache()
            # for j in range(i+1, min(i + window, n+2)):  # n+1 is for the end topic
            for j in range(i+1, min(i + window, n+1)):  # without the end topic
                topics = list(range(len(self.cats)))
                if i == 0:
                    # topics = topics + [-1]
                    topics = [-1]
                for t1 in topics:
                    # if j == n + 1:
                    #     probs = probs + [[[i, t1], [j, -1], {"weight": 0.}]]
                    # else:
                        probs = probs + [[[i, t1], [j, t2], {"weight": p}] for t2, p in
                                         enumerate(self.segment_score(start=i, end=j, prev_topic=t1)) if p != -np.inf]
        probs = probs + [[[n, t1], [n+1, -1], {"weight": 0.}] for t1 in list(range(len(self.cats)))]


        if name != "":
            with open(name, "w+") as outfile:
                json.dump(probs, outfile)
        return probs

    def find_segments_k(self, k, probs=None, window=15):
        """
        Finds segmentation and topics (max method) for a given number of segments (k)
        :param k: number of segments
        :return: list of segments (as spans) and list of topoic probability vectors (for each segment)
        """
        if probs is None:
            probs = self.make_probs(window=window)
        path = k_shortest(n=len(self.sents), probs=probs, k=k)
        logging.info(path)

        self.topic_assignments = [[p[1] for p in path]]
        self.ps = []
        for i, t in enumerate(self.topic_assignments):
            _ps = np.full(len(self.cats), -np.inf)
            _ps[t] = 0.
            self.ps.append(_ps)
        ends = [p[0] for p in path]
        # starts = [0] + [e+1 for e in ends[:-1]]
        starts = [0] + [e for e in ends[:-1]]
        self.segment_spans = [self.doc[self.sents[s].start:self.sents[e-1].end] for s, e in zip(starts, ends)]

        self.doc.user_data = {}  # removing all user data
        self.doc.spans["segments"], self.doc._.ps = self.segment_spans, self.ps
        return self.segment_spans, self.ps

    def sample_topics(self, num=3, allow_doubles=False):
        """
        Samples topic assignments for a segmented document
        :param num: number of assignments to find
        :param allow_doubles: whether to allow the same topic twice
        :return: the assigments found
        """
        # returns random topic assignments, without doubles, for the requested amount
        def has_doubles(l):
            for i in range(1, len(l)):
                # for i, v in enumerate(l[1:], start=1):
                #     if v == l[i-1]:
                if l[i] == l[i-1]:
                    return True
            return False

        attempts = 0
        found = 0
        # topic_assignments = []
        while attempts < 5000:
            topics = [np.random.choice(a=len(p), p=p/p.sum()) for p in self.ps]
            if allow_doubles or not has_doubles(topics):
                logging.info("Found a topic assignment")
                found += 1
                self.topic_assignments.append(topics)
                if found == num:
                    self.doc._.topics = self.topic_assignments[0]
                    return self.topic_assignments
            attempts += 1

        if len(self.topic_assignments) > 0:  # in the marginal case it might be hard to find topics without doubles!!
            self.doc._.topics = self.topic_assignments[0]
        return self.topic_assignments

    def find_srls(self, topic_assignment, count=1):  # not used
        # finds srl units for each segment, and ranks them according to the connection with the chosen topic_assignment
        # returns a list (even if count=1)
        # TODO: for now this is only used for the first topic assignment. maybe we can use the assignment distribution??
        self.max_srls = []
        for span, topic in zip(self.segment_spans, topic_assignment):
            self.parser.srler.add_to_new_span(span)  # this was probably done already
            # srls, first_last = self.parser.srler.parse_simple(span.text)
            # span._.srls = [span[first:last+1] for first, last in first_last]
            for s in span._.srls:
                s._.feature_vector = self.parser.make_new_features(s, bin=int(5 * (s.start + s.end) / len(self.doc)))
            if span._.srls is not None:
                # should we use priors? should we sample?
                sorted_srls = sorted(zip([self.model.predict(srl)[1][topic] for srl in span._.srls], range(len(span._.srls))), reverse=True)
                self.max_srls.append([span._.srls[i] for _, i in sorted_srls[:count]])
            else:
                self.max_srls.append([])
        return [[srl.text for srl in srls] for srls in self.max_srls]  # this is the texts

    def find_summaries(self, topic_assignment, count=1, ratio=None):
        """
        Created summaries for each segment separately.
        :param topic_assignment:
        :param count:
        :param ratio:
        :return:
        """
        for span, topic in zip(self.segment_spans, topic_assignment):
            s = self.summarizer.get_ranked_sents(span.as_doc(), max_depth=6, class_num=topic, count=count, simple=False, ratio=ratio)
            logging.info(f"Summary: {s}")
            self.summaries.append(s)  # is topic the same as the label for the model???

        self.doc._.summaries = self.summaries
        return self.summaries

    def print_segments(self, name=None):
        """
        Prints the segments
        :param name: for saving (does not save if None)
        :return:
        """
        # segments is list of first sents
        if name is not None:
            f = open(name, "a+")
        # for assignment in self.topic_assignments:
        for i, segment in enumerate(self.segment_spans):
            logging.info(f"************************** Segment {i}, topic: {[self.cats[assignment[i]] for assignment in self.topic_assignments]} ***********************")
            if self.max_srls is not None and len(self.max_srls[i]) > 0:
                logging.info(f"************************** srls (for first): {[srl.text for srl in self.max_srls[i]]} ***********************")
            elif len(self.summaries[i]) > 0:
                logging.info(f"************************** summaries (for first): {[summary[1] for summary in self.summaries[i]]} ***********************")
                # logging.info(f"************************** summaries (for first): {self.summaries[i]} ***********************")
            else:
                logging.info(f"************************** {None} ***********************")
            logging.info(segment.text)
            if name is not None:
                f.write(segment.text)
        if name is not None:
            f.close()

        avg_len = np.mean([len(s) for s in self.segment_spans])
        var = np.var([len(s) for s in self.segment_spans])
        logging.info(f"Average segment length (in spacy tokens): {avg_len}")
        logging.info(f"Segment length variance (in spacy tokens): {var}")


    def get_for_eval(self):
        """
        Makes a dataframe with 3-segments for evaluation
        :return:
        """
        ks = np.random.choice(range(1, len(self.segment_spans)-1, 2), size=5, replace=False)
        texts = []
        topics = []
        summaries = []
        for k in ks:
            text1 = "\n".join([s.text.strip() for s in self.segment_spans[k-1].as_doc().sents])
            text2 = "\n".join([s.text.strip() for s in self.segment_spans[k].as_doc().sents])
            text3 = "\n".join([s.text.strip() for s in self.segment_spans[k+1].as_doc().sents])
            texts.append("\n************\n".join([text1,text2,text3]))
            topics.append([self.cats[ta[k]] for ta in self.topic_assignments])
            summaries.append([s[1] for s in self.summaries[k]])
        df = pd.DataFrame({"testimony": self.i, "index": ks, "texts": texts, "topics": topics,
                           "summaries": summaries})
        return df

    def save_doc(self, path="/cs/snapless/oabend/eitan.wagner/segmentation/out_docs", bin=False):
        # remove_extensions()
        # self.doc.spans.pop("clusters", None)
        # self.doc.spans.pop("srls", None)
        # self.doc.spans.pop("segments", None)
        # self.doc._.trf_data = None
        if bin:
            self.doc.to_disk(path + "doc_" + str(self.i) + "_" + str(self.gpt2ratio) + "_b")
        else:
            self.doc.to_disk(path + "doc_" + str(self.i) + "_" + str(self.gpt2ratio))

    def segmentation_ll(self, doc):
        # computes the log likelihood for a given doc with segmentation (in doc.spans["segments"]
        # we will use the likelihood for the max topic-assigment (and not marginalize?)
        # this is exactly like fitting an HMM

        # calculate all classification probs. together with P(X)/P(t) these are the emission probabilites
        #
        return


def get_testimony_sents(i, data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Obatin a list of sentences for testimony i (in the Yale corpus)
    :param i:
    :param data_path:
    :return:
    """
    with open(data_path + 'sents.json', 'r') as infile:
        sents = json.load(infile)[str(i)]
    return sents

def get_testimony_text(i, data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Obatin the raw text of testimony i (in the Yale corpus)
    :param i:
    :param data_path:
    :return:
    """
    with open(data_path + 'raw_text.json', 'r') as infile:
        text = json.load(infile)[str(i)]
    return text

def get_gold_text(i, data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Obatin the raw text of gold_testimony i
    :param i:
    :param data_path:
    :return:
    """
    gold_doc = Doc(Vocab()).from_disk(data_path + 'gold_docs/doc_' + str(i))
    return gold_doc.text

def get_sf_testimony_text(i, data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Obatin a list of sentences for testimony i (in the SF corpus)
    :param i:
    :param data_path:
    :return:
    """
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        text = json.load(infile)[str(i)]
    return text

def get_sf_testimony_nums(data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Get list of testimony ids for the SF corpus
    :param data_path:
    :return:
    """
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        nums = list(json.load(infile).keys())
    return nums

def get_testimony_nums(data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Get list of testimony ids for the Yale corpus
    :param data_path:
    :return:
    """
    with open(data_path + 'raw_text.json', 'r') as infile:
        nums = list(json.load(infile).keys())
    return nums

def split_eval_test(r, n):
    """
    Choose the eval set
    :param r: range of all gold docs
    :param n: number of docs for eval set
    :return: list of docs for eval
    """
    eval = list(np.random.choice(r, size=n, replace=False))
    logging.info(f"Eval set: {eval}")
    return eval

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })


    gpt2_window, gpt2_ratio, model_id = sys.argv[1], sys.argv[2], sys.argv[3]
    method = 'max'
    if len(sys.argv) > 4:
        method = sys.argv[4]

    suffix1 = str(gpt2_ratio) + method

    logging.info("\n\n**************************\n")
    smoothing_factor = 1
    # with_bin = False
    with_bin = True
    use_gpt2 = False
    logging.info(f"With bin: {with_bin}")
    no_mc = False
    use_saved = False
    use_close = False
    second_pass = False
    with_test = False  # when this is True then we check only the eval set
    if method[-3:] == 'flm':
        use_gpt2 = True
        method = method[:-4]
    if method[-5:-3] == 'sp':
        threshold = float(method[-3:])
        method = method[:-6]
        second_pass = True
        logging.info(f"Threshold: {threshold}")
    if method[-2:] == 'sp':
        method = method[:-3]
        second_pass = True
        threshold = 0.
        logging.info(f"Threshold: {threshold}")
    # if method[-1:] == 'c':
    #     method = method[:-2]
    #     use_close = True
    if method[-1:] == 's':
        method = method[:-2]
        use_saved = True
    # no_mc = False
    if method[-4:] == 'nomc':
        method = method[:-5]
        no_mc = True
    if method[-3:] == 's.5':
        method = method[:-4]
        smoothing_factor = 0.5
    if method[-1:] == '1':
        method = method[:-2]
        np.random.seed(1)
    else:
        np.random.seed(0)  # gave set 102, 107, 109, 110, 115
    if method[-4:] == 'test':
        method = method[:-5]
        with_test = True

    logging.info(f"Using saved: {use_saved}")
    logging.info(f"Using closeness: {use_close}")
    logging.info(f"Using second_pass: {second_pass}")
    # if method != 'lda':
    # r = ["sf_43019", "sf_38929", "sf_32788", "sf_38936", "sf_20505", "sf_23579", "sf_48155", "sf_35869", "sf_30751", "sf_30753",
    #      "sf_25639", "sf_45091", "sf_32809", "sf_34857"]
    r = ["sf_43019", "sf_38929", "sf_32788", "sf_38936", "sf_20505", "sf_23579", "sf_48155", "sf_35869", "sf_30751", "sf_30753",
         "sf_45091", "sf_25639", "sf_46120", "sf_32809", "sf_34857", "sf_46122", "sf_30765", "sf_24622", "sf_21550", "sf_26672"]

    # in "sf_46120", "sf_46122", "sf_30765", "sf_24622", "sf_21550", "sf_26672"
    r = [_r+"_Y" for _r in r]
    if True:
        logging.info("Using a test set")
        logging.info(f"with_test: {with_test}")
        # with_test = True

        # evals = split_eval_test(r=r, n=5)
        # evals = split_eval_test(r=range(101, 116), n=5)
        # logging.info(evals)
        # evals = ["sf_43019", "sf_38929", "sf_32788", "sf_38936"]
        # evals = [_e+"_Y" for _e in evals]
        evals = r[:10]


    # scale = 150
    # TODO these need to be updated with the new data
    scale = 200
    nt_scale = 20

    # all_mean, nt_mean = 262.2, 67.3
    # all_mean, nt_mean = 366, 67.3
    # all_mean, nt_mean = 350, None
    all_mean, nt_mean = 430, None
    mean_scale = 1

    find_min = False
    if len(sys.argv) > 5:
        find_min = True
        # use_gpt2 = True
        scale = int(sys.argv[5])
    if len(sys.argv) > 6:
        mean_scale = float(sys.argv[6])
    if nt_mean is not None:
        all_mean, nt_mean = all_mean/mean_scale, nt_mean/mean_scale

    # gpt2_window, gpt2_ratio, model_id = 2, 0.5, "trf"
    logging.info("\n\n\nStarting")
    logging.info("Method: " + method)
    logging.info(f"*********model: {model_id}******************")
    logging.info(f"GPT2 window: {gpt2_window}")
    logging.info(f"GPT2 ratio: {gpt2_ratio}")
    batch = False
    # if method != 'lda':
    #     logging.info(f"Batch: {batch}")
    #     logging.info(f"mean: {all_mean}")
    #     logging.info(f"scale: {scale}")
    #     logging.info(f"nt_mean: {nt_mean is not None}")
    # sys.stdout.flush()

    # model_id = sys.argv[3][:3]
    # if model_id[:3] == "svm":
    #     model_id = "-" + model_id[:3]

    # model = SpacyCat(model_id=model_id)
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    # model = SVMTextcat(base_path=base_path).from_path()
    # model = TransformerClassifier(base_path=base_path, model_name='distilbert-textcat')
    if method == "graph" or no_mc:
        # mc = MC(base_path=base_path, name="models/transitions/mc.json")
        mc = None
        logging.info(f"Markov chain: {mc is not None}")
    else:
        # with open(base_path + 'models/transitions/mcc5_iner5.pkl', 'rb') as infile:
        with open(base_path + 'models/transitions/mcc5_iner1_iter15_data5.pkl', 'rb') as infile:
            # with open(base_path + 'models/transitions/mcc5_iner5_iter15.pkl', 'rb') as infile:
            mc = joblib.load(infile)
        # mc = MCClusters(k=5).load()
        logging.info(f"Markov chain ******** 5 clusters **********: {mc is not None}")
        # logging.info(f"Markov chain ******** 10 clusters **********: {mc is not None}")


    scales = [scale]
    if gpt2_ratio == "0.75":
        scales = [200]
    elif gpt2_ratio == "0.0":  # ratio=0.0
        scales = [150]
    if not with_test and not with_bin:  # using test set
        if gpt2_ratio == "0.75":
            # scales = [150, 175, 200, 225, 250]
            scales = [1500, 1000, 550, 450, 350, 250]
        if gpt2_ratio == "0.8":
            scales = [200]
        if gpt2_ratio == "0.5":
            scales = [150, 175, 200]
        elif gpt2_ratio == "0.0":  # ratio=0.0
            scales = [100, 125, 150]
    if with_bin:
        logging.info("scale not used")

    for scale in scales:
        suffix = f"s{scale}_" + suffix1
        logging.info(f"\n\nScale: {scale}")

        nums = get_sf_testimony_nums()
        # nums = get_testimony_nums()
        # r = range(112, 115)
        # r = nums[10:12]
        # r = range(101, 121)
        if not with_test:
            r = [t for t in r if t not in evals]
        # r = range(111, 116)
        # evals = []
        if method == "graph":
            # r = range(104, 105)
            r = range(106, 107)
        if with_test:
            r = evals
        # r = [110, 109]

        if find_min:
            logging.info("Using saved probabilities")
        if method == 'lda':
            num_topics = int(model_id)
            logging.info(f"Using lda model, num_topics: {num_topics}")
            model = LDAScorer(num_topics=num_topics, base_path=base_path)
        elif method == "graph" or use_gpt2:
            logging.info("Using full LM scores")
            model = TransformerClassifier(base_path=base_path, model_name=model_id, mc=mc, full_lm_scores=True)
        else:
            logging.info("No full LM scores")
            model = TransformerClassifier(base_path=base_path, model_name=model_id, mc=mc, full_lm_scores=False, use_bins=with_bin)

        # scale = 20
        # model.find_priors(mean=256.29, scale=scale)  # the mean for spacy token count with /200
        # logging.info(f"calculated priors. Scale: {scale}")
        if method != 'lda':
            model.find_priors(mean=all_mean, scale=scale, nt_mean=nt_mean, nt_scale=nt_scale, r=r, smoothing_factor=smoothing_factor)
            # logging.info(f"calculated priors with separate for no topic. Scales: {scale}, {nt_scale}")
            logging.info(f"calculated priors with separate for no topic - {nt_mean is not None}")
        # summary_ratio=.15
        # logging.info(f"Summary_ratio: {summary_ratio}")


        # all_segments = {}
        # dfs = []
        num_segs = []
        methods = ["dynamic", "uniform", "gpt2", "nsp", "gpt2_dynamic"]
        topic_methods = ["dynamic", "mc", "uniform"]
        # topic_methods = ["gestalt", "edit"]

        accu_scores = {m: [] for m in methods}
        wd_scores = {m: [] for m in methods}

        topic_scores = {"gestalt": {m: [] for m in topic_methods}, "edit": {m: [] for m in topic_methods}, "num_segments": []}
        # gest_scores = {m: [] for m in topic_methods}
        # edit_scores = {m: [] for m in topic_methods}
        # edit_scores = []
        # edit_scores_mc = []
        # edit_scores_uni = []

        if method == "precomputed":
            r = [101, 103, 104, 105]
        for i in r:
            # if not with_test:
            #     if i in evals:
            #         continue

            if method == "precomputed":
                if i == 101:
                    l = [(5, 35), (6, 14), (7, 35), (8, 50), (9, 39), (10, 50), (11, 35), (22, 50), (23, 35), (24, 14), (25, 35), (39, 47), (46, 20), (60, 50), (61, 35), (62, 14), (63, 35), (64, 14), (78, 50), (90, 39), (91, 35), (103, 50), (117, 1), (118, 50), (132, 14), (133, 8), (134, 50), (147, 43), (148, 50)]
                elif i == 103:
                    l = [(11, 45), (25, 47), (26, 35), (40, 10), (41, 50), (48, 47), (62, 10), (63, 35), (64, 50), (65, 35), (66, 38), (80, 10), (81, 35), (95, 14), (97, 43), (111, 0), (122, 22), (136, 0), (150, 47), (164, 43), (173, 45), (174, 43), (180, 0), (194, 43), (195, 50), (196, 35), (198, 50), (199, 43), (203, 45), (205, 43), (206, 35), (207, 14), (209, 35), (210, 50)]
                elif i == 104:
                    l = [(14, 45), (23, 50), (30, 8), (41, 20), (55, 38), (68, 47), (69, 50), (83, 39), (84, 35), (98, 50), (99, 38), (100, 35), (101, 50), (115, 30), (129, 38), (137, 30), (151, 0), (152, 50), (153, 43), (154, 35), (155, 0), (156, 22), (157, 8), (171, 38), (185, 47), (186, 35), (200, 43), (201, 35), (202, 43), (203, 35), (204, 14)]
                elif i == 105:
                    l = [(14, 14), (15, 35), (27, 10), (39, 38), (51, 10), (52, 38), (66, 17), (80, 0), (93, 1), (107, 11), (121, 1), (135, 11), (139, 1), (151, 11), (165, 43), (167, 50), (168, 35)]
                else:
                    continue

            logging.info(f'\n\n\nTestimony {i}:')
            # d = Segmentor(i=i, text=get_testimony_text(i)[:], model=model, method='max')

            data_path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/'
            if use_saved:
                doc = Doc(Vocab()).from_disk(base_path + "out_docs_" + method + "/doc_" + str(i) + "_" + str(gpt2_ratio))
                gold_doc = Doc(Vocab()).from_disk(data_path + 'gold_docs/doc_' + str(i))
            else:
                doc = Doc(Vocab()).from_disk(data_path + 'gold_docs/doc_' + str(i))
                gold_doc = Doc(Vocab()).from_disk(data_path + 'gold_docs/doc_' + str(i))
                d = Segmentor(i=i, text=get_gold_text(i)[:], model=model, method=method, spacy_doc=doc, summarizer=False, window_lm=use_gpt2)
                # d = Segmentor(text=get_sf_testimony_text(i)[:], model=model)
                if float(gpt2_window) > 0:
                    d.combine_sents(window=int(gpt2_window), ratio=float(gpt2_ratio))

                logging.info("\nFinding segments: ")
                if method == "max":
                    c = d.find_segments()
                    logging.info("\nSampling topics: ")
                    assignments = d.sample_topics(num=1, allow_doubles=False)  # this might be an empty list!
                    if second_pass:
                        d.second_pass(threshold=threshold, path=base_path + f"models/{model_id}")
                    num_segs.append(len(c[0]))
                if method == 'lda':
                    c = d.find_segments(with_topics=False)
                    num_segs.append(len(c))

                if method == "precomputed":
                    c = d.from_list(l)
                    logging.info("\nSampling topics: ")
                    assignments = d.sample_topics(num=2, allow_doubles=False)  # this might be an empty list!

                if method == "graph":
                    logging.info("\nMaking Probabilities: ")
                    if not find_min:
                        d.make_probs(window=15, from_file=False, name=f'/cs/snapless/oabend/eitan.wagner/segmentation/probs{i}_{15}_{gpt2_ratio}')
                    else:
                        probs = d.make_probs(window=15, from_file=True, name=f'/cs/snapless/oabend/eitan.wagner/segmentation/probs{i}_{15}_{gpt2_ratio}')
                        logging.info("\nFinding best: ")
                        c = d.find_segments_k(k=len(gold_doc.spans["segments"]), probs=probs)

            if method == "max" or method == "precomputed" or method == 'lda' or find_min:
                # logging.info(len(c))
                # d.save_doc(path=base_path + f"out_docs_{method}_r{gpt2_ratio}_w{gpt2_window}/")
                if not find_min and not use_saved:
                    logging.info(f"\nSaving doc {i}")
                    d.save_doc(path=base_path + f"out_docs_{method}/", bin=with_bin)

                if method != 'lda' and not with_test:
                    topic_scores["num_segments"].append(len(gold_doc.spans["segments"]))
                    for m in ["gestalt", "edit"]:
                        if use_saved:
                            topic_eval = evaluate_topics(doc=doc, gold_doc=gold_doc, method=m)
                        else:
                            topic_eval = evaluate_topics(doc=d.doc, gold_doc=gold_doc, method=m)
                        for j, t_m in enumerate(topic_methods):
                            topic_scores[m][t_m].append(topic_eval[j])
                        # edit_scores.append(topic_eval[0])
                        # edit_scores_mc.append(topic_eval[1])
                        # edit_scores_uni.append(topic_eval[2])


                # for m in methods:
                #     eval = evaluate(doc=d.doc, gold_doc=gold_doc, t=i, method=m)
                #     # eval = evaluate(doc=d.doc, gold_doc=gold_doc, return_all=True)
                #     accu_scores[m].append(eval[0][0])
                #     wd_scores[m].append(eval[1])
            # accu_scores_uni.append(eval[2][0])
            # accu_scores_gpt.append(eval[4][0])
            # wd_scores_uni.append(eval[3])
            # wd_scores_gpt.append(eval[5])

            # assignment = d.sample_topics(num=4, allow_doubles=True)[0]  # it seems it's hard to find one sometimes!!
            # d.find_srls(topic_assignment=assignment, count=2)

            # logging.info("\nFinding summaries: ")
            # d.find_summaries(topic_assignment=assignment, count=2)
            # d.find_summaries(topic_assignment=assignment, ratio=summary_ratio)
            # d.print_segments()

            # dfs.append(d.get_for_eval())
            #
            # with open('/cs/snapless/oabend/eitan.wagner/TM_clustering/temp_segments.json', "w+") as outfile:
            #     json.dump(all_segments, outfile)
            #
        # df = pd.concat(dfs)
        # df.to_csv('/cs/snapless/oabend/eitan.wagner/segmentation/scripts/for_eval1.csv')
        logging.info(f"avg_segments: {np.mean(num_segs)}")

        if method == "max" or method == "precomputed" or method == 'lda' or find_min:
            logging.info(f"\nMethod: {method}, Ratio: {gpt2_ratio}, mean scale: {mean_scale}")

            if method != 'lda' and not with_test and not use_saved:
                logging.info("Num segments:")
                logging.info(topic_scores["num_segments"])
                for m in ["gestalt", "edit"]:
                    logging.info(f"Topic score method - {m}")
                    for t_m in topic_methods:
                        logging.info(suffix)
                        logging.info(f"Topic scores - {t_m}: {topic_scores[m][t_m]}")
                        logging.info(f"Avg: {np.mean(topic_scores[m][t_m])}")
                        if m == "edit":
                            logging.info(f"Normalized Topic Scores - {t_m}: {np.array(topic_scores[m][t_m]) / np.array(topic_scores['num_segments'])}")
                            logging.info(f"Avg - {np.mean(np.array(topic_scores[m][t_m]) / np.array(topic_scores['num_segments']))}")
                        # logging.info(f"Topic edit scores - dynamic: {edit_scores}")
                        # logging.info(f"Avg: {np.mean(edit_scores)}")
                        # logging.info(f"Topic edit scores - mc: {edit_scores_mc}")
                        # logging.info(f"Avg: {np.mean(edit_scores_mc)}")
                        # logging.info(f"Topic edit scores - uniform: {edit_scores_uni}")
                        # logging.info(f"Avg: {np.mean(edit_scores_uni)}")

            from evaluation import with_segeval
            if use_saved:
                with_segeval(ratio=gpt2_ratio, r=r, method=method, avg=350, use_close=use_close, suffix=suffix, bin=with_bin)
            else:
                # with_segeval(ratio=gpt2_ratio, r=r, method=method, avg=d.model.prior_length, use_close=use_close, suffix=suffix)
                with_segeval(ratio=gpt2_ratio, r=r, method=method, avg=d.model.prior_length, use_close=use_close,
                             suffix=suffix, annotators=None, bin=with_bin)
            # for m in methods:
            #     logging.info(m + " accuracy scores and average: ")
            #     logging.info(accu_scores[m])
            #     logging.info(np.mean(accu_scores[m]))
            #     logging.info(m + " windowdiff and average: ")
            #     logging.info(wd_scores[m])
            #     logging.info(np.mean(wd_scores[m]))

