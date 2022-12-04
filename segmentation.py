
import argparse
import logging
import json
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from gpt2 import GPT2Scorer
import numpy as np
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
import joblib

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    logging.info("Running on the GPU")
else:
    dev = torch.device("cpu")
    logging.info("Running on the CPU")


# ****************** algorithm **********

def dynamic_segmentation(doc, diffs, num_segments, window, alpha=0.2):
    """
    Dynamic algorithm to segment the doc by the diffs with a weighted length penalty
    :param doc:
    :param diffs:
    :param num_segments:
    :param window:
    :param alpha:
    :return:
    """
    sents = list(doc.spans["sents"])
    n = len(sents)
    k = num_segments
    diffs = [-np.inf] * window + [d[0] for d in diffs] + [-np.inf] * window

    # USING LINEAR PENALTY"
    L = int(n / k)

    prevs = np.zeros((n, k - 1), dtype=int)  #
    costs = np.zeros((n, k - 1))  #
    costs[0, 1:] = np.inf
    for _n in range(1, n):
        for _k in range(1, k - 1):
            arr = costs[:_n, _k - 1] + alpha * abs((_n - np.arange(_n)) - L) / L
            m = np.argmin(arr)
            costs[_n, _k] = arr[m] - (1-alpha) * diffs[_n]
            prevs[_n, _k] = int(m)

    arr = costs[:n, k - 2] + alpha * abs(n - np.arange(n) - L) / L
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


# ************** Segmentors ****************

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
        sents_arr = np.arange(len(sents))
        segments = np.array_split(sents_arr, num_segments)
        doc.spans["segments"] = [doc[sents[seg[0]].start:sents[seg[-1]].end] for seg in segments]
        return doc


class NSPSegmentor:
    """
    A segmentor based on Next Sentence Prediction scores
    """
    def __init__(self, cache_dir='/cs/snapless/oabend/eitan.wagner/cache/'):
        """
        :param base_path:
        :param cache_dir:
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        logging.info("Using Bert")

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
        if doc.spans.get("sents", None) is None:
            doc.spans["sents"] = [s for s in doc.sents]
        sents = list(doc.spans["sents"])

        # calculate all NSP scores
        self.model.to(dev)
        diffs = []
        with torch.no_grad():
            for j, s in enumerate(sents):
                if j < window or j + window > len(sents):
                    continue

                score = self._NSP_score(doc[sents[j-window].start:sents[j].start].text,
                                        doc[sents[j].start:sents[j+window-1].end].text)
                diffs.append((-score, j))
                # the score is high probability, so the cost of putting a boundary here is high. But in the algorithm we take the minus

        if not dynamic:
            diffs.sort(reverse=True)  # This is only correct with the minus
            last_js = sorted([d[1] for d in diffs[:num_segments-1]])

            doc.spans["segments"] = [doc[:sents[last_js[0]-1].end]]
            for i, j in enumerate(last_js[:-1]):
                doc.spans["segments"].append(doc[sents[j].start:sents[last_js[i+1]-1].end:])
            doc.spans["segments"].append(doc[sents[last_js[-1]].start:])
        else:
            dynamic_segmentation(doc, diffs, num_segments, window, alpha)
        return doc


class Gpt2Segmentor:
    """
    Segmentor based on gpt2 scores. Possibly with classification and length consideration
    """
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/', with_topics=False,
                 model_name='distilroberta', run_id=None, large=False, xlarge=False, from_path=None, use_len=False,
                 avg_lens=None, class_encoder=None):
        """

        :param base_path:
        :param with_topics:
        :param model_name:
        :param run_id:
        :param large:
        :param xlarge:
        :param use_nb:
        :param from_path:
        :param use_len:
        :param avg_lens:
        :param class_encoder:
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes('ner')
        self.scorer = GPT2Scorer(large=large, xlarge=xlarge, from_path=from_path)
        self.with_topics = with_topics
        self.run_id = run_id
        self.use_len = use_len

        if with_topics:
            from transformer_classification import TransformerClassifier
            self.avg_lens = avg_lens
            self.class_encoder = class_encoder
            self.model = TransformerClassifier(base_path=base_path, model_name=model_name, full_lm_scores=False)
            logging.info("Loaded classifier")

    def _make_segments(self, doc, diffs, num_segments, window=3):
        """
        Put the segmentation into the doc
        :param doc:
        :param diffs:
        :param num_segments:
        :param window:
        :return:
        """
        sents = list(doc.spans["sents"])
        diffs.sort(reverse=True)  # reversed since we use the loss
        last_js = sorted([d[1] for d in diffs[:num_segments-1]])

        doc.spans[f"segments{num_segments}"] = [doc[:sents[last_js[0]-1].end]]
        for i, j in enumerate(last_js[:-1]):
            doc.spans[f"segments{num_segments}"].append(doc[sents[j].start:sents[last_js[i+1]-1].end:])
        doc.spans[f"segments{num_segments}"].append(doc[sents[last_js[-1]].start:])

    def _get_classification_costs(self, sents, t=None):
        """
        Get classification losses for each relevant span
        :param sents:
        :param t:
        :return:
        """
        logging.info("Calculating classification scores")
        costs = np.full((len(sents), len(sents)+1, len(self.class_encoder.classes_)), np.inf)
        for i, s1 in enumerate(sents):
            for j, s2 in enumerate(sents[i:]):
                span = s1.doc[s1.start: s2.end]
                bins = 10
                loc = 0.5 * (span.start + span.end) / len(span.doc)
                # ignores spans that are too long or too short
                if len(span) > 1500 or len(span) < 15:
                    continue
                else:
                    costs[i, i + j, :] = - self.model.predict_raw(span.text + " [SEP] " + str(int(bins * loc)))
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
        classes = self.class_encoder.classes_
        c = len(classes)
        sents = list(doc.spans["sents"])
        k = num_segments + 1
        diffs = [-np.inf] * window + [d[0] for d in diffs] + [-np.inf] * window  # large difference means not to divide!
        n = len(sents)
        # L = int(n / k)

        sent_lens = [0]+[len(s) for s in sents]
        _sent_lens = np.cumsum(sent_lens)
        _Ls = self.avg_lens

        prevs = np.zeros((n, c, k-1), dtype=int)
        prev_topic = np.zeros((n, c, k-1), dtype=int)
        costs = np.full((n, c, k-1), np.inf)  # costs[n, c, k] the score for having the k+1-th (starting from 1)  break *before* the n-th sentence with topic c
        costs[0, :, 0] = 0.

        lens = _sent_lens[:n, None] - _sent_lens[:n]  # len_costs[r, c] contains the distance from *before* sent r to before sent c
        print("using linear length penalties (by topic average length)")
        len_costs = np.broadcast_to(lens[..., None], lens.shape + (c,))
        len_costs = alpha * abs(len_costs - _Ls) / _Ls

        c_costs = self._get_classification_costs(sents, t=t)  # this *includes* the second sentence
        c_costs = beta * c_costs  # this *includes* the second sentence

        for _n in range(1, n):  # current considered breaking point. i.e. at _n=1 we consider a breakpoint *before* the second sentence
            _cost = len_costs[_n, :_n, :]
            for _k in range(1, min(k-1, _n+1)):  # number of breaking points until now (not included)
                for _c in range(c):  # current topic
                    arr = costs[:_n, :, _k-1] + _cost + c_costs[:_n, _n-1, _c, None]  # prev costs + cost for new segments
                    arr[:_n, _c] = np.inf  # can't have two consecutive of the same
                    m, t = np.unravel_index(np.argmin(arr), arr.shape)
                    costs[_n, _c, _k] = arr[m, t] - (1-alpha-beta) * diffs[_n]  # ??
                    prevs[_n, _c, _k] = int(m)
                    prev_topic[_n, _c, _k] = int(t)

        _cost = _sent_lens[n] - _sent_lens[:n]
        _cost = np.broadcast_to(_cost[..., None], _cost.shape + (c,))
        _cost = alpha * abs(_cost - _Ls) / _Ls

        last_cost = np.inf
        for _c in range(c):
            arr = costs[:n, :, k-2] + _cost + c_costs[:n, n-1, _c, None]  # why not -1??
            arr[:n, _c] = np.inf
            _m, _t = np.unravel_index(np.argmin(arr), arr.shape)
            if arr[_m, _t] <= last_cost:
                last_cost = arr[_m, _t]
                i, t, t2 = int(_m), int(_t), _c

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

        doc.spans["segments"] = []
        doc._.topics = topics
        for i, j in enumerate(assignment[:-1]):
            doc.spans["segments"].append(doc[sents[j].start:sents[assignment[i+1]-1].end])

    def segment_doc(self, doc, num_segments, window, alpha=0.2, beta=1e-3):
        """
        Segment a given spacy doc
        :param dynamic: whether to use the dynamic spacing method
        :param alpha: weight for the dynamic method. If 0 then almost like uniform, and if 1 then like without the dynamic
        :param window: window for gpt2-scorer
        :param doc:
        :param num_segments:
        :return: self
        """
        if isinstance(num_segments, list):
            num_segments_l = num_segments
        else:
            num_segments_l = [num_segments]
        if doc.spans.get("sents", None) is None:
            doc.spans["sents"] = [s for s in doc.sents]
        sents = list(doc.spans["sents"])

        # calculate all PMI differences
        diffs = []
        with torch.no_grad():
            for j, s in enumerate(sents):
                if j < window or j + window > len(sents):
                    continue
                # since the scores are the loss, so this is all minused. So a large diff means we want to put a boundary
                # this always represents the loss for putting a boundary *before*
                gpt2_p1 = self.scorer.sentence_score(doc[sents[j-window].start:sents[j+window-1].end].text)
                gpt2_p2 = self.scorer.sentence_score(doc[sents[j-window].start:sents[j].start].text) \
                          + self.scorer.sentence_score(doc[sents[j].start:sents[j+window-1].end].text)
                diffs.append((gpt2_p1 - gpt2_p2, j))

        for num_segments in num_segments_l:
            if not self.use_len:
                self._make_segments(doc, diffs, num_segments, window=window)
            else:
                if self.with_topics:
                    self._make_segments_dynamic_topics(doc, diffs, num_segments, window, alpha=alpha, beta=beta)
                else:
                    dynamic_segmentation(doc, diffs, num_segments, window, alpha)
        return doc


# ********** main ***************

def make_segmentor(args):
    """
    create the segmentor by properties
    :param args:
    :return:
    """
    large, xlarge = False, False
    if args.size == "large":
        large, xlarge = True, False
    elif args.size == "xlarge":
        large, xlarge = True, True
    with_topics = False if args.classifier_path is None else True

    class_encoder = None
    if args.class_encoder_path is not None:
        class_encoder = joblib.load(args.class_encoder_path)

    avg_lens = None
    if args.lens_path is not None:
        with open(args.lens_path, 'r') as infile:
            avg_lens = json.load(infile)

    segmentor = None
    if args.method == 'pmi':
        segmentor = Gpt2Segmentor(with_topics=with_topics, large=large, xlarge=xlarge, class_encoder=class_encoder,
                                  use_len=args.use_len, avg_lens=avg_lens)
    elif args.method == 'nsp':
        if args.cache_dir is not None:
            segmentor = NSPSegmentor(cache_dir=args.cache_dir)
        else:
            segmentor = NSPSegmentor()
    elif args.methods == 'uniform':
        segmentor = UniformSegmentor()

    return segmentor

def main(args):
    # open data
    nlp = spacy.load("en_core_web_sm")
    if args.from_spacy:
        doc_bin = DocBin().from_disk(args.data_path)
        texts = list(doc_bin.get_docs(nlp.vocab))
    else:
        with open(args.data_path, 'r') as infile:
            text_dict = json.load(infile)

        doc_bin = DocBin(store_user_data=True)
        ts, texts = list(text_dict.keys()), list(text_dict.values())

    # create segmentor
    segmentor = make_segmentor(args)

    # run on texts
    for i, doc in tqdm(list(enumerate(texts))[:1]):
        if not args.from_spacy:
            logging.info(f"Document: {ts[i]}")
            doc = nlp(doc)
        doc = segmentor.segment_doc(doc, num_segments=args.num_segments, window=args.window)
        doc_bin.add(doc)
    doc_bin.to_disk(args.out_path + "doc_bin")
    logging.info("Done")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from utils import parse_args
    args = parse_args()
    main(args)
