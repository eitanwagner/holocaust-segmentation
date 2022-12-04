
import difflib
import logging
import spacy
from spacy.tokens import Doc
from spacy.tokens import DocBin
Doc.set_extension("topics", default=None, force=True)
import numpy as np
import segeval
import json
import joblib
from sklearn.preprocessing import LabelEncoder

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


# ************ topic evaluation functions *********************

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


def get_per_sent(doc):
    """
    Get list of sentence segment-end labels.
    :param doc:  a spaCy doc
    :return: list with 1 for last in segment and 0 o.w., for each sentence
    """
    # converts a spacy doc with segments into a list of 0s (for no boundary after sent) and 1s (for last in segment).
    ends = [segment.end for segment in doc.spans["segments"]]
    if doc.spans.get("sents", None) is None:
        return [1 if s.end in ends else 0 for s in doc.sents]
    return [1 if s.end in ends else 0 for s in doc.spans["sents"]]


def topics_score(pred_topics, gold_topics, method="gestalt", cor_matrix_path=None):
    """
    Calculate cost for the topic list based on the edit distance.
    :param pred_doc: spacy doc after the model's segmentation
    :param gold_doc: the reference doc
    :param cor_matrix_path
    :return: edit distance. lower is better
    """
    if method == "edit":
        cor_matrix = None
        if cor_matrix_path is not None:
            cor_matrix = np.load(cor_matrix_path)
        return edit_distance(pred_topics, gold_topics, transpositions=True, cor_matrix=cor_matrix)
    return gestalt_diff(pred_topics, gold_topics)


def evaluate_topics(doc, gold_doc, with_classifier=True, method="gestalt", encoder=None, path=None, name=None):
    """
    Evaluate using edit distance, for the given doc and for a random (markov or classifier) assignment, both compared to the gold_doc
    :param doc:
    :param gold_doc:
    :return:
    """
    if with_classifier:
        ta = TopicAssigner(from_classifier=True, encoder=encoder, path=path, name=name)
        # logging.info(f"Classifier topic scores:")
        c_score = topics_score(ta.create_dynamic(doc), encoder.transform(gold_doc._.topics), method=method)
    else:
        ta = TopicAssigner(markov_chain=True, encoder=encoder)
        # logging.info(f"Markov Chain topic scores:")
        c_score = topics_score(ta.create(doc), encoder.transform(gold_doc._.topics), method=method)
    logging.info(c_score)

    ta2 = TopicAssigner(markov_chain=False, encoder=encoder)
    # logging.info(f"Uniform topic scores:")
    uni_score = topics_score(ta2.create(doc, avoid_doubles=True), encoder.transform(gold_doc._.topics), method=method)
    logging.info(uni_score)
    return c_score, uni_score


# *************** segeval evaluation ******************


def _make_length_dict(doc, from_dict=True):
    """
    Makes lengths for one testimony.
    :param doc:
    :return:
    """
    def add_to_dict(ddict, doc, method, use_encoder=False):
        ends = np.array(get_per_sent(doc))
        # idxs = ends != 0
        idxs = [-1] + np.nonzero(ends)[0].tolist()  # the ends are included so the "previous end" is -1
        diffs = []
        for i in range(1, len(idxs)):
            diffs.append(idxs[i] - idxs[i-1])
        ddict[method] = diffs

    ddict = {}
    if from_dict:
        for m, d in doc.items():
            add_to_dict(ddict, d, method=m, use_encoder=(True if m[:4] == "gold" else False))
        return ddict

def make_len_dict(path="/cs/snapless/oabend/eitan.wagner/segmentation/", from_dict=None, r=None):
    """
    Makes dictionary of segment lengths for each method, for using segeval.
    Saves the dict in path.
    :param path:
    :return:
    """
    ddict = {r: _make_length_dict(doc=from_dict, from_dict=True)}

    # reverse the order of testimonies and methods
    _dict = {k: {} for k in list(ddict.values())[0].keys()}  # for each method
    for i, d in ddict.items():
        for method, l in d.items():
            _dict[method][str(i)] = l

    full_dict = {"items": _dict, "segmentation_type": "linear"}
    with open(path + f'segmentation_lens.json', 'w+') as outfile:
        json.dump(full_dict, outfile)


def seg_eval(path="/cs/snapless/oabend/eitan.wagner/segmentation/", gold2=False, r=None, suffix="", bin=False):
    """
    Evaluate with segeval package.
    :param path:
    :return:
    """
    if r is None:
        r = range(101, 111)

    def f_measure(seg1, gold):
        conf_matrix = segeval.boundary_confusion_matrix(seg1, gold)
        return segeval.fmeasure(conf_matrix)
    def precision(seg1, gold):
        conf_matrix = segeval.boundary_confusion_matrix(seg1, gold)
        return segeval.precision(conf_matrix)
    if bin:
        dataset = segeval.input_linear_mass_json(path + f'segmentation_lens{suffix}.json')
    else:
        dataset = segeval.input_linear_mass_json(path + f'segmentation_lens{suffix}.json')

    sims = {}
    golds = ["gold"]
    # methods = ["gpt2", "gpt2-topics", "uniform", "nsp"]
    methods = ["segmented"]

    for g in golds:
        g_sims = sims
        for func in [segeval.boundary_similarity, segeval.segmentation_similarity, segeval.window_diff,
                     segeval.pk, f_measure, precision]:
            avg = {}
            for m in methods:
                # print(m)
                _sims = []
                # for i in range(101, 111):
                for i in r:
                    seg1 = dataset[m][str(i)]
                    gold = dataset["gold"][str(i)]
                    _sims.append(func(seg1, gold))
                avg[m] = float(np.mean(_sims))
            g_sims[func.__name__] = avg
    logging.info(g_sims)
    return g_sims

# ************************ baselines for topic assignment **************************

class TopicAssigner:
    """
    Assigns a random topic assignment (by markov chain or frequency)
    """
    def __init__(self, markov_chain=False, frequencies=None, path='/cs/snapless/oabend/eitan.wagner/segmentation/models',
                 name="transitions/mc.json", from_classifier=False, encoder=None):
        """
        :param markov_chain: whether the assigner is based on MC probabilities. Otherwise by given probabilities
        :param frequencies: probilities for the topic. if None then uses uniform
        """
        self.encoder = encoder
        if frequencies is None:
            frequencies = np.ones(len(encoder.classes_)) / len(encoder.classes_)
        self.frequencies = frequencies
        self.markov_chain = markov_chain
        if markov_chain:
            if name[-4:] == "json":
                self.mc = MC(base_path=path, name=name)
        if from_classifier:
            from transformer_classification import TransformerClassifier
            self.model = TransformerClassifier(path=path, model_name=name, mc=None, full_lm_scores=False)

    def create(self, doc, avoid_doubles=False):
        """
        Creates an assignment for a segmented doc
        :param avoid_doubles:
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
                    prev = int(np.random.choice(len(self.encoder.classes_), size=1, p=probs))
                    topics.append(prev)
                return topics
            return list(np.random.choice(len(self.encoder.classes_), size=k, p=self.frequencies))

    def create_dynamic(self, doc):
        k = len(doc.spans["segments"])
        c = len(self.encoder.classes_)
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
            prev_c = prevs[_k, prev_c]
            assigmnent.insert(0, prev_c)
        return assigmnent


def evaluate(args, segmented_docs, eval_docs, num_segments=20):
    """

    :param args:
    :param segmented_docs:
    :param eval_docs:
    :return:
    """
    encoder = None
    if args.class_encoder_path is not None:
        encoder = joblib.load(args.class_encoder_path)

    s_es = []  # segevals
    lengths = []
    b_s0 = []
    sm0 = [None] * len(segmented_docs)
    sm_gold = [None] * len(segmented_docs)  # gold segments (and dynamic topics)
    sm_uni_t = [None] * len(segmented_docs)  # uniform topics (not uniform segmentation)
    ed0 = [None] * len(segmented_docs)
    ed_gold = [None] * len(segmented_docs)
    ed_uni_t = [None] * len(segmented_docs)

    for i, (doc, gold_doc) in enumerate(zip(segmented_docs, eval_docs)):
        lengths.append(len(gold_doc.spans['segments']))

        from_dict = {"gold": gold_doc}
        # evaluate gold_doc with dynamic topics
        sm_gold[i], _ = evaluate_topics(doc, gold_doc, method="gestalt", encoder=encoder, path=args.classifier_path,
                                        name=args.classifier_name)
        ed_gold[i], _ = evaluate_topics(doc, gold_doc, method="edit", encoder=encoder, path=args.classifier_path,
                                        name=args.classifier_name)

        from_dict["segmented"] = doc
        sm0[i], sm_uni_t[i] = evaluate_topics(doc, gold_doc, method="gestalt", encoder=encoder, path=args.classifier_path,
                                              name=args.classifier_name)
        ed0[i], ed_uni_t[i] = evaluate_topics(doc, gold_doc, method="edit", encoder=encoder, path=args.classifier_path,
                                              name=args.classifier_name)

        make_len_dict(from_dict=from_dict, suffix="_topic", r=str(i))
        s_e = seg_eval(r=[i], suffix="_topic")

        s_es.append(s_e)
        # print(s_e)
        b_s0.append(s_e['boundary_similarity']['segmented'])

    print(f"\n\"Lengths\": {lengths}")
    print(f",\"Boundary scores\": {b_s0}")
    print(f",\"Sequence Matching (gpt)\": {sm0}")
    print(f",\"Sequence Matching (gold segments, dynamic topics)\": {sm_gold}")
    print(f",\"Sequence Matching (gpt2 segments, uniform topics)\": {sm_uni_t}")
    print(f",\"Edit distance\": {ed0}")
    print(f",\"Edit distance (gold segments, dynamic topics)\": {ed_gold}")
    print(f",\"Edit distance (gpt2 segments, uniform topics)\": {ed_uni_t}")


# ************ main **********

def main(args):
    nlp = spacy.load("en_core_web_sm")

    # open data
    doc_bin = DocBin().from_disk(args.data_path)
    segmented_docs = list(doc_bin.get_docs(nlp.vocab))

    # open evaluation docs
    doc_bin = DocBin().from_disk(args.gold_path)
    eval_docs = list(doc_bin.get_docs(nlp.vocab))

    evaluate(args=args, segmented_docs=segmented_docs, eval_docs=eval_docs, num_segments=args.num_segments)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from utils import parse_args
    args = parse_args()
    main(args)