
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.tokens import Span
from spacy.tokens import Token

import nltk
from nltk.corpus import propbank
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import allennlp_models.structured_prediction.predictors.srl

import torch
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


from pathlib import Path
CACHE_ROOT = Path("/cs/snapless/oabend/eitan.wagner/segmentation/.allennlp")
CACHE_DIRECTORY = str(CACHE_ROOT / "cache")
DEPRECATED_CACHE_DIRECTORY = str(CACHE_ROOT / "datasets")

import os
CACHE_DIR = "/cs/snapless/oabend/eitan.wagner/cache/"
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
os.environ['HF_METRICS_CACHE'] = CACHE_DIR

span_extensions = ["ent_type", "srls", "arg0", "arg1", "verb", "verb_id", "arg0_id", "arg1_id"]  # for removing
token_extensions = ["ent_span", "srl_span", "arg0_span", "arg1_span", "verb_span"]
Doc.set_extension("srl2sent", default=None, force=True)
Doc.set_extension("sent2srls", default=None, force=True)

def add_extensions():
    for ext in span_extensions:
        Span.set_extension(ext, default=None, force=True)
    for ext in token_extensions:
        Token.set_extension(ext, default=None, force=True)

def remove_extensions():
    for ext in span_extensions:
        Span.remove_extension(ext)
    for ext in token_extensions:
        Token.remove_extension(ext)


class Referencer:
    def __init__(self, nlp, num_ents=50):
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
        self.predictor._model = self.predictor._model.to(dev)
        # witness_name= "Witness"
        # interviewer_name="Interviewer"
        # self.ents = [witness_name, interviewer_name, "Family - mother, father, sister, brother, aunt, uncle", "Nazis, Germans", "Concentration camp, Extermination camp", "Israel, Palastine", "United-states, America"]
        self.nlp = nlp
        self.num_ents = num_ents

    def classify_clusters(self, clusters, document, use_len=True):
        # gets list of cluster indices and returns an ent for each cluster
        # document is just text

        # for now just the index by the order!!! (and -1 for over 50)
        if use_len:
            # in this case we return a list of cluster indices by decreasing order, and -1 if no such cluster
            len_ordered = [-1] * self.num_ents
            # print("num ents:",  self.num_ents )
            with_lens = [(len(c), i) for i, c in enumerate(clusters)]
            # print("len(with_lens):",  len(with_lens) )
            # len_ordered[:len(with_lens)] = list(zip(*sorted(with_lens)))[1][:len(with_lens)]
            num_ents = min(len(with_lens), self.num_ents)
            if num_ents != 0:
                _sorted = list(zip(*sorted(with_lens)))[1][:num_ents]
                len_ordered[:num_ents] = _sorted
            # len_ordered = list(list(zip(*sorted(with_lens)))[1])
            return len_ordered

        # use number of mentions
        lens = [len(c) for c in clusters]
        largest1 = np.argmax(lens)
        lens.pop(largest1)
        c1 = clusters.pop(largest1)
        largest2 = np.argmax(lens)  # this is after popping, so we need to insert this first
        c2 = clusters.pop(largest2)
        c1, c2 = "Witness", "Interviewer"  # add more validation
        # maybe insted use the name or the 'interviewer' word?

        ent_docs = self.nlp.pipe(self.ents)
        span_clusters = [self.nlp.pipe([" ".join(document[s[0]:s[1]+1]) for s in c]) for c in clusters]  # list of lists
        # for best match
        # max_sims = [self.ents[np.argmax([d.similarity(e) for d in sc for e in ent_docs])] for sc in span_clusters]  # list
        # if only good matches
        max_ents = []
        for sc in span_clusters:
            sims = [d.similarity(e) for d in sc for e in ent_docs]
            if max(sims) > 0.5:
                max_ents.append(self.ents[np.argmax(sims)])
            else:
                max_ents.append("Other")
        max_ents.insert(largest2, c2)
        max_ents.insert(largest1, c1)
        return max_ents

    def get_cr(self, text):
        # this receives the text and not the doc object
        # we assume that the tokens are spacy ones!!!
        try:
            cr = self.predictor.predict(text)
        except RuntimeError:  # not enough memory on the gpu
            self.predictor._model = self.predictor._model.cpu()
            cr = self.predictor.predict(text)
            self.predictor._model = self.predictor._model.to(dev)
        cluster_ents = self.classify_clusters(clusters=cr['clusters'], document=cr['document'])
        return cr['clusters'][:self.num_ents], cluster_ents[:self.num_ents]

    def add_to_Doc(self, doc, clusters, max_ents):
        # TODO divide into two - add clusters with CR, and then when classifying add the ent_type
        # adds the cluster lists to the doc, and adds the category to each span
        cluster_spans = []
        # doc._.clusters = clusters
        for c, e in zip(clusters, max_ents):
            # if e in self.ents:
                for s in c:
                    span = doc[s[0]:s[1]+1]
                    cluster_spans.append(span)
                    span._.ent_type = max_ents.index(e)
                    for t in span:
                        t._.ent_span = span
        doc.spans["clusters"] = cluster_spans  # use doc.span[] instead!!!
        return


class SRLer:
    def __init__(self, nlp=None, for_features=False):
        self.predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        self.predictor._model = self.predictor._model.to(dev)
        # self.nlp = nlp

        if for_features:
            # self.verbs = (v for v in self.nlp.vocab if v.pos_ == "VERB")
            nltk.download('propbank')
            self.verbs = propbank.verbs()


    def sent_parse(self, doc, events=False):
        """
        NEW
        :param doc:
        :return:
        """
        # find srls and add to doc
        doc.spans["srls"] = []
        doc.spans["sents"] = [s for s in doc.sents]
        doc._.srl2sent = []
        doc._.sent2srls = []
        srl_count = 0
        for i, s in enumerate(doc.sents):
            srl = self.predictor.predict(s.text)
            if events:  # take only srls with events
                locs = [[i for i, t in enumerate(v['tags']) if t != 'O'] for v in srl['verbs']
                        if np.any(np.array(v['tags']) != "O") and sum([tok._.is_event for tok in s]) > 0]  # TODO: this takes all srls if one is an event
            else:
                locs = [[i for i, t in enumerate(v['tags']) if t != 'O'] for v in srl['verbs']
                    if np.any(np.array(v['tags']) != "O")]

            # print(locs)

            # TODO combine overlapping srls
            # take longest??
            if len(locs) > 1:
                loc_lens = [len(loc) for loc in locs]
                max_len = np.argmax(loc_lens)
                locs = [locs[max_len]]
            # new_locs = []
            # loc = []
            # for j, l in enumerate(locs):
            #     if len(loc) == 0:
            #         loc = list(range(l[0], l[-1]+1))
            #     else:
            #         if l[0] > loc[-1] + 1 or l[-1] + 1 < loc[0]:
            #             new_locs.append(list(loc))
            #             loc = []
            #             continue
            #         else:
            #             loc = list(range(min(loc[0], l[0]), l[-1]+1))
            #     if j == len(locs)-1 or locs[j+1][0] > l[-1] + 1:  # non adjacent o overlapping
            #         if len(loc) > 0:
            #             new_locs.append(list(loc))
            #         loc = []
            # locs = new_locs[:]

            for l in locs:
                # print(i)
                # print(l)
                srl_span = doc[s.start + l[0]: s.start + l[-1]]
                doc.spans["srls"].append(srl_span)
                doc._.srl2sent.append(i)
            doc._.sent2srls.append(list(range(srl_count, srl_count + len(locs))))
            srl_count += len(locs)


    def parse(self, text):
        # TODO: take care of long texts!!!
        srl = self.predictor.predict(text)
        locs_w_tags = [[(i, t) for i, t in enumerate(v['tags']) if t != 'O'] for v in srl['verbs']]
        # these are all of the same length
        arg0_locs = [[i for i, (_, t) in enumerate(l) if t.find("ARG0") != -1] for l in locs_w_tags]  # relative location
        arg1_locs = [[i for i, (_, t) in enumerate(l) if t.find("ARG1") != -1] for l in locs_w_tags]
        v_locs = [[i for i, (_, t) in enumerate(l) if t.find("-V") != -1] for l in locs_w_tags]
        locs = [[i for i, t in l] for l in locs_w_tags]
        return (locs, arg0_locs, arg1_locs, v_locs)

    def get_phrases(self, text):  # not used!!
        srl = self.predictor.predict(text)
        verb_phrases_w_tags = [[(srl['words'][i], t) for i, t in enumerate(v['tags']) if v != 'O'] for v in srl['verbs']]
        return verb_phrases_w_tags

    def parse_simple(self, text):  # not used!!
        # returns a list of verb phrases for this text. Does not consider the roles
        srl = self.predictor.predict(text)
        locs_w_tags = [[(i, t) for i, t in enumerate(v['tags']) if t != 'O'] for v in srl['verbs']]
        first_last = [(l[0][0], l[-1][0]) for l in locs_w_tags]
        verb_phrases = [" ".join([srl['words'][i] for i, t in enumerate(v['tags']) if v != 'O']) for v in srl['verbs']]
        return verb_phrases, first_last

    def add_to_Span(self, span, loc_tuples, to_doc=False):
        # add also to doc!!!
        if not to_doc:
            doc = span.doc
        else:
            doc = span
        locs, arg0_locs, arg1_locs, v_locs = loc_tuples
        # gets spacy span (segment) and adds the srl attribute to each span
        srls = []
        # doc._.clusters = clusters
        for l, a0, a1, v in zip(locs, arg0_locs, arg1_locs, v_locs):
            if len(l) == 0:
                continue
            srl_span = span[l[0]:l[-1]]
            srls.append(srl_span)
            for t in srl_span:
                t._.srl_span = srl_span
            if len(a0) > 0:
                arg0_span = srl_span[a0[0]:a0[-1]+1]
                srl_span._.arg0 = arg0_span
                arg0_span._.arg0_id = arg0_span._.ent_type
                for t in arg0_span:
                    t._.arg0_span = arg0_span
            if len(a1) > 0:
                arg1_span = srl_span[a1[0]:a1[-1]+1]
                srl_span._.arg1 = arg1_span
                arg1_span._.arg1_id = arg1_span._.ent_type
                for t in arg1_span:
                    t._.arg1_span = arg1_span
            if len(v) > 0:
                verb_span = srl_span[v[0]:v[-1]+1]
                srl_span._.verb = verb_span
                for t in verb_span:
                    t._.verb_span = verb_span
                    if t.pos_ == "VERB" and t.lemma_ in self.verbs:  # if more than one with pos verb then takes the last!!
                        srl_span._.verb_id = self.verbs.index(t.lemma_)
        if to_doc:
            doc.spans["srls"] = srls
        else:
            doc.spans["srls"].extend(srls)  # should not be None. Is this used????
            span._.srls = srls
        return

    def add_to_new_span(self, span):
        span._.srls = [srl for srl in span.doc.spans["srls"] if span.start <= srl.start < span.end]

    # def span_srls(self, span):
    #     span._.srls = [srl for srl in span.doc.spans["srls"] if span.start <= srl.start <= span.end]
    #     return
