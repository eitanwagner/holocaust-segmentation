
from scipy import stats
import statsmodels.api as sm
import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
Doc.set_extension("topics", default=None, force=True)
import joblib

class LengthEstimators:
    def __init__(self, k):
        self.k = k
        self.mus = np.zeros(k)
        self._res_1 = np.zeros(k)
        self.ps = np.zeros(k)
        self.ns = np.zeros(k)

    def fit(self, lens, i):
        X = np.ones_like(lens)
        res = sm.NegativeBinomial(lens, X).fit(start_params=[1,1])
        self.mus[i] = np.exp(res.params[0])
        self._res_1[i] = res.params[1]
        self.ps[i] = 1/(1+np.exp(res.params[0])*res.params[1])
        self.ns[i] = self.mus[i] * self.ps[i]/(1-self.ps[i])
        return self

    def predict(self, len, i):
        # returns log
        return stats.nbinom.logpmf(len, self.ns[i], self.ps[i])
        # return stats.nbinom(self.ns[i], self.ps[i]).logpmf(len)

    def smooth(self, self_weight=0.9):
        # TODO: is this right??
        self_weight = np.array([self_weight if self.mus[i] != 0 else 0 for i in range(self.k)])
        avg_mu = np.mean(self.mus)
        avg_res1 = np.mean(self._res_1)
        self.mus = self_weight * self.mus + (1-self_weight) * avg_mu
        self._res_1 = self_weight * self._res_1 + (1-self_weight) * avg_res1
        self.ps = 1/(1+self.mus*self._res_1)
        self.ns = self.mus * self.ps/(1-self.ps)
        return self


def train(num_bins=10, segment_count=False, by_topic=False, smooth=None, num_docs=5):
    path="/cs/snapless/oabend/eitan.wagner/segmentation/"
    r = ["sf_43019", "sf_38929", "sf_32788", "sf_38936", "sf_20505" , "sf_23579", "sf_48155", "sf_35869", "sf_30751", "sf_30753", "sf_45091", "sf_25639", "sf_46120", "sf_32809",
         "sf_34857", "sf_46122", "sf_30765", "sf_24622", "sf_21550", "sf_26672"]
    evals = r[:num_docs]
    docs = []
    for e in evals:
        docs.append(Doc(Vocab()).from_disk(path + 'data/gold_docs/doc_' + e + "_Y"))

    if segment_count:
        num_bins = 1
        segment_counts = [len(d.spans["segments"]) for d in docs]
    elif by_topic:
        # here the bins are the topics
        encoder_path = path + '/models/distilroberta/'
        encoder = joblib.load(encoder_path + "label_encoder.pkl")
        num_bins = len(encoder.classes_)
        bin_lens = [[] for _ in range(num_bins)]  # lengths by topic number
        for d in docs:
            topics = encoder.transform(d._.topics)
            for t, s in zip(topics, d.spans["segments"]):
                bin_lens[t].append(len(s))
    else:
        bin_lens = [[] for _ in range(num_bins)]
        for d in docs:
            for i, s in enumerate(d.spans["segments"]):
                bin_lens[int(num_bins*i/len(d.spans["segments"]))].append(len(s))

    estimators = LengthEstimators(k=num_bins)
    if segment_count:
        estimators.fit(segment_counts, i=0)
    else:
        for i in range(num_bins):
            if len(bin_lens[i]) > 1:  # at least two examples
                estimators.fit(bin_lens[i], i=i)
        if smooth is not None:
            estimators.smooth(self_weight=smooth)

    out_path = path + '/models/distilroberta'
    if segment_count:
        joblib.dump(estimators, out_path + '/segments_estimator.pkl')
    elif by_topic:
        joblib.dump(estimators, out_path + f'/t_length_estimator{num_bins}.pkl')
    else:
        joblib.dump(estimators, out_path + f'/length_estimator{num_bins}.pkl')

if __name__ == "__main__":
    train(num_bins=10, by_topic=False, smooth=0.9, num_docs=5)