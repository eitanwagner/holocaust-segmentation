import numpy as np
from pomegranate import MarkovChain
from pomegranate import DiscreteDistribution
from pomegranate import ConditionalProbabilityTable
import json
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import logging
from scipy.special import logsumexp

class MC:
    """
    Class for a Markov Chain over the topics
    """
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/', name="models/transitions/mc.json"):
        self.base_path = base_path
        if name != '':
            with open(base_path + name, 'r') as f:
                self.mc = MarkovChain.from_json(f.read())
        else:
            self.mc = None

        encoder_path = base_path + '/models/deberta-large/'
        # encoder_path = base_path + '/models/xlnet-large-cased/'
        self.encoder = joblib.load(encoder_path + "label_encoder.pkl")  # this does not have a "begin" and "end" label

    def fit(self, chains, out_name=None, inertia=0.5):
        """
        Fit from a list of chains
        :param chains: list of topic-lists
        :param out_name:
        :return: self
        """
        # create uniform and then update
        dict = {i: 1 / len(self.encoder.classes_) for i in range(len(self.encoder.classes_))}
        d1 = DiscreteDistribution(dict)
        t_list = [[i, j, 1 / (len(self.encoder.classes_) - 1)] if i != j else [i, j, 0]
                  for i in range(len(self.encoder.classes_)) for j in range(len(self.encoder.classes_))]
        d2 = ConditionalProbabilityTable(t_list, [d1])
        self.mc = MarkovChain([d1, d2])

        self.mc.fit(chains, inertia=inertia)
        # self.mc = MarkovChain.from_samples(chains)
        if out_name:
            with open(base_path + out_name, 'w+') as f:
                f.write(self.mc.to_json())
        return self

    def predict(self, topic, prev_topic, encoded=True):
        """
        Predict the probability for a topic given a previous topic
        :param topic:
        :param prev_topic:
        :param encoded:
        :return: the probability
        """
        if not encoded:
            topic, prev_topic = self.encoder.transform([topic, prev_topic])

        return self.mc.distributions[1].parameters[0][self.mc.distributions[1].keymap[(prev_topic, topic)]][2]

    def predict_vector(self, prev_topic, encoded=True):
        """
        Predict the probability vector for a given previous topic
        :param prev_topic: previous topic. if -1 then this is the first so give the initial probabilities
        :param encoded:
        :return: list of probabilities
        """
        if not encoded:
            prev_topic = self.encoder.transform(prev_topic)

        if prev_topic == -1:
            return list(self.mc.distributions[0].parameters[0].values())

        first = self.mc.distributions[1].keymap[(prev_topic, 0)]
        last = first + len(self.encoder.classes_)  # do not include the last one here

        return [p for _, _, p in self.mc.distributions[1].parameters[0][first:last]]

    def sample(self, k):
        """
        Sample a Markov chain of length k
        :param k:
        :return:
        """
        return self.mc.sample(k)


def make_chains(base_path, save=False):
    """
    Makes topic-chains from the SF data
    :param base_path:
    :param save: whether to save to file
    :return: list of topic-lists
    """
    # encoder_path = base_path + '/models/xlnet-large-cased/'
    encoder_path = base_path + '/models/deberta-large2/'
    encoder = joblib.load(encoder_path + "label_encoder.pkl")  # this does not have a "begin" and "end" label

    docs_path = base_path + '/data/docs/'
    with open(docs_path + "data6.json", 'r') as infile:
        data = json.load(infile)

    # topic_lists = [encoder.transform(np.ravel([_d[2] for _d in d])).tolist() for _, d in data.items()]
    _topic_lists = [(t, encoder.transform(np.ravel([_d[2] for _d in d])).tolist()) for t, d in data.items()]
    ts, topic_lists = list(zip(*_topic_lists))
    for l in topic_lists:
        for i in range(len(l)-1, 0, -1):
            if l[i] == l[i-1]:
                l.pop(i)
    dict = {t: l for t, l in zip(ts, topic_lists)}

    if save:
        print(topic_lists)
        with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/chain6.json', "w+") as outfile:
            json.dump(list(topic_lists), outfile)
        with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/chain_dict6.json', "w+") as outfile:
            json.dump(dict, outfile)

        # save the list of testimonies (for the classifier)
        print(ts)
        with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/ts6.json', "w+") as outfile:
            json.dump(list(ts), outfile)
        print(encoder.classes_)
        with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/topics6.json', "w+") as outfile:
            json.dump(encoder.classes_.tolist(), outfile)

    return topic_lists

def get_sf_testimony_texts(l, data_path='/cs/snapless/oabend/eitan.wagner/segmentation/data/'):
    """
    Obatin a list of sentences for testimony i (in the SF corpus)
    :param i:
    :param data_path:
    :return:
    """
    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        texts = json.load(infile)
    return [texts[i] for i in l]


def get_topic_list(base_path):
    return [t for c in make_chains(base_path=base_path) for t in c]


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'



