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


class MultinomialMixture:
    def __init__(self, classifier, vectorizer):
        """
        :param classifier: the classifier for obtaining the probability weights
        """

        self.ps = None  # a table of probabilities (as column vectors)
        # self.priors = None  # prior probability for each vector
        self.n = None  # dimension
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.weights = None

    def from_dmm(self, dmm):
        """
        Obtain probabilities
        :param dmm:
        :return:
        """
        self.ps = dmm.psi

    def set_weights(self, text):
        """
        Get the weights from the classifier, given a text
        :return:
        """
        x = self.vectorizer.fit_transform([text]).astype(dtype=float).todense()
        self.classifier.predict_proba(x)[0]  #

    def prod_prob(self, weights=None):
        """

        :param weights:
        :return:
        """
        if weights is None:
            weights = self.weights
        return self.psi @ np.array(weights)


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


class LocationClassifier:
    def __init__(self, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/', bins=20):
        # self.base_path = base_path
        self.encoder_path = base_path + 'models/deberta-large2/'
        self.encoder = joblib.load(self.encoder_path + "label_encoder.pkl")  # this does not have a "begin" and "end" label
        self.bins = bins
        self.probs = np.ones((bins, len(self.encoder.classes_)))  # each row will be a probability for this bin

    def fit(self, chains, pseudo_counts=1):
        def to_bin(i, j):
            return int(self.bins * i / j)

        self.probs = self.probs * pseudo_counts
        # in_bin = np.zeros(self.bins)
        for c in chains:
            for i, _c in enumerate(c):
                self.probs[to_bin(i, len(c)), _c] += 1

        # normalize
        self.probs = self.probs / np.sum(self.probs, axis=1, keepdims=True)
        return self

    def predict(self, loc, encoded=True):
        # predict vector
        # loc between 0 and 1 (not including 1!)
        return self.probs[int(loc * self.bins), :]

    def save_probs(self):
        np.save(self.encoder_path + "bin_probs.npy", self.probs)


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

def make_long_train(base_path):
    encoder_path = base_path + '/models/deberta-large2/'
    encoder = joblib.load(encoder_path + "label_encoder.pkl")  # this does not have a "begin" and "end" label

    data_path = base_path + '/data/'
    with open(data_path + "chain_dict6.json", 'r') as infile:
        chain_data = json.load(infile)

    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        texts = json.load(infile)

    data = [(texts[t], " </s> ".join(encoder.inverse_transform(chain))) for t, chain in chain_data.items()]

    with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/long_train2.json', "w+") as outfile:
        json.dump(data, outfile)
    return data

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

def fit_dmm(chains, k):
    from GPyM_TM import GSDMM
    # corpus = [" ".join([str(_c) for _c in c]) for c in chains]
    corpus = [[str(_c) for _c in c] for c in chains]
    dmm = GSDMM.DMM(corpus=corpus, nTopics=k)
    # dmm = GSDMM.DMM(corpus=chains, nTopics=k)
    dmm.topicAssigmentInitialise()
    dmm.inference()
    dmm.worddist()
    finalAssignments = dmm.writeTopicAssignments()
    return dmm.topicAssignments


def train_classifier(base_path, out_path=None, k=5, topic_assignments=None):
    """
    train a classifier from testimony to MC in the mixture model
    :return:
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from scipy.sparse import csr_matrix
    from sklearn.metrics import accuracy_score

    # with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/chains5.json', "r") as infile:
    #     topic_lists = json.load(infile)
    tfidf = TfidfVectorizer()
    # encoder = LabelEncoder()

    with open(base_path + '/data/ts5.json', "r") as infile:
        ts = json.load(infile)

    texts = get_sf_testimony_texts(ts)
    if topic_assignments is None:
        # load id2set from the MCC classifier
        mcc = joblib.load(base_path + f'models/transitions/mcc{k}_iner1_iter15_data5.pkl')
        y = mcc.id2set
    else:
        y = topic_assignments

    X = tfidf.fit_transform(texts).astype(dtype=float)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14)
    X_train = csr_matrix(np.nan_to_num(X_train.todense()), dtype=float)
    X_test = csr_matrix(np.nan_to_num(X_test.todense()), dtype=float)
    # train
    # clf = DecisionTreeClassifier(max_depth=5)
    clf = SVC(probability=True)
    # logging.info("Training SVM")
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    # logging.info(X_train[:10])
    logging.info(y_train[:10])
    y_pred = clf.predict(X_train)
    logging.info(f"Accuracy (train): {accuracy_score(y_train, y_pred)}")
    y_pred = clf.predict(X_test)
    logging.info(f"Accuracy (test): {accuracy_score(y_test, y_pred)}")
    # save
    if out_path is not None:
        joblib.dump(clf, out_path + f'svm5_{k}.pkl')
    return clf


def get_topic_list(base_path):
    return [t for c in make_chains(base_path=base_path) for t in c]

# not used
def make_transition_matrix(base_path, out_name="models/transitions/mc.json"):
    # should we apply smoothing?

    chains = make_chains(base_path)
    mc = MarkovChain.from_samples(chains)

    with open(base_path + out_name, 'w+') as f:
        f.write(mc.to_json())

    return mc


class MCClusters:
    """
    Class for clustering as a mixture of Markov Chains
    """
    def __init__(self, k=10, inertia=0.5):
        self.inertia = inertia
        self.k = k
        self.mcs = [None] * k
        self.sets = [None] * k  # each set is a list of chain indices
        self.chains = None
        self.id2set = []
        self.ll = None

        self.base_path='/cs/snapless/oabend/eitan.wagner/segmentation/'
        # encoder_path = self.base_path + '/models/xlnet-large-cased/'
        encoder_path = self.base_path + '/models/deberta-large/'
        self.encoder = joblib.load(encoder_path + "label_encoder.pkl")  # this does not have a "begin" and "end" label

    def make_testimony2chain(self, testimony_list):
        """
        Makes a dict from testimony id to the best MC
        :return:
        """
        return dict(zip(testimony_list, self.id2set))

    def _chain2topics(self, chain):
        return self.encoder.inverse_transform(chain)

    def save_id2set(self, name=''):
        """
        Save sets of chains as a dict by testimony
        :param name:
        :return:
        """
        with open(f'/cs/snapless/oabend/eitan.wagner/segmentation/data/'+name, "w+") as outfile:
            json.dump(self.id2set, outfile)

    def save_sets(self, name=''):
        """
        Save sets of chains as a dict by testimony
        :param name:
        :return:
        """
        sets = {i: [self._chain2topics(self.chains[j]).tolist() for j in js] for i, js in enumerate(self.sets)}
        if name == '':
            name = f'topic_chains_{self.k}.json'
        with open(f'/cs/snapless/oabend/eitan.wagner/segmentation/data/'+name, "w+") as outfile:
            json.dump(sets, outfile)

    def load(self):
        """
        Load data from file
        :return:
        """
        name = f'topic_chains_{self.k}.json'
        with open(f'/cs/snapless/oabend/eitan.wagner/segmentation/data/'+name, "r") as infile:
            dict = json.load(infile)
            self.sets = [[self.encoder.transform(l) for l in s] for s in dict.values()]
        with open('/cs/snapless/oabend/eitan.wagner/segmentation/data/chains.json', "r") as infile:
            self.chains = json.load(infile)
        self._fit_on_sets(self.sets, weighted=True)
        return self

    def _init_sets(self, X):
        """
        initialize the clustering with k-means on the count vector
        :param X:
        :return:
        """
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(X)
        for i, l in enumerate(kmeans.labels_):
            if self.sets[int(l)] is None:
                self.sets[int(l)] = list()
            self.sets[int(l)].append(i)
            self.id2set.append(int(l))

        #return a copy
        return [list(s) for s in self.sets]

    def _fit_on_sets(self, sets, weighted=False, idxs=None):
        """
        Fit the MCs with a given set of sets
        :param sets:
        :param weighted:
        :param idxs:
        :return: the likelihood for this partition (after fitting)
        """
        if idxs is not None:
            for i in idxs:
                if len(sets[i]) >= 0:
                    self.mcs[i] = MC().fit([self.chains[s] for s in sets[i]], inertia=self.inertia)
            sets = [sets[i] for i in idxs]
            mcs = [self.mcs[i] for i in idxs]
            chains = [self.chains[j] for s in sets for j in s]

        else:
            for i in range(len(sets)):
                if len(sets[i]) >= 0:
                    self.mcs[i] = MC().fit([self.chains[s] for s in sets[i]], inertia=self.inertia)
            mcs = self.mcs
            chains = self.chains

        if weighted:
            # log_weights = np.log([len(s) / len(self.chains) for s in sets])
            weights = np.array([len(s) / len(self.chains) for s in sets])
            # logging.info(weights)
            # log_probs = [np.array([mc.mc.log_probability(chain) for mc in self.mcs]) for chain in self.chains]
            log_probs = [logsumexp(a=np.array([mc.mc.log_probability(chain) for mc in mcs]), b=weights) for chain in chains]
            # logging.info(log_probs)
            return sum(log_probs)
            # return sum([log_weights @ np.array([mc.mc.log_probability(chain) for mc in self.mcs]) for chain in self.chains])

        log_prob = sum([self.mcs[i].mc.log_probability(self.chains[j]) for i, s in enumerate(sets) for j in s])
        # log_prob = sum([self.mcs[i].mc.log_probability(self.chains[c]) for i in self.id2set for c in sets(i)])
        return log_prob

    def fit(self, chains, iterations=20, weighted=False):
        """
        Fit the mixture model
        :param chains:
        :param iterations:
        :param weighted:
        :return: self
        """
        self.chains = chains

        # initial clustering using k-means
        X = np.zeros((len(chains), len(self.encoder.classes_)))
        for i, chain in enumerate(chains):
            for j, l in enumerate(chain):
                X[i, l] += 1
        sets = self._init_sets(X)

        ll = self._fit_on_sets(sets, weighted=weighted)
        for t in range(iterations):  # iterations
            logging.info(f"ll: {ll}")
            logging.info(f"**************************************************** {t} ")
            # fit mcs
            # js = list(range(0, len(chains), 2))  # these are chain-index pairs
            js = list(range(len(chains)))  # these are chain-index pairs
            set_idxs = list(range(self.k))  # these are chain-index pairs
            np.random.shuffle(js)

            # predict with mcs
            # we need to check with each cluster if it's better to change!!!!
            for j in js:
                # for j1, j2 in zip(js1, js2):
                for s_i in set_idxs:
                    if self.id2set[j] == s_i:
                        continue

                    # score for the 2 sets considered, before swapping. This is NOT GOOD!!
                    # ll2sets = self._fit_on_sets(sets, idxs=[self.id2set[j], s_i], weighted=weighted)

                    sets[self.id2set[j]].remove(j)
                    sets[s_i].append(j)

                    ll2 = self._fit_on_sets(sets, weighted=weighted)
                    # ll2sets2 = self._fit_on_sets(sets, idxs=[self.id2set[j], s_i], weighted=weighted)

                    if ll2 > ll:
                        ll = ll2
                        # if ll2sets2 > ll2sets:
                        ll = ll2
                        # print(f"ll: {ll}")
                        self.id2set[j] = s_i

                    else:
                        sets[s_i].remove(j)
                        sets[self.id2set[j]].append(j)

        logging.info(f"ll: {ll}")
        self.ll = ll
        self.sets = sets
        return self

    def predict(self, chain):
        """
        predict best cluster (markov chain)
        :param chain:
        :return: index of best cluster for a new chain
        """
        log_probs = [mc.mc.log_probability(chain) if self.mcs is not None else -np.inf for mc in self.mcs]
        m = np.argmax(log_probs)
        return m

    def sample(self, k):
        """
        Sample a length k chain of topic. First choose a chain and then sample from it.
        :param k:
        :return:
        """
        weights = np.array([len(s) / len(self.chains) for s in self.sets])  # should sum up to 1
        mc = np.random.choice(self.mcs, 1, p=weights).item()
        return mc.sample(k)

    def predict_vector(self, prev_topic):
        """
        Calculate the probability from t1 to all other topics.
        We calculate the probability for each MC and then average by weight/
        :param t1:
        :return:
        """
        if self.sets is None or self.mcs is None:
            return

        weights = np.array([len(s) / len(self.chains) for s in self.sets])  # should sum up to 1
        vectors = [mc.predict_vector(prev_topic=prev_topic) for mc in self.mcs]
        return np.average(vectors, axis=0, weights=weights)
        # return np.sum(weights[:, np.newaxis] * vectors, axis=0)

def train_mcc():
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'

    # chains = [[1,2,3,4,5],[2,3,4,5,1],[4,3,4,3,2], [3,4,5,1,2], [5,4,3,2,1]]
    logging.info("Using max cluster")
    # lls = []
    train_mcc = True
    train_dmm = False
    inertias = [0.1]
    # inertias = [0.1, 0.3, 0.5, 0.7, 0.9]
    # for k in [2, 3, 5, 7, 10, 15, 20]:
    # for k in [10]:
    # for k in [3,4,7]:
    for k in [5]:
        # for k in [2]:
        logging.info(f"k: {k}")
        for iner in inertias:
            if train_mcc:
                logging.info(f"inertia: {iner}")
                mcc = MCClusters(k=k, inertia=iner).fit(chains, iterations=15)
                mcc.save_sets()
                logging.info(f"likelikhood for {k} {iner}: {mcc.ll}")
                # lls.append(mcc.ll)
                # logging.info(f"likelikhoods for {k}:")
                # logging.info(lls)
                #         if k == 5 and iner == 0.1:
                if True:
                    joblib.dump(mcc, base_path + f'models/transitions/mcc{k}_iner{str(iner)[-1]}_iter15_data5.pkl')

            # if not train_dmm:
            #     train_classifier(base_path=base_path, out_path=base_path + 'models/transitions/', k=k)
            # else:
            #     ta = fit_dmm(chains=chains, k=k)
            #     logging.info(ta[:10])
            #     train_classifier(base_path=base_path, topic_assignments=ta)
    # logging.info("Using weighted probability")
    # # for k in [2, 3, 5, 7, 10, 15, 20]:
    # lls = []
    # for k in [5]:
    #     logging.info(f"k: {k}")
    #     mcc = MCClusters(k=k).fit(chains, weighted=True, iterations=15)
    #     lls.append(mcc.ll)
    #     # mcc.save_sets(name=f'topic_chains_{k}_kmeans.json')
    # logging.info(f"likelkhoods for {k}:")
    # logging.info(lls)
    # joblib.dump(mcc, base_path + 'models/transitions/mcc5_iner5_iter15.pkl')
    # x=3
    # print(mcc)

def train_bin():
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    chains = make_chains(base_path, save=False)
    l_c = LocationClassifier(base_path=base_path, bins=20).fit(chains=chains)
    l_c.save_probs()

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'

    # train_bin()
    # make_transition_matrix(base_path=base_path)
    # # p = mc.predict(topic=4, prev_topic=5)
    # # p2 = mc.predict(topic=14, prev_topic=35)
    # # p3 = mc.predict(topic=14, prev_topic=14)
    # p4 = mc.predict(topic=25, prev_topic=20)
    # ps = mc.predict_vector(20)

    # chains = make_chains(base_path, save=True)
    make_long_train(base_path)
    # t_l = get_topic_list(base_path)
    # print("done")
    # mc = MC(name='').fit(chains, out_name="models/transitions/mc_iner1")

