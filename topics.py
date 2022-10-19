
import logging
import spacy
import pandas as pd
import json
import pickle
import numpy as np

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


class LDAScorer:
    """
    Class for segment scores by lda
    """
    def __init__(self, num_topics, base_path='/cs/snapless/oabend/eitan.wagner/segmentation/'):
        self.model = gensim.models.ldamodel.LdaModel.load(base_path + f"models/lda/lda{num_topics}")
        self.base_path = base_path
        self.topics = None

    def predict(self, span):
        """
        Calculate log probability for the span.
        :param span:
        :return: tuple with the log-probability
        """
        if len(span) < 20:  # too short
            return (-np.inf,)

        doc = self.model.id2word.doc2bow(lemmatize(span.text))
        # doc = make_corpus([span.text])[0]  # this should be a list of one doc in bow format
        # logging.info(doc)
        # logging.info(self.model.log_perplexity([doc]))
        if len(doc) > 0:
            return (self.model.log_perplexity([doc]) * len(doc[0]), )
        else:  # empty doc
            return (-np.inf,)

    def save_cache(self):
        return

    def load_cache(self, i):
        return


def lemmatize(text, word_list=None):
    """
    Lemmatize a given text (and remove stopwords etc.)
    :param text:
    :param word_list: a list of words to ignore
    :return: list of lemmatized words
    """
    if word_list is None:
        word_list = []

    doc = nlp(text)
    t_text = []
    for w in doc:
        if not w.is_stop and not w.is_punct and not w.like_num and not w.is_space and w not in word_list:
            t_text.append(w.lemma_)
    return t_text

def make_corpus(data, return_lemmatized=False):
    """
    Make a corpus for the lda model from given data
    :param data:
    :param return_lemmatized: whether to return the lemmatized list instead of the dict
    :return: corpus (in bow format) and the id2word dict.
        if return_lemmatized then return the lemmatized list (with words) instead of id2word
    """
    _data = [" ".join(_words) for text in data for _words in np.array_split(text.split(), len(text.split())//200) ]
    lemmatized = [lemmatize(text) for text in _data]  # list of tokenized texts
    id2word = corpora.Dictionary(lemmatized)
    id2word.filter_extremes(no_below=20, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in lemmatized]
    logging.info(f'Number of unique tokens: {len(id2word)}')
    logging.info(f'Number of documents: {len(corpus)}')
    if return_lemmatized:
        return corpus, lemmatized
    return corpus, id2word

def train_lda(data, path='/cs/snapless/oabend/eitan.wagner/segmentation/models/lda/', num_topics=20, save=False):
    """
    Trains the lda model
    :param data:
    :param path:
    :param num_topics:
    :param save: whether to save the model
    :return: the trained model
    """
    corpus, id2word = make_corpus(data)

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)

    if save:
        lda_model.save(path + f"lda{num_topics}")

    return lda_model


def mdl_perplixity(nums, corpus, lemmatized, path='/cs/snapless/oabend/eitan.wagner/segmentation/models/lda/',
                   coherence=True):
    """
    Calculate the model perplexity and coherence score. The models are already trained and saved.
    :param coherence: whether to measure also coherence
    :param nums: list of topic numbers to test
    :param corpus:
    :param lemmatized:
    :param path: path of the model
    :return:
    """
    for num in nums:
        mdl = gensim.models.ldamodel.LdaModel.load(path + f"lda{num}")
        logging.info(f'\nPerplexity ({num}): {mdl.log_perplexity(corpus)}')  # a measure of how good the model is. lower is better.

        # Compute Coherence Score. higher is better
        if coherence:
            coherence_model_lda = CoherenceModel(model=mdl, texts=lemmatized, dictionary=mdl.id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            logging.info(f'\nCoherence Score ({num}): {coherence_lda}')


if __name__ == '__main__':

    data_path = '/cs/snapless/oabend/eitan.wagner/segmentation/data/'
    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })

    with open(data_path + 'sf_raw_text.json', 'r') as infile:
        data = json.load(infile)
    with open(data_path + 'sf_unused.json', 'r') as infile:
        unused = json.load(infile)
    texts = [text for t, text in data.items() if int(t) not in unused]

    # nums = [30]
    # for n in nums:
    #     logging.info(f"n: {n}")
    #     train_lda(texts, num_topics=n, save=True)

    nums = [30]
    corpus, lemmatized = make_corpus(texts)
    logging.info("Made corpus")
    mdl_perplixity(nums=nums, corpus=corpus, lemmatized=lemmatized, coherence=False)

