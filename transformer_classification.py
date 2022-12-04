
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab

import os
CACHE_DIR = "/cs/snapless/oabend/eitan.wagner/cache/"
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
os.environ['HF_METRICS_CACHE'] = CACHE_DIR
# os.environ["WANDB_DISABLED"] = "true"

import torch
# print(torch.__version__)
# print(torch.version.cuda)
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import json
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import log_softmax
from scipy.stats import poisson
from scipy.special import logsumexp
import logging
from gpt2 import GPT2Scorer

from sklearn.utils.class_weight import compute_class_weight

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments


if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


import numpy as np
from datasets import load_metric
import joblib


class TransformerClassifier:
    """
    A classifier based on transformers.
    This class does not train the model.
    Predicts segment scores based on the classifier, Markov chain and length probabilities
    """
    def __init__(self, path='/cs/snapless/oabend/eitan.wagner/segmentation/models/', model_name='distilroberta',
                 mc=None, full_lm_scores=False):
        # does not use extra features
        self.mc = mc
        self.model = AutoModelForSequenceClassification.from_pretrained(path + model_name,
                                                                        cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.to(dev)
        self.model.eval()
        self.model_name = model_name
        # save also the tokenizer!!!
        if model_name == 'distilroberta':
            self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base',
                                                      cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")

        self.probs = None

        if full_lm_scores:
            self.lm_scorer = GPT2Scorer(window=3)
        else:
            self.lm_scorer = None
        self.vocab_len = len(self.tokenizer.get_vocab())
        self.path = path
        self.encoder = joblib.load(self.path + model_name + '/label_encoder.pkl')
        self.topics = self.encoder.classes_
        self.prior_length = None
        self.prior_scale = None
        self.nt_prior_length = None  # for NO_TOPIC
        self.nt_prior_scale = None

        self.topics_prior = None

        self.cache = {}  # classifier cache
        self.cache_id = None

    def save_cache(self):
        """
        Save classification cache to file
        :return:
        """
        np.save(self.path + self.model_name + f"/pred_cache{str(self.cache_id)}.npy", self.cache)
        if self.lm_scorer is not None:
            self.lm_scorer.save_cache()

    def load_cache(self, i):
        """
        Load classification cache from file
        :param i: testimony number
        :return:
        """
        self.cache_id = i
        try:
            self.cache = np.load(self.path + self.model_name + f"/pred_cache{str(i)}.npy", allow_pickle='TRUE').item()
        except IOError as err:
            # except:
            pass
        if self.lm_scorer is not None:
            self.lm_scorer.load_cache(i)

    def _predict(self, span, prev_topic=None, spans=None):
        """
        Calculates the probabilities for the span
        :param span:
        :param prev_topic:
        :return: tuple of the total probability vector, and the classification probability vector
        """
        if len(span) > 1500:
            # TODO think about this
            extra = len(span) - 1500
            text = span[extra//2: -extra//2].text
        else:
            text = span.text
        if text in self.cache.keys():
            out = self.cache[text]
        else:
            encodings = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
            encodings.to(dev)
            try:
            # if len(span) < 1200:
                out = self.model(encodings['input_ids']).logits.detach()[0].cpu().numpy()
                self.cache[text] = out
            except RuntimeError:
                raise
        pred_logp = np.ravel(log_softmax(out))  # logP(t|x,s)

        if self.probs is not None:
            bins = self.probs.shape[0]
            len_bins = 10
            loc = 0.5 * (span.start+span.end)/len(span.doc)

        if prev_topic is not None:
            if self.mc is None:
                if self.probs is not None:

                    pred_logp += np.log(self.probs[int(bins * loc), :])
                    # _pred_logp = pred_logp
                    # logging.info(f"probs: {np.log(self.probs[int(bins * loc), :])}")
                    # _p = pred_logp
                pred_logp[prev_topic] = -np.inf  # no need to normalize since we choose the max. we do!!!
                pred_logp = pred_logp - logsumexp(pred_logp)  # normalize?? for max it doesn't matter
                if self.probs is not None:
                    pred_logp -= np.log(self.probs[int(bins * loc), :])  # TODO
                # logging.info(f"pred_logp2: {pred_logp}")
                # pred_logp = pred_logp - _p  # /P(t|i)
            else:  # uses transition probabilities
                with np.errstate(divide='ignore'):
                    # pred_logp = pred_logp + np.log(self.mc.predict_vector(prev_topic=prev_topic))
                    t_prob = np.log(self.mc.predict_vector(prev_topic=prev_topic)) - np.log(self.topics_prior)  # P(t_i|t_{i-1})/P(t_i)
                    pred_logp = pred_logp + t_prob


        if self.lm_scorer is None:
            lm_logp = -len(span) * np.log(len(span.doc.vocab))  # logP(x|s). This shouldn't have any effect
        else:
            # lm_logp = -self.lm_scorer.sentence_score(span.text)  # this is the LM loss so we need the minus to get the probability
            lm_logp = -self.lm_scorer.sentence_score(sent=[s.text for s in spans])  # this is the LM loss so we need the minus to get the probability
            # logging.info(f"lm_logp: {lm_logp}")

        if self.probs is not None:
            factor = 1.  # larger factor means a LARGER penalty for long segments
            len_logp = (self.len_estimators.predict(len=len(span) * factor, i=int(len_bins * loc)) / (len(span.doc) / len(span))
                        + self.segments_estimator.predict(len(span.doc) * factor // len(span), i=0) / (len(span.doc) / len(span))) \
                       * np.ones(len(pred_logp))

        elif self.prior_scale is not None:
            len_logp = poisson.logpmf(len(span) // self.prior_scale, self.prior_length // self.prior_scale) * np.ones(len(pred_logp)) # logP(s). Check for a more accurate prior!!
        else:
            len_logp = 0.
        if self.nt_prior_length is not None:
            len_logp[self.encoder.transform(["NO_TOPIC"])[0]] = poisson.logpmf(len(span) // self.nt_prior_scale, self.nt_prior_length // self.nt_prior_scale)

        logp = pred_logp + lm_logp + len_logp
        return logp, pred_logp

    def predict_all(self, span, prev_topic=None):
        """
        Get probabilities for all possible topics, given a prev_topic
        :param span:
        :param prev_topic:
        :return: a list of probabilities of len num_topics
        """
        if len(span) > 1500:
            return np.full(len(self.encoder.classes_), -np.inf)
        logp, pred_logp = self._predict(span, prev_topic=prev_topic)
        return logp

    def predict_max(self, span, prev_topic=None, spans=None):
        """
        Predicts the max class for a given span
        :param span: spacy span to classify
        :param prev_topic: topic of the previous span (for transition probabilities). This is the id and not the actual topic
        :return: tuple of probability of max class, probablity vector, and max class id
        """


        logp, pred_logp = self._predict(span, prev_topic=prev_topic, spans=spans)

        max_p = int(np.argmax(logp))
        return logp[max_p], pred_logp, max_p

    def predict(self, span):
        """
        Predicts the marginal probablity for a given span
        :param span: spacy span to classify
        :param prev_topic: topic of the previous span (for transition probabilities)
        :return: tuple of marginal probability and the classification probability vector
        """
        logp, pred_logp = self._predict(span)
        return logsumexp(logp), pred_logp

    def predict_raw(self, text):
        # logging.info(f"text len: {len(text)}")
        if len(text) > 6000:
            l = (len(text) - 6000) // 2
            text = text[l:-l]
        encodings = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        encodings.to(dev)
        try:
            # if len(span) < 1200:
            out = self.model(encodings['input_ids']).logits.detach()[0].cpu().numpy()
        except RuntimeError:
            raise

        return np.ravel(log_softmax(out))  # logP(t|x,s)


    # ********************** For finetuning **********************

class SFDataset(torch.utils.data.Dataset):
    """
    Dataset object
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class_wts = None
label_smoothing_factor = 1e-2

class MultilabelTrainer(Trainer):
    """
    Trainer object that can use weighted classes
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        label_id = int(inputs.pop("labels").item())

        outputs = model(**inputs)
        logits = outputs.logits

        # logging.info(f"outputs: {outputs}")
        # logging.info(f"Size: {labels.size()}")
        # logging.info(f"class_wts: {class_wts}")

        labels = torch.ones(1, self.model.config.num_labels, dtype=torch.float64).to(dev) \
                 * label_smoothing_factor / (self.model.config.num_labels-1)
        labels[0, label_id] = 1 - label_smoothing_factor
        # logging.info(f"labels: {labels}")

        # label_smoother = LabelSmoother(epsilon=label_smoothing_factor)
        loss_fct = nn.BCEWithLogitsLoss(weight=torch.tensor(class_wts/sum(class_wts), dtype=torch.float64).to(dev))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def train_model(base_path, name=None, label_smoothing_factor=0., weighted=False, save=None, label_type=None, with_bins=False):
    """
    Trains a transformer model using HuggingFace's trainer
    :param save_model: what to save - can be "model", "eval" or None
    :param base_path:
    :param name:
    :param label_smoothing_factor:
    :param weighted:
    :return:
    """
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    r = ["sf_43019", "sf_38929", "sf_32788", "sf_38936", "sf_20505", "sf_23579", "sf_48155", "sf_35869", "sf_30751", "sf_30753",
         "sf_25639", "sf_45091", "sf_32809", "sf_34857", "sf_46120", "sf_46122", "sf_30765", "sf_24622", "sf_21550", "sf_26672"]
    r = [_r[3:] for _r in r]

    docs_path = base_path + '/data/docs/'
    if label_type is None:
        with open(docs_path + "data6.json", 'r') as infile:
            data = json.load(infile)
        data = [data_tuple for t, t_data in data.items() if t not in r for data_tuple in t_data]
    elif label_type == "time":
        with open(docs_path + "time_all.json", 'r') as infile:
            data = json.load(infile)
        with open(base_path + "/data/sf_nonrounds.json", 'r') as infile:
            nonrounds = json.load(infile).keys()
        data = [data_tuple for t, t_data in data.items() if t not in nonrounds for data_tuple in t_data]
    # np.random.shuffle(data)
    # store the inputs and outputs
    if label_type is None:
        if with_bins:
            texts, bins, labels = list(zip(*data))
            texts = [t + " [SEP] " + str(b) for t, b in zip(texts, bins)]
        else:
            texts, _, labels = list(zip(*data))
    elif label_type == "time":
        texts, labels = list(zip(*data))
    encoder = LabelEncoder()

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, encoder.fit_transform(labels), test_size=.2)
    print("made data")

    if label_type is None:
        # out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/deberta-large2'
        # out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/distilroberta'
        out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/gpt2-3'
    elif label_type == "time":
        # out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/xlnet-base-time'
        out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/distilroberta-time'
        # out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/xlnet-large-time'
    # out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/distilbert-textcat'
    # out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/electra-textcat'
    # out_path = base_path + 'models/electra-large-textcat'
    # out_path = '/cs/snapless/oabend/eitan.wagner/segmentation/models/xlnet-base-textcat'
    if "model" in save:
        # model.save_pretrained(out_path)
        # model = DistilBertForSequenceClassification.from_pretrained('/cs/snapless/oabend/eitan.wagner/segmentation/models/distilbert-textcat')
        # print("Evaluating")
        # print(trainer.evaluate())
        joblib.dump(encoder, out_path + '/label_encoder.pkl')

    if weighted:
        global class_wts
        class_wts = compute_class_weight('balanced', np.unique(train_labels), train_labels)
        # class_wts = class_wts/sum(class_wts)
        # logging.info(f"class_wts: {class_wts}")

    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    # tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    tokenizer = AutoTokenizer.from_pretrained('gpt2-large', cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    # tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large', cache_dir="/cs/snapless/oabend/eitan.wagner/cache/")
    print('distilroberta')
    # print('deberta-large')

    tokenizer.pad_token = tokenizer.eos_token
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    # train_encodings = tokenizer(train_texts, truncation=True, padding=False)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    # val_encodings = tokenizer(val_texts, truncation=True, padding=False)
    print("made encodings")

    train_dataset = SFDataset(train_encodings, train_labels)
    val_dataset = SFDataset(val_encodings, val_labels)


    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=5,              # total number of training epochs
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
        save_strategy="epoch",
        load_best_model_at_end=True,
        label_smoothing_factor=label_smoothing_factor,
        # report_to=None,
    )

    # model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base",
    #                                                             cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
    #                                                             num_labels=len(encoder.classes_))
    model = AutoModelForSequenceClassification.from_pretrained("gpt2-large",
                                                                cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
                                                                num_labels=len(encoder.classes_))
    # model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-large",
    #                                                            cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
    #                                                            num_labels=len(encoder.classes_))
    # model = XLNetForSequenceClassification.from_pretrained('xlnet-large-cased',
    #                                                        cache_dir="/cs/snapless/oabend/eitan.wagner/cache/",
    #                                                        num_labels=len(encoder.classes_))
    model.to(dev)
    # for name, param in model.named_parameters():
    #     if 'classifier' not in name: # classifier layer
    #         param.requires_grad = False

    print("Training")
    if not weighted:
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,             # evaluation dataset
            compute_metrics=compute_metrics,
        )
    else:
        trainer = MultilabelTrainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,
        )
    #
    trainer.train()


    if "model" in save:
        model.save_pretrained(out_path)
        # model = DistilBertForSequenceClassification.from_pretrained('/cs/snapless/oabend/eitan.wagner/segmentation/models/distilbert-textcat')
        # print("Evaluating")
        # print(trainer.evaluate())
        # joblib.dump(encoder, out_path + '/label_encoder.pkl')

    if "eval" in save:
        model.eval()
        tensors = torch.split(torch.tensor(val_encodings['input_ids'], dtype=torch.long), 1)
        preds = np.array([model(t.to(dev)).logits.detach().cpu().numpy().ravel() for t in tensors])  # should be a 2-d array
        np.save(out_path + "/eval_preds.npy", preds)


def eval_at_k(model, encodings, labels, k=None):
    """
    Calculated the @k accuracy
    :param model:
    :param encodings:
    :param labels:
    :param k:
    :return: list of accuracies
    """
    def find_k(arr):
        # the label is the last entry in the array
        return np.where(arr[:-1] == arr[-1])[0][0]
    model.eval()
    tensors = torch.split(torch.tensor(encodings['input_ids'], dtype=torch.long), 1)
    preds = np.array([model(t.to(dev)).logits.detach().cpu().numpy().ravel() for t in tensors])
    # preds = model(torch.tensor(encodings['input_ids'], dtype=torch.long).to(dev)).detach().cpu().numpy()
    # logging.info(preds.shape)
    # logging.info(len(labels))
    # logging.info(len(encodings))
    _sorted = np.argsort(preds, axis=1)[:, ::-1]
    ks = np.apply_along_axis(func1d=find_k, axis=1, arr=np.vstack((_sorted.T, labels)).T)
    accs = [sum(ks <= _k) / len(ks) for _k in range(k)]
    return accs


def find_correlations(path='/cs/snapless/oabend/eitan.wagner/segmentation/models/deberta-large'):
    """
    Make correlation scores (should be from 0 to 1) between all labels
    :param path:
    :return:
    """
    preds = np.load(path + "/eval_preds.npy")
    corr = np.corrcoef(preds, rowvar=False)
    np.save(path + "/correlation_matrix.npy", corr)
    # print("Done")

def cluster_correlations(path='/cs/snapless/oabend/eitan.wagner/segmentation/models/xlnet-large-cased'):
    import scipy.cluster.hierarchy as spc
    from scipy.spatial.distance import squareform
    cor_matrix = np.load(path + "/correlation_matrix.npy").astype(np.float32)

    pdist = squareform((1 + cor_matrix) / 2)
    linkage = spc.linkage(pdist, method='complete')
    id2cluster = (spc.fcluster(linkage, 5, 'maxclust') - 1).tolist()  # so it will start from 0

    clusters = [[] for _ in range(len(set(id2cluster)))]
    for i, c in enumerate(id2cluster):
        clusters[c].append(i)
    logging.info("clusters: ")
    logging.info(clusters)

def _train():
    base_path = '/cs/snapless/oabend/eitan.wagner/segmentation/'
    # model_names = ['distilbert-textcat', 'electra-large-textcat', 'xlnet-base-textcat']
    label_smoothing_factor = 1e-2
    # label_smoothing_factor = 1e-1
    # label_smoothing_factor = 0
    # logging.info(f"Smoothing factor: {label_smoothing_factor}")
    # weighted = True
    weighted = False
    logging.info(f"Is weighted: {weighted}")

    train_model(base_path=base_path, label_smoothing_factor=label_smoothing_factor,
                weighted=weighted, save=["model", "eval"], label_type=None)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    import logging.config
    logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True, })
    _train()

