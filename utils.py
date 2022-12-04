
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_segments', type=int, default=20, help="number of segments to divide into")
    parser.add_argument('--method', type=str, default='pmi', help="segmentation method")
    parser.add_argument('--window', type=int, default=3, help="window size for pmi and nsp methods")
    parser.add_argument('--size', type=str, default='base', help="PMI language model size. can be 'base', 'large' or 'xlarge'")
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--data_path', type=str, help="path to documents (name included)")
    parser.add_argument('--from_spacy', type=bool, default=False, help="whether the documents are in json format or already parsed by spacy")
    parser.add_argument('--out_path', type=str, help="path to save spacy DocBin with docs with segments (in doc.spans['segments20'] etc.)")
    parser.add_argument('--gold_path', type=str, default=None, help="path to gold documents (name included) - for evaluation")
    parser.add_argument('--classifier_path', type=str, default=None, help="path (folder) to trained huggingface classifier")
    parser.add_argument('--classifier_name', type=str, default=None, help="name of huggingface classifier folder")
    parser.add_argument('--class_encoder_path', type=str, default=None, help="path to label encoder (for the classifier)")
    parser.add_argument('--use_len', type=bool, default=False, help="whether do use length regularization")
    parser.add_argument('--lens_path', type=str, default=None, help="path to json list with average length for each class")
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)

    args = parser.parse_args()
    return args
