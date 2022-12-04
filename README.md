
# Holocaust-Segmentation Repository

Code for the paper "Topical Segmentation of Spoken Narratives: A Test Case on Holocaust Survivor Testimonies"
https://arxiv.org/abs/2210.13783

---

## Install
```bash
git clone https://github.com/eitanwagner/holocaust-segmentation.git
cd holocaust-segmentation
pip install -e requirements.txt

```


---

## Segmentation

You can run a segmentation model with various options. 

For the topic-based segmentor you must provide your own classifier and a list of length-per-class.

The output will be a SpaCy DocBin and the segments will be in doc.spans["segments<num_segments>"].


Arguments for segmentation:
- **--num_segments**, type=int, default=20 - number of segments to divide into.
- **--method**, type=str, default='pmi' - segmentation method - can be "pmi", "nsp" or "uniform".
- **--window**, type=int, default=3 - window size for pmi and nsp methods"
- **--data_path**, type=str - path to documents (name included)"
- **--out_path**, type=str, - path to save spacy DocBin with docs with segments (in doc.spans['segments20'] etc.)
- **--size**, type=str, default='base' - PMI language model size. can be 'base', 'large' or 'xlarge'
- **--alpha**, type=float, default=0.2 - alpha parameter for the dynamic method
- **--beta**, type=float, default=0.2 - beta parameter for the dynamic method
- **--classifier_path**, type=str, default=None - path (folder) to trained huggingface classifier
- **--classifier_name**, type=str, default=None - name of huggingface classifier folder
- **--from_spacy**, type=bool, default=False - whether the documents are in json format or already parsed by spacy
- **--cache_dir**, type=str, default=None
- **--class_encoder_path**, type=str, default=None - path to label encoder (for the classifier)
- **--use_len**, type=bool, default=False - whether do use length regularization
- **--lens_path**, type=str, default=None - path to json list with average length for each class

  
For example, 
```bash
python segmentation.py --data_path <data_path>raw_text.json --out_path <out_path>

```


---
 
## Evaluation
 
You can run evaluation for a previously segmented document. For this you must provide a SpaCy DocBin with segmented docs. 

Arguments for evaluation:
- **--num_segments**, type=int, default=20 - number of segments that the documents were divided into.
- **--classifier_path**, type=str, default=None - path (folder) to trained huggingface classifier
- **--classifier_name**, type=str, default=None - name of huggingface classifier folder
- **--cache_dir**, type=str, default=None
- **--class_encoder_path**, type=str, default=None - path to label encoder (for the classifier)
- **--gold_path**, type=str, default=None, help="path to gold documents (name included) - for evaluation

For example, 
```bash
python evaluation.py --data_path <data_path>doc_bin --gold_path <data_path>doc_bin-gold --class_encoder_path <encoder_path>label_encoder.pkl --classifier_path <classifier_path> --classifier_name distilroberta
```


---

# Citation

```bibtex
@misc{https://doi.org/10.48550/arxiv.2210.13783,
  doi = {10.48550/ARXIV.2210.13783},
  url = {https://arxiv.org/abs/2210.13783},
  author = {Wagner, Eitan and Keydar, Renana and Pinchevski, Amit and Abend, Omri},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Topical Segmentation of Spoken Narratives: A Test Case on Holocaust Survivor Testimonies},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```
