
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


---

## Segmentation

You can run a segmentation model with various options. 
For the topic-based segmentor you must provide your own classifier and a list of length-per-class.

- **--out_path** .


  
For example, 
```bash
python segmentation.py --data_path <data_path>raw_text.json --out_path <out_path>

```


---
 
## Evaluation
 
You can run evaluation for a previously segmented document. For this you must provide a SpaCy DocBin with segmented docs. 

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
