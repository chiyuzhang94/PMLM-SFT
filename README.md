

# [Improving Social Meaning Detection with Pragmatic Masking and Surrogate Fine-Tuning](https://arxiv.org/abs/2108.00356)
#### Accepted by WASSA@ACL-2022: 

![](https://github.com/chiyuzhang94/PMLM-SFT/blob/master/pic/title.png)

![](https://github.com/chiyuzhang94/PMLM-SFT/blob/master/pic/illustration.png)

* [Pragmatic Masking Pre-training Script](https://github.com/chiyuzhang94/PMLM-SFT/language_modeling_emohash_h5.py)

* We continue training RoBERTa released on HuggingFace models: [RoBERTa-Base](https://huggingface.co/docs/transformers/model_doc/roberta)

* [Directory 'scripts'](https://github.com/chiyuzhang94/PMLM-SFT/tree/main/scripts) includes shell scripts to train PMLM with PyTorch distributed training.

* We run our codes with:
  * Python==3.6.8
  * torch==1.6.0
  * transformers==4.12.0
  * h5py==3.1.0

## Trained Models
* PragS1 (PMLM with Hashtag_end dataset followed by SFT-E): https://huggingface.co/UBC-NLP/prags1
* PragS2 (Best Model, PMLM with Emoji_any dataset followed by SFT-H): https://huggingface.co/UBC-NLP/prags2

You can load these models and use for downstream fine-tuning. For example:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('UBC-NLP/prags1', use_fast = True)
model = AutoModelForSequenceClassification.from_pretrained('UBC-NLP/prags1',num_labels=lable_size)
```
We evaluate our models on 15 social meaning tasks:

![](https://github.com/chiyuzhang94/PMLM-SFT/blob/master/pic/model_perf.png)

Please cite our paper:
```
@inproceedings{zhang-abdul-mageed-2022-improving,
    title = "Improving Social Meaning Detection with Pragmatic Masking and Surrogate Fine-Tuning",
    author = "Zhang, Chiyu  and
      Abdul-Mageed, Muhammad",
    booktitle = "Proceedings of the 12th Workshop on Computational Approaches to Subjectivity, Sentiment {\&} Social Media Analysis",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wassa-1.14",
    pages = "141--156",
```

