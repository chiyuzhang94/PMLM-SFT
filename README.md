

# [Improving Social Meaning Detection with Pragmatic Masking and Surrogate Fine-Tuning](https://arxiv.org/abs/2108.00356)
#### Accepted by WASSA@ACL-2022: 

[Pragmatic Masking Pre-training Script](https://github.com/chiyuzhang94/PMLM-SFT/language_modeling_emohash_h5.py)

We continue training RoBERTa released on HuggingFace models: [RoBERTa-Base](https://huggingface.co/docs/transformers/model_doc/roberta)

[Directory 'scripts'](https://github.com/chiyuzhang94/PMLM-SFT/tree/main/scripts) includes shell scripts to train PMLM with PyTorch distributed training.

We run our codes with:
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


