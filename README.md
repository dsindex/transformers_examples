# Description

- reference code for named entity tagging using huggingface's [transformers](https://github.com/huggingface/transformers)
- fintuning or training RoBERTa from scratch.

# Requirements

```
* python 3.6
* CUDA 10
$ pip install -r requirements
$ pip install tensorflow-gpu==2.0.0
$ pip install git+https://github.com/huggingface/transformers.git
```

# Examples

- BERT usage
```
$ python example1.py
```

# NER for CoNLL2003 eng dataset(using word only)

- train and evaluate
```
$ ./train-ner.sh -v -v

* select the checkpoint dir for the best model, you may refer to the tensorboard.
* modify eval-ner.sh
* ex) evaluate ${OUTPUT_DIR}/checkpoint-3550
$ ./eval-ner.sh -v -v

1. bert-base-cased

* dev.txt

* test.txt

2. bert-large-cased

* dev.txt

* test.txt

3. roberta-large

* we can find possible max-f1-score for 'test.txt'.
  => results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")
  actual f1-score for 'test.txt' is usually lower than this.
  => results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")

* dev.txt
f1 = 0.9560697518443997
loss = 0.047375404791811564
precision = 0.9525559639158035
recall = 0.9596095590710199

* test.txt
f1 = 0.9216322948661694
loss = 0.15627447860190471
precision = 0.913694101270228
recall = 0.9297096317280453

```

- tensorboardX
```
$ tensorboard --logdir engeval-model/runs/ --port port-number --bind_all
```

# Finetune RoBERTa

- download raw corpus
  - ex) https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

- train
```
$ ./finetune-roberta.sh -v -v
```

# Training RoBERTa from scratch

- train
```
* prepare data

* split data if necessary
* $ python split.py --data_path=korean/all.txt --base_path=korean/data.txt --ratio=1000

* edit vocab_size in config-roberta-base/config.json
$ ./train-roberta.sh -v -v

* something goes wrong for training Korean RoBERTa from scratch. why?
* i guess that byte-pair encoding is not suitable for Korean. try to use char-based bpe tokenizer.
```

# References

- https://huggingface.co/blog/how-to-train
