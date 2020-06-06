# Description

- reference code for huggingface's [transformers](https://github.com/huggingface/transformers)
- fintuning or training RoBERTa from scratch.
- training DistilBert


# Requirements

```
* python >= 3.6
$ pip install -r requirements
$ pip install git+https://github.com/huggingface/transformers.git
```



# Examples

```
$ python example1.py
$ python example2.py
$ python example3.py
```



# NER for CoNLL2003 eng dataset

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



# GLUE Task

- run
```
$ ./run-glue.sh -v -v
...
05/14/2020 14:01:58 - INFO - __main__ -   ***** Eval results sst-2 *****
05/14/2020 14:01:58 - INFO - __main__ -     acc = 0.9162844036697247
05/14/2020 14:01:58 - INFO - __main__ -     loss = 0.6246312452214104
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

```


# Training DistilBert

- train
```
$ cp -rf ../transformers/examples/distillation .
$ cp distillation/training_configs/distilbert-base-cased.json distilbert-base.json
* place teacher model, ex) `pytorch.all.bpe.4.8m_step`
* modify distilbert-base.json, train-distilbert.sh, `vocab_size`
* modify distillation/train.py, `args.max_model_input_size = 512`
$ ./train-distilbert.sh -v -v
* training parameters
$ cat korean/kor-distil-bpe-bert/parameters.json
   ...
   "n_epoch": 3,
   "batch_size": 5,
   "group_by_size": true,
   "gradient_accumulation_steps": 50,
   ...

* tensorboardX
$ tensorboard --logdir korean/kor-distil-bpe-bert/log/train --port port-number --bind_all

* make model archive, ex) kor-distil-bpe-bert.v1
$ cp -rf distilbert-base-uncased kor-distil-bpe-bert.v1
$ cp -rf korean/kor-distil-bpe-bert/config.json kor-distil-bpe-bert.v1
** add to kor-distil-bpe-bert.v1/config.json
   "architectures": [
     "DistilBertModel"
   ],
** copy vocab
$ cp pytorch.all.bep.4.8m_step/vocab.txt kor-distil-bpe-bert.v1
** copy model
$ cp korean/kor-distil-bpe-bert/checkpoint.pth kor-distil-bpe-bert.v1/pytorch_model.bin

```


# References

- https://huggingface.co/blog/how-to-train
- https://github.com/huggingface/transformers/blob/master/examples/distillation/README.md
