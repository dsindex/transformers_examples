# Description

- reference code for huggingface's [transformers](https://github.com/huggingface/transformers)
  - simple examples
  - NER task
  - GLUE task
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
* `run_ner.py` is old version from `transformers/example`.

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
* `transformers/examples/text-classification/run_glue.py` was copied and modified. 

$ ./run-glue.sh -v -v
...
10/29/2020 14:10:47 - INFO - __main__ -   ***** Eval results sst2 *****
10/29/2020 14:10:47 - INFO - __main__ -     eval_loss = 0.2560681700706482
10/29/2020 14:10:47 - INFO - __main__ -     eval_accuracy = 0.9243119266055045
10/29/2020 14:10:47 - INFO - __main__ -     epoch = 3.0
10/29/2020 14:10:47 - INFO - __main__ -     total_flos = 16988827310258688

* old version of `run_glue.py` == `run_glue_old_versoin.py`

$ python download_glue_data.py
$ ./run-glue-old-version.sh -v -v
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
$ cp -rf ../transformers/examples/language-modeling/run_language_modeling.py .
$ ./finetune-roberta.sh -v -v
```



# Training RoBERTa from scratch

- train
```
* prepare data

* split data if necessary
* $ python split.py --data_path=korean/all.txt --base_path=korean/data.txt --ratio=1000

$ cp -rf ../transformers/examples/language-modeling/run_language_modeling.py .

* edit vocab_size in config-roberta-base/config.json
$ ./train-roberta.sh -v -v

```


# Training DistilBert

- train
```
$ cp -rf ../transformers/examples/distillation .
$ cp distillation/training_configs/distilbert-base-cased.json distilbert-base.json
* place teacher model to current dir, ex) `pytorch.all.bpe.4.8m_step`
* modify distilbert-base.json, train-distilbert.sh : `vocab_size`
{
	"activation": "gelu",
	"attention_dropout": 0.1,
	"dim": 768,
	"dropout": 0.1,
	"hidden_dim": 3072,
	"initializer_range": 0.02,
	"max_position_embeddings": 512,
	"n_heads": 12,
	"n_layers": 6,
	"sinusoidal_pos_embds": true,
	"tie_weights_": true,
	"vocab_size": 202592
}
* modify distillation/train.py : `max_model_input_size`
args.max_model_input_size = 512

$ ./train-distilbert.sh -v -v
...
06/17/2020 21:37:02 - INFO - transformers.configuration_utils - PID: 2470 -  Configuration saved in korean/kor-distil-bpe-bert/config.json
06/17/2020 21:37:04 - INFO - utils - PID: 2470 -  Training is finished
06/17/2020 21:37:04 - INFO - utils - PID: 2470 -  Let's go get some drinks.

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
** add kor-distil-bpe-bert.v1/config.json
   "architectures": [
     "DistilBertModel"
   ],

** copy vocab
$ cp pytorch.all.bep.4.8m_step/vocab.txt kor-distil-bpe-bert.v1
** copy model
$ cp korean/kor-distil-bpe-bert/pytorch_model.bin kor-distil-bpe-bert.v1/
```

- what about distilling from BERT large?
  - 'attention heads', 'hidden size', 'FFN inner hidden size' are different.
  - therefore, we should train a modified BERT large with same 'attention heads', 'hidden size', 'FFN inner hidden size' from scratch.
  - and then, distil to distilbert.

# References

- https://huggingface.co/blog/how-to-train
- https://github.com/huggingface/transformers/blob/master/examples/distillation/README.md
