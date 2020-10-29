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
$ cp -rf ../transformers/examples/token-classification .

* roberta-base

$ ./train-ner.sh -v -v

$ ./eval-ner.sh -v -v
```

- tensorboardX
```
$ tensorboard --logdir engeval-model/runs/ --port port-number --bind_all
```



# GLUE Task

- run
```
$ cp -rf ../transformers/examples/text-classification .

$ ./run-glue.sh -v -v
...
10/29/2020 14:10:47 - INFO - __main__ -   ***** Eval results sst2 *****
10/29/2020 14:10:47 - INFO - __main__ -     eval_loss = 0.2560681700706482
10/29/2020 14:10:47 - INFO - __main__ -     eval_accuracy = 0.9243119266055045
10/29/2020 14:10:47 - INFO - __main__ -     epoch = 3.0
10/29/2020 14:10:47 - INFO - __main__ -     total_flos = 16988827310258688
```



# Finetune RoBERTa

- download raw corpus
  - ex) https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

- train
```
$ cp -rf ../transformers/examples/language-modeling .

$ ./finetune-roberta.sh -v -v

* trouble shooting
  ...
  File "/usr/local/lib/python3.6/dist-packages/transformers/modeling_roberta.py", line 98, in forward
      position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
        File "/usr/local/lib/python3.6/dist-packages/transformers/modeling_roberta.py", line 1333, in create_position_ids_from_input_ids
            mask = input_ids.ne(padding_idx).int()
  ...
  ne() received an invalid combination of arguments - got (NoneType)

  modify 'pad_token_id: 1' in config-roberta-base/config.json

```



# Training RoBERTa from scratch

- train
```
* prepare data

* split data if necessary
* $ python split.py --data_path=korean/all.txt --base_path=korean/data.txt --ratio=1000

$ cp -rf ../transformers/examples/language-modeling .

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

- distilling from BERT large?
  - 'attention heads', 'hidden size', 'FFN inner hidden size' are different.
  - therefore, we should train a modified BERT large with same 'attention heads', 'hidden size', 'FFN inner hidden size' from scratch.
  - and then, distil to distilbert.

# References

- https://huggingface.co/blog/how-to-train
- https://github.com/huggingface/transformers/blob/master/examples/distillation/README.md
