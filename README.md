# examples of transformers
reference code for [transformers](https://github.com/huggingface/transformers) of huggingface

# requirements

```
* python 3.6
* CUDA 10
$ pip install tensorflow-gpu==2.0
$ pip install torch==1.2.0
$ pip install git+https://github.com/huggingface/transformers.git
$ pip install seqeval
$ pip install tensorboardX
```

# contents

- Examples
  - BERT usage
```
$ python example1.py
```

- NER for CoNLL2003 eng dataset(only using word)
```
$ ./train-ner.sh -v -v
* select the checkpoint dir for the best model
* modify predict-ner.sh
* ex) predict ${OUTPUT_DIR}/checkpoint-3550
$ ./predict-ner.sh -v -v

1. bert-base-cased

* dev.txt

* test.txt

2. bert-large-cased

* dev.txt

* test.txt

3. roberta-large

* dev.txt

* test.txt

```
  - tensorboardX
```
$ tensorboard --logdir runs/event-dir-name/ --port port-number --bind_all
```
  - f1 score for dev set
![](/data/eval_f1.png)
