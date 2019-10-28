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

- NER for CoNLL2003 eng dataset(using word only)
```
$ ./train-ner.sh -v -v

* select the checkpoint dir for the best model
* modify evalt-ner.sh
* ex) evaluate ${OUTPUT_DIR}/checkpoint-3550
$ ./eval-ner.sh -v -v

1. bert-base-cased

* dev.txt

* test.txt

2. bert-large-cased

* dev.txt

* test.txt

3. roberta-large

* dev.txt
f1 = 0.9630064591896651
loss = 0.0625708463778838
precision = 0.960026760327814
recall = 0.9660047122181084

* test.txt
f1 = 0.9118423367939468
loss = 0.2133701834467194
precision = 0.9062609303952431
recall = 0.9174929178470255

```
  - tensorboardX
```
$ tensorboard --logdir runs/event-dir-name/ --port port-number --bind_all
```
  - f1 score for dev set
![](/data/eval_f1.png)
