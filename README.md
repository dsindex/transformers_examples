# Description

reference code for [transformers](https://github.com/huggingface/transformers) of huggingface

# Requirements

```
* python 3.6
* CUDA 10
$ pip install tensorflow-gpu==2.0
$ pip install torch==1.2.0
$ pip install git+https://github.com/huggingface/transformers.git
$ pip install seqeval
$ pip install tensorboardX
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
f1 = 0.9567476948868399
loss = 0.0387861595115888
precision = 0.9530728122912492
recall = 0.9604510265903736

* test.txt
f1 = 0.9148431320854206
loss = 0.14893963946945982
precision = 0.9082184610015704
recall = 0.9215651558073654

```

- tensorboardX
```
$ tensorboard --logdir engeval-model/runs/ --port port-number --bind_all
```

- f1 score for dev set
![](/data/eval_f1.png)
