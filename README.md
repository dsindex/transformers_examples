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
$ ./run_ner.sh -v -v

1. bert-base-cased
* dev.txt
f1 = 0.9456403383867995
loss = 0.039552512255717376
precision = 0.9403631517574546
recall = 0.9509770889487871
* test.txt
f1 = 0.9065683175272216
loss = 0.12373894704744627
precision = 0.8986768802228412
recall = 0.9145995747696669

2. bert-large-cased
* dev.txt
f1 = 0.9561646131925237
loss = 0.03611929177992928
precision = 0.9524127567206545
recall = 0.9599461460787614
* test.txt
f1 = 0.9126537785588752
loss = 0.11602993070967689
precision = 0.9059665038381018
recall = 0.9194405099150141

3. roberta-large
* dev.txt
f1 = 0.9591888041565406
loss = 0.036815324224663694
precision = 0.9552662326823569
recall = 0.9631437226523056
* test.txt
f1 = 0.9138248443937933
loss = 0.13418018649573932
precision = 0.9050182323320021
recall = 0.9228045325779037
```
  - tensorboardX
```
$ tensorboard --logdir runs/event-dir-name/ --port port-number --bind_all
```
![data/eval_f1.png][]
