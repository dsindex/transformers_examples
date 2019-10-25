# examples of transformers
reference code for [transformers](https://github.com/huggingface/transformers) of huggingface

# requirements

- python 3.6
- pip install tensorflow-gpu==2.0
  - CUDA 10
- pip install torch==1.2.0
  - CUDA 10
- pip install seqeval
- pip install tensorboardX

# contents

- examples1.py
  - BERT usage
```
$ python example1.py
```

- run_ner.sh, run_ner.py
  - NER for CoNLL2003 eng dataset(only using word)
```
$ ./run_ner.sh -v -v

* bert-base-cased, dev.txt
10/25/2019 18:25:08 - INFO - __main__ -   ***** Running evaluation  *****
10/25/2019 18:25:08 - INFO - __main__ -     Num examples = 3250
10/25/2019 18:25:08 - INFO - __main__ -     Batch size = 8
10/25/2019 18:25:20 - INFO - __main__ -   ***** Eval results  *****
10/25/2019 18:25:20 - INFO - __main__ -     f1 = 0.9456403383867995
10/25/2019 18:25:20 - INFO - __main__ -     loss = 0.039552512255717376
10/25/2019 18:25:20 - INFO - __main__ -     precision = 0.9403631517574546
10/25/2019 18:25:20 - INFO - __main__ -     recall = 0.9509770889487871

* bert-base-cased, test.txt
10/25/2019 18:25:30 - INFO - __main__ -   ***** Running evaluation  *****
10/25/2019 18:25:30 - INFO - __main__ -     Num examples = 3453
10/25/2019 18:25:30 - INFO - __main__ -     Batch size = 8
10/25/2019 18:25:43 - INFO - __main__ -   ***** Eval results  *****
10/25/2019 18:25:43 - INFO - __main__ -     f1 = 0.9065683175272216
10/25/2019 18:25:43 - INFO - __main__ -     loss = 0.12373894704744627
10/25/2019 18:25:43 - INFO - __main__ -     precision = 0.8986768802228412
10/25/2019 18:25:43 - INFO - __main__ -     recall = 0.9145995747696669

* bert-large-cased, dev.txt

* bert-large-cased, test.txt
```
  - tensorboardX
```
$ tensorboard --logdir runs/event-dir-name/ --port 22092 --bind_all
```
