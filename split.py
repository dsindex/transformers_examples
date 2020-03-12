from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
from collections import Counter 
import pdb
import logging

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(data_path):
    logger.info("[read data]")
    data = []
    doc = []
    tot_num_line = sum(1 for _ in open(data_path, 'r')) 
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(tqdm(f, total=tot_num_line)):
            line = line.strip()
            if line == '':
                data.append(doc)
                doc = []
            else:
                doc.append(line)
        if len(doc) != 0:
            data.append(doc)
    logger.info("number of docs : {}".format(len(data)))
    return data

def write_data(data, ratio, base_path):
    logger.info("[write data]")
    train_path = base_path + '.train'
    valid_path = base_path + '.valid'
    tot_num_docs = len(data)
    limit = tot_num_docs / ratio
    f_train = open(train_path, 'w', encoding='utf-8')
    f_valid = open(valid_path, 'w', encoding='utf-8')
    for idx, doc in enumerate(tqdm(data)):
        if idx > limit:
            f_train.write('\n'.join(doc))
            f_train.write('\n')
            f_train.write('\n')
        else:
            f_valid.write('\n'.join(doc))
            f_valid.write('\n')
            f_valid.write('\n')
    f_train.close()
    f_valid.close()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, default='korean/all.txt')
    parser.add_argument('--base_path', type=str, default='korean/data.txt')
    parser.add_argument('--ratio', type=int, default=1000)
    opt = parser.parse_args()

    data = read_data(opt.data_path)
    write_data(data, opt.ratio, opt.base_path)

if __name__ == '__main__':
    main()
