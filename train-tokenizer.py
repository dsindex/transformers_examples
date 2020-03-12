from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
from collections import Counter 
import pdb
import logging

from tqdm import tqdm
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='wikitext-2-raw')
    parser.add_argument('--file_suffix', type=str, default='raw')
    parser.add_argument('--vocab_size', default=52000, type=int)
    parser.add_argument('--min_frequency', default=2, type=int)
    parser.add_argument('--tokenizer_name', type=str, default='roberta')
    opt = parser.parse_args()

    inc_paths = [str(x) for x in Path(opt.data_dir).glob("**/*.%s" % (opt.file_suffix))]
    exc_paths = [str(x) for x in Path(opt.data_dir).glob("**/*cached*")]
    paths = list(set(inc_paths) - set(exc_paths))

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=opt.vocab_size, min_frequency=opt.min_frequency, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save(".", opt.tokenizer_name)

if __name__ == '__main__':
    main()
