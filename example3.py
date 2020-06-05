from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import json
import pdb
import logging

from transformers import BertTokenizer
from transformers import AlbertTokenizer
from transformers import RobertaTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='roberta')
    parser.add_argument('--model_name_or_path', type=str, default='config-kor-roberta-base',
                        help="Path to pre-trained model or shortcut name(ex, bert-base-uncased)")
    parser.add_argument('--do_lower_case', action='store_true',
                        help="Set this flag if you are using an uncased model.")
    opt = parser.parse_args()

    TOKENIZER_CLASSES = {
        "bert": BertTokenizer,
        "albert": AlbertTokenizer,
        "roberta": RobertaTokenizer
    }
    Tokenizer = TOKENIZER_CLASSES[opt.model_type]

    tokenizer = Tokenizer.from_pretrained(opt.model_name_or_path,
                                          do_lower_case=opt.do_lower_case)
  
    print('vocab_size: ', tokenizer.vocab_size)

    sample = "나는 학교에 간다. i wanna go out. 저항하는 개막전 호랑이가"
    tokens = []
    for word in sample.split():
        word_tokens = tokenizer.tokenize(word)
        tokens.extend(word_tokens)
    print(' '.join(tokens))



if __name__ == '__main__':
    main()
