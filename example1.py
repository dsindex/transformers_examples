import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

## 1
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']

print(tokenized_text)
'''
['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer', '[SEP]']
'''

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

print(indexed_tokens)
'''
[101, 2040, 2001, 3958, 27227, 1029, 102, 3958, 103, 2001, 1037, 13997, 11510, 102]
'''

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

print(tokens_tensor)
'''
tensor([[  101,  2040,  2001,  3958, 27227,  1029,   102,  3958,   103,  2001,
           1037, 13997, 11510,   102]])
'''

## 2
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')

"""
# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0]
    print(encoded_layers)
    '''
    tensor([[[-0.5570,  0.2839, -0.6436,  ..., -0.7274,  0.4557,  0.6204],
         [-1.1572,  0.0354,  0.0355,  ...,  0.0591,  0.1097, -0.3150],
         [ 0.1008, -0.5286, -0.4688,  ..., -0.0431,  0.4889,  0.4134],
         ...,
         [ 0.1948,  0.0761,  0.2893,  ..., -0.0807,  0.7071,  0.0502],
         [-0.1119,  0.0714,  0.6101,  ...,  0.4044,  0.1614, -0.3569],
         [ 0.8296,  0.2729, -0.3090,  ...,  0.2452, -0.3581, -0.1970]]],
       device='cuda:0')
    '''
# We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
assert tuple(encoded_layers.shape) == (1, len(indexed_tokens), model.config.hidden_size)
"""

## 3
# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)  # predicted_token == 'henson'
'''
[unused171]
?
'''
