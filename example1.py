import sys
import torch
from transformers import ElectraTokenizer, ElectraForMaskedLM

## 1
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer
tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-discriminator')

# Tokenize input
text = "[CLS] 죽느냐 사느냐 그것이 문제로다. [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
'''
['[CLS]', '죽', '##느냐', '사', '##느냐', '그것이', '문제로', '##다', '.', '[SEP]']
'''

# Mask a token that we will try to predict back
masked_index = 5
tokenized_text[masked_index] = '[MASK]'
print(tokenized_text)

# Convert token to vocabulary indices
token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
token_type_ids = [0] * len(token_ids)

print(token_ids)
print(token_type_ids) # segment_ids

# Convert inputs to PyTorch tensors
token_ids_tensor = torch.tensor([token_ids]).to('cuda')
token_type_ids_tensor = torch.tensor([token_type_ids]).to('cuda')

## 2
# Load pre-trained model (weights)
model = ElectraForMaskedLM.from_pretrained('monologg/koelectra-base-discriminator')

# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()
model.to('cuda')

## 3
# Predict all tokens
with torch.no_grad():
    outputs = model(token_ids_tensor, token_type_ids=token_type_ids_tensor)
    predictions = outputs[0]
print(predictions)

predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print('[MASK] =>', predicted_token)
