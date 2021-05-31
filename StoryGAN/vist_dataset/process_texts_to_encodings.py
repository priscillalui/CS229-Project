'''
NOTE: This does the encodings in batches/splits/partitions so the GPU/CPU
doesn't run out of memory. Use combine_encoding_splits.py 
next to combine the tensors.

Pre-processing Step 1:
Input: JSON file
Output: Pytorch (.pt) file
Generates a pytorch file containing a list of the text caption CLIP encodings.
This is generated as a preprocessing step because it is more efficient for 
the CLIP model to encode in large batches rather than 1 string at a time.


'''
import json
import os.path
from os import path
import clip
import torch

NUM_SPLITS = 20
JSON_FILE = '/home/Priscilla/VIST/dii/train.description-in-isolation.json'
ENCODINGS_FILE = '/home/Priscilla/VIST/Train/dii_train_text_encodings_{}_{}.pt' # a list of the text encodings in the same order as they appear in the file

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: %s" % device)

# Load the CLIP text encoder model
print("Loading CLIP model...")
model, preprocess = clip.load('ViT-B/32', device)

# Load the json annotation file
print("Loading json file...")
with open(JSON_FILE) as f:
  data = json.load(f)

annotations = data['annotations']
one_split_size = len(annotations)//(NUM_SPLITS-1)

# Do the encodings split-by-split
file_num = 1
for start in range(0, len(annotations), one_split_size):
  end = start + one_split_size
  if end > len(annotations):
    end = len(annotations)
  print('Start', start, 'End', end, 'Total', len(annotations))

  raw_texts = []
  for annot in annotations[start:end]:
    # Extract relevant fields
    annot_dict = annot[0]
    raw_texts.append(annot_dict['text'])

  print("Clearing cuda cache...")
  torch.cuda.empty_cache()

  # Prepare the inputs
  print("Tokenizing raw texts...")
  text_inputs = torch.cat([clip.tokenize(text) for text in raw_texts]).to(device)

  # Calculate features
  print("Encoding tokenized texts...")
  with torch.no_grad():
    text_features = model.encode_text(text_inputs)

  split_file_name = ENCODINGS_FILE.format(file_num, NUM_SPLITS)
  print("Saving to torch file %s" % split_file_name)
  torch.save( text_features, open( split_file_name , "wb" ) )
  file_num += 1
    