"""
Combines a list of encodings, in the form of pytorch files.
"""

NUM_SPLITS = 2 # Must match total number of files
ENCODINGS_FILES = ['/home/Priscilla/VIST/Val/dii_val_text_encodings_{}_{}.pt'.format(i, NUM_SPLITS) for i in range(1, NUM_SPLITS+1)]
FINAL_ENCODINGS = '/home/Priscilla/VIST/Val/dii_val_text_encodings.pt'

import torch

print('Combining', len(ENCODINGS_FILES), 'encodings files')
final_encodings = torch.load(ENCODINGS_FILES[0])
for i in range(1, len(ENCODINGS_FILES)):
  print('Current size', final_encodings.size())
  encoding_file = ENCODINGS_FILES[i]
  encoding_split = torch.load(encoding_file)
  final_encodings = torch.cat((final_encodings, encoding_split))

print('Final size', final_encodings.size())

print("Saving to torch file %s" % FINAL_ENCODINGS)
torch.save( final_encodings, open( FINAL_ENCODINGS, "wb" ) )