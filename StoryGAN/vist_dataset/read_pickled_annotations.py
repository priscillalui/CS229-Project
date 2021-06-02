import pickle
import torch

# ENCODINGS_FILE = '../Val/val_text_encodings.pt'
PICKLE_FILE = '/home/Priscilla/VIST/Val/val_annotations.pickle'

# encodings = torch.load(ENCODINGS_FILE)
stories = pickle.load( open( PICKLE_FILE, "rb" ) )

for i,story in enumerate(stories):
  if not story:
    print('Story %d is None or empty' % i)
  story_len = len(story)
  # print("Story len: %d" % story_len)
  # for img_id,index in story:
    # print(index)
    # encoding = encodings[index]
    # print((img_id, '<Encoding of size %d>' % encoding.size()))

print("Num total stories: %d" % len(stories))