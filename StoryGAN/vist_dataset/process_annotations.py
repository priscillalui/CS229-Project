'''
Pre-processing Step 2:
Input: Description-in-isolation JSON file
Output: Pickle file

Generates a list of lists, where each inner list represents a story containing tuples of (img_id, index into the encodings tensor)
[
  [(123, 0), (126, 1), ... )],
  ...
  [(183, 15), (189, 16), ... )],
  ...
]
Omits stories that have 1 or more missing images.
Saves this object to PICKLE_FILE.

'''
from PIL import Image
import json
import os.path
from os import path
import pickle
import torch

IMG_DIR = '/home/Priscilla/VIST/Train/images/train'
JSON_FILE = '/home/Priscilla/VIST/dii/test.description-in-isolation.json'
PICKLE_FILE = '/home/Priscilla/VIST/Test/dii_test_annotations.pickle' # where to save the final list of story annotations

num_missing_images = 0
num_omitted_stories = 0


# Load the json annotation file
with open(JSON_FILE) as f:
  data = json.load(f)

annotations = data['annotations']
stories = []
story = []
prev_photo_index = 0
is_story_intact = True # whether all scene image files for this story exist

for index,annot in enumerate(annotations):

  # Extract relevant fields
  annot_dict = annot[0]
  photo_id = annot_dict['photo_flickr_id']
  photo_index = annot_dict['photo_order_in_story']

  # Check if this is a new story
  if prev_photo_index != 0 and photo_index == 0:
    if is_story_intact:
      # print("Adding story, contains %d scenes" % len(story))
      stories.append(story.copy())
    else:
      print("Omitting story because missing a scene")
      num_omitted_stories += 1
      is_story_intact = True 
    story.clear()

  # Check if image exists
  img_file = os.path.join('%s/%s.jpg' % (IMG_DIR, photo_id))
  # print("Checking if %s exists..." % img_file)
  if not os.path.exists(img_file):
    num_missing_images += 1
    is_story_intact = False
  else:
    try:
      Image.open(img_file)
    except:
      num_missing_images += 1
      is_story_intact = False

  # Append to story
  story.append((photo_id, index))
  prev_photo_index = photo_index

print("Adding final story, contains %d scenes" % len(story))
stories.append(story)


print("Num missing images: %d" % num_missing_images)
print("Num omitted stories: %d" % num_omitted_stories)

print("Num total stories: %d" % len(stories))
# print(stories)

# Pickle stories
print("Pickling to %s" % PICKLE_FILE)
pickle.dump( stories, open( PICKLE_FILE, "wb" ), protocol=4)


# # Pickle 1000 stories at a time
# PICKLE_FILE_LEN = 100
# for i in range(0, len(stories), PICKLE_FILE_LEN):
#   pickle_file = PICKLE_FILE + '_' + str(i) + '.pickle'
#   print("Pickling to %s" % pickle_file)
#   start = i
#   end = i+PICKLE_FILE_LEN if (i+PICKLE_FILE_LEN < len(stories)) else len(stories) 
#   pickle.dump( stories[start:end], open( pickle_file, "wb" ), protocol=4)
#   break
    
