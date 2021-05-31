import json
import argparse

import numpy as np
import nltk

from collections import defaultdict
from nltk.tokenize import word_tokenize
from tqdm import tqdm

def texts_to_labels(src_path, common_words_path=None, K=50):
    with open(src_path) as src_file:
        src_json = json.load(src_file)
        annotations = src_json['annotations']
        tokenized_texts = _annotations_to_tokens_list(annotations)

        if common_words_path is None:
            common_words = _build_common_keywords(K)
        else:
            common_words = np.load(common_words_path)
        common_words_idx = {word: i for i, word in enumerate(common_words)}
        
        print('Begin creating labels...')
        labels = [np.zeros(K) for _ in range(len(tokenized_texts))]
        for i, sentence in enumerate(tqdm(tokenized_texts)):
            for word in sentence:
                if word in common_words_idx:
                    labels[i][common_words_idx[word]] += 1
        print('Created %d labels with %d common words' % (len(labels), len(labels[0])))
    return labels

def _annotations_to_tokens_list(annotations):
    for annot in annotations:
        data = annot[0]
        tokenized_texts.append(_sentence_to_tokens(data['text']))    

def _sentence_to_tokens(self, sentence):
    return word_tokenize(sentence)

def _build_common_keywords(self, K):
    is_noun = lambda pos: pos[:2] == 'NN'

    with open(src_path) as src_file:
        src_json = json.load(src_file)
        annotations = src_json['annotations']
        tokenized_texts = _annotations_to_tokens_list(annotations)
        
        print('Begin counting word occurrences....')
        fdist = nltk.FreqDist()
        for sentence in tqdm(tokenized_texts):
            for word, pos in nltk.pos_tag(sentence):
                if is_noun(pos):
                    fdist[word] += 1
        common_words = fdist.most_common(K)
        print('K-Common Nouns: ', common_words)        
        return common_words

def extract_stories(self):
    annotations = src_json['annotations']
    stories = defaultdict(list)
    for annotation in annotations:
        data = annotation[0]
        story_id, storylet_id, image_id = \
            data["story_id"], data["storylet_id"], data["photo_flickr_id"]
        stories[story_id].append({"storylet_id": storylet_id, "image_id": image_id})
    return stories        

def save_to_json(self, arr, dst_path):
    with open(dst_path, 'w') as dst_file:
        json.dump(arr, dst_file)

def save_to_npy(self, arr, dst_path):
    with open(dst_path, 'w') as dst_file:
        np.save(dst_path, arr)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--src', dest='src_path', type=str, default='train.story-in-sequence.json')
    # parser.add_argument('--dst', dest='dsc_path', type=str, default='train.stories.json')
    # args = parser.parse_args()


    # Extract necessary story, storylet, and image information from the orignal json file
    # stories = extract_stories()
    # save_to_json(stories, 'train.stories.json')

    # Extract most common K words from the original json file
    labels = texts_to_labels(src_path='train.story-in-sequence.json', common_words_path='~/VIST/Train/train_common_nouns.npy', K=50)
    save_to_npy(labels, 'train_labels.npy')
