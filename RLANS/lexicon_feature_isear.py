from data_collection.py_isear.isear_loader_spacy import IsearLoader
import nltk
import random


attributes = []
target = ['EMOT']
loader = IsearLoader(attributes, target, True)
dataset = loader.load_isear('isear_cleaned.csv')
documents = []
for text, target in zip(dataset.get_data(), dataset.get_target()):
    documents.append((text, target[0]))
random.shuffle(documents)

all_tokens = [element for lis in dataset.get_data() for element in lis]
Freq_dist_nltk = nltk.FreqDist(all_tokens)
# print(Freq_dist_nltk)
most_common_word = [word for (word, _) in Freq_dist_nltk.most_common(8000)]


def doc_feature(doc):
    doc_words = set(doc)
    feature = {}
    for word in most_common_word:
        feature[word] = (word in doc_words)
    return feature


train_set = nltk.apply_features(doc_feature, documents)
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features(n=20)
features = classifier.most_informative_features(n=8000)

# print(features)
features_count = []
for word in features:
    w = word[0]
    class_count = [0] * 7
    for doc in documents:
        word_set = set(doc[0])
        if w in word_set:
            # print(doc[1]-1)
            class_count[doc[1]-1] += 1
    # print(class_count)
    features_count.append((w, class_count))
import json
json.dump(features_count, open('RLANS/data_collection/py_isear/isear_feature_count.json', 'w'), indent=2)