from data_collection.py_tec.tec_loader_spacy import TecLoader
import nltk
import random


loader = TecLoader(True)
dataset = loader.load_tec('/mount/arbeitsdaten31/studenten1/dialog-system/2020/student_directories/Changxu_Duan/thesis/RLANS/data_collection/tec.txt')
documents = []
for text, target in zip(dataset.get_data(), dataset.get_target()):
    documents.append((text, target))
random.shuffle(documents)

all_tokens = [element for lis in dataset.get_data() for element in lis]
Freq_dist_nltk = nltk.FreqDist(all_tokens)
most_common_word = [word for (word, _) in Freq_dist_nltk.most_common(2000)]


def doc_feature(doc):
    doc_words = set(doc)
    feature = {}
    for word in most_common_word:
        feature[word] = (word in doc_words)
    return feature


train_set = nltk.apply_features(doc_feature, documents)
classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier.show_most_informative_features(n=20)
features = classifier.most_informative_features(n=100)

# print(features)
features_count = []
for word in features:
    w = word[0]
    class_count = [0] * 6
    for doc in documents:
        word_set = set(doc[0])
        if w in word_set:
            class_count[doc[1]-1] += 1
    print(class_count)
    features_count.append((w, class_count))
import json
json.dump(features_count, open('/mount/arbeitsdaten31/studenten1/dialog-system/2020/student_directories/Changxu_Duan/thesis/RLANS/data_collection/py_tec/tec_feature_count.json', 'w'), indent=2)