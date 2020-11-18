from data_collection.py_ssec.ssec_loader_spacy import SsecDataset, SsecLoader
import nltk
import random
import json


dataset = SsecDataset([],[])
dataset.text_data = json.load(open('onetime_scripts/ssec_new_data.json'))
dataset.target = json.load(open('onetime_scripts/ssec_new_label.json'))


f = []
def doc_feature(doc):
        doc_words = set(doc)
        feature = {}
        for word in most_common_word:
            feature[word] = (word in doc_words)
        return feature

for i in range(8):
    documents = []
    for text, target in zip(dataset.get_data(), dataset.get_target()):
        documents.append((text, target[i]))
    random.shuffle(documents)

    all_tokens = [element for lis in dataset.get_data() for element in lis]
    Freq_dist_nltk = nltk.FreqDist(all_tokens)
    most_common_word = [word for (word, _) in Freq_dist_nltk.most_common(8000)]


    train_set = nltk.apply_features(doc_feature, documents)
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    classifier.show_most_informative_features(n=10)
    features = classifier.most_informative_features(n=8000)

    # print(features)
    features_count = []
    for word in features:
        w = word[0]
        true_count = 0
        false_count = 0
        for doc in documents:
            word_set = set(doc[0])
            if w in word_set:
                false_count += 1
                true_count += doc[1]
        false_count -= true_count
        features_count.append((w, [true_count, false_count]))
    f += features_count

json.dump(f, open('/mount/arbeitsdaten31/studenten1/dialog-system/2020/student_directories/Changxu_Duan/thesis/RLANS/data_collection/py_ssec/ssec_feature_count.json', 'w'), indent=2)