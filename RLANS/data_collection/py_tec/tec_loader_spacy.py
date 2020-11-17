import re
import html
from nltk.corpus import stopwords
remove_list = set(stopwords.words('english'))
import string
remove_list = remove_list.union(set(string.punctuation))
remove_list = remove_list.union({'“', '”', ' ', '-', "'m", "n't", "'s", "'ll", "'ve", "'re"})
import spacy
from spacy.symbols import ORTH

nlp = spacy.load("en_core_web_sm")
tokenizer = nlp.tokenizer
tokenizer.add_special_case("[smile]", [{ORTH: "[smile]"}])
tokenizer.add_special_case("[sad]", [{ORTH: "[sad]"}])
tokenizer.add_special_case("[laugh]", [{ORTH: "[laugh]"}])
tokenizer.add_special_case("[neutral]", [{ORTH: "[neutral]"}])
tokenizer.add_special_case("[username]", [{ORTH: "[username]"}])


from data_collection.py_tec.data_augmentation import DataAugment


class TecDataset:
    def __init__(self,
                 text_data,
                 target,
                 tokenize=True,
                 augment=False):
        self.target = target
        self.tokenize = tokenize
        self.text_data = self.__cleaning(text_data)
        if augment:
            extra = []
            extra_label = []
            da = DataAugment()
            for i in range(len(target)):
                if target[i] == 3:
                    extra.append(da.augment(text_data[i]))
                    print(extra[-1])
                    extra_label.append(3)
                if target[i] == 5:
                    for j in range(3):
                        extra.append(da.augment(text_data[i]))
                        print(extra[-1])
                        extra_label.append(5)
            self.target += extra_label
            self.text_data += self.__cleaning(extra)

    def __cleaning(self, text_list):
        """
            Cleaning the text
        """
        clean_text_list = []
        for text in text_list:
            text = re.sub(r"@[a-zA-Z0-9\_]*", '[username]', text)
            text = re.sub(r"((\^_+\^)|([:;]-?[\)pP])|(\(:))", '[smile]', text)
            text = re.sub(r"((:'?\()|(:/))", '[sad]', text)
            text = re.sub(r"((XD)|(:D)|(xd))", '[laugh]', text)
            text = re.sub(r"-_+-", '[neutral]', text)
            text = re.sub(r"(ha){2,}a*h*", 'ha', text)
            text = re.sub(r"\.{2,}", '..', text)
            text = text.lower()
            if self.tokenize:
                text = tokenizer(text)
                text = [w.text for w in text if w.text not in remove_list]
            # print(text)
            clean_text_list.append(text)
        return clean_text_list

    def get_data(self):
        return self.text_data

    def get_target(self):
        return self.target


class TecLoader:
    def __init__(self,
                 tokenize=True,
                 augment=False):
        self.tokenize = tokenize
        self.augment = augment

    def load_tec(self, s_tec_path):
        f_tec = open(s_tec_path, 'r', encoding="utf8")
        text_data = []
        target = []
        for line in f_tec:
            splited = html.unescape(line).split('\t')
            # print(splited)
            if len(splited) > 3:
                text_data.append(' '.join(splited[1:-1]))
            else:
                text_data.append(splited[1])
            # print(splited[2])
            target.append(LABEL_MAPPING[splited[-1]])
        f_tec.close()

        return TecDataset(
            text_data,
            target,
            self.tokenize,
            self.augment
        )


LABEL_MAPPING = {
    ':: joy\n': 1,
    ':: fear\n': 2,
    ':: anger\n': 3,
    ':: sadness\n': 4,
    ':: disgust\n': 5,
    ':: surprise\n': 6
}


if __name__ == "__main__":
    loader = TecLoader(True)
    # print(remove_list)
    dataset = loader.load_tec('./tec.txt')
    print(dataset.get_data())