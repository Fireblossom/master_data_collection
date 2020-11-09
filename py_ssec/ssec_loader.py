import re
import html
from nltk.corpus import stopwords
remove_list = set(stopwords.words('english'))
import string
remove_list = remove_list.union(set(string.punctuation))
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
special_tokens_dict = {'additional_special_tokens': ['[smile]','[sad]','[laugh]','[neutral]', '[username]']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
remove_list.add('[CLS]')
remove_list.add('[SEP]')
remove_list.add('[BOS]')
remove_list.add('[EOS]')
remove_list.add('[PAD]')
remove_list.add('[MASK]')


class SsecDataset:
    def __init__(self,
                 text_data,
                 target,
                 tokenize=True):
        self.target = target
        self.tokenize = tokenize
        self.text_data = self.__cleaning(text_data)

    def __cleaning(self, text_list):
        """
            Cleaning the text
        """
        clean_text_list = []
        for text in text_list:
            text = text.replace('#SemST', '')
            text = re.sub(r"@[a-zA-Z0-9\_]*", '[username]', text)
            text = re.sub(r"((\(:)|(:\))|(\^_+\^)|;-\)|(;\)))", '[smile]', text)
            text = re.sub(r"((:\()|(:/))", '[sad]', text)
            text = re.sub(r":D", '[laugh]', text)
            text = re.sub(r"-_+-", '[neutral]', text)
            text = re.sub(r"(ha){2,}a*h*", 'haha', text)
            if self.tokenize:
                text = tokenizer.tokenize(text)
                text = [w for w in text if w not in remove_list]
            # print(text)
            clean_text_list.append(text)
        return clean_text_list

    def get_data(self):
        return self.text_data

    def get_target(self):
        return self.target


class SsecLoader:
    def __init__(self,
                 tokenize=True):
        self.tokenize = tokenize

    def load_ssec(self, s_tec_path):
        f_ssec = open(s_tec_path, 'r', encoding="utf8")
        text_data = []
        target = []
        for line in f_ssec:
            splited = html.unescape(line).split('\t')
            if len(splited) < 9:
                continue
            else:
                text_data.append(splited[-1])
                target.append(self.__label_process(splited[:-1]))
        f_ssec.close()

        return SsecDataset(
            text_data,
            target,
            self.tokenize
        )

    def __label_process(self, label_list):
        def binary_label(label):
            return 0 if label == '---' else 1
        return list(map(binary_label, label_list))


LABEL_MAPPING = {
    'Anger': 1,
    'Anticipation': 2,
    'Disgust': 3,
    'Fear': 4,
    'Joy': 5,
    'Sadness': 6,
    'Surprise': 7,
    'Trust': 8
}


if __name__ == "__main__":
    loader = SsecLoader(True)
    dataloader = loader.load_ssec('./ssec-aggregated/test-combined-0.0.csv')
    print(dataloader.get_data())