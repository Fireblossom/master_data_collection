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


class TecDataset:
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
            text = re.sub(r"@[a-zA-Z0-9\_]*", '[username]', text)
            text = re.sub(r"((\(:)|(:\))|(\^_+\^)|;-\)|(;\)))", '[smile]', text)
            text = re.sub(r"((:\()|(:/))", '[sad]', text)
            text = re.sub(r"((XD)|(:D))", '[laugh]', text)
            text = re.sub(r"-_+-", '[neutral]', text)
            text = re.sub(r"(ha){2,}a*h*", 'ha', text)
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


class TecLoader:
    def __init__(self,
                 tokenize=True):
        self.tokenize = tokenize

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
            self.tokenize
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
    dataset = loader.load_tec('./tec.txt')
    print(dataset.get_data())