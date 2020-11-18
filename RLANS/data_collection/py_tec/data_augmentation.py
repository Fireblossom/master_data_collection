import textaugment, gensim, nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from textaugment import Wordnet


class DataAugment:
    def __init__(self):
        self.t = Wordnet(p = 0.8)

    def augment(self, sentence):
        return self.t.augment(sentence)

if __name__ == '__main__':
    da = DataAugment()
    print(da.augment('@mama I am angry.'))