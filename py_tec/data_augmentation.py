import textaugment, gensim, nltk
from textaugment import Word2vec


class DataAugment:
    def __init__(self):
        model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin.gz', binary=True)
        self.t = Word2vec(model=model)

    def augment(self, sentence):
        return self.t.augment(sentence)

if __name__ == '__main__':
    da = DataAugment()
    print(da.augment('@mama I am angry.'))