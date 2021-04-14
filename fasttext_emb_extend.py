import data
import pickle

'''train_data = data.SSEC_DataSet('data_collection/ssec-aggregated/train-combined-0.33.csv') 
train_data.load(dictionary=Corpus_Dic)
print(len(Corpus_Dic.idx2word))
train_data = data.TEC_ISEAR_DataSet('isear')
train_data.load(dictionary=Corpus_Dic, train_mode=True, tokenize=True, addition='data_collection/scrape/blog_en.txt')
print(len(Corpus_Dic.idx2word))
train_data = data.TEC_ISEAR_DataSet('tec')
train_data.load(dictionary=Corpus_Dic, train_mode=True, tokenize=True, addition='data_collection/scrape/twitter_data.txt')
print(len(Corpus_Dic.idx2word))
pickle.dump(Corpus_Dic, open('dictionary.pkl', 'wb'))'''
dic_isear = pickle.load(open('isear/action_dictionary.pkl', 'rb'))
print(len(dic_isear.idx2word))
dic_tec = pickle.load(open('tec/action_dictionary.pkl', 'rb'))
print(len(dic_tec.idx2word))

voc = set()
with open('wiki-news-300d-1M-subword.vec.new', encoding='utf-8') as f:
    f.readline()
    for line in f:
        try:
            word, *vector = line.split()
            voc.add(word)
        except:
            if line == '\n':
                print(line)
oov = []
for word in set(dic_isear.idx2word+dic_tec.idx2word):
    if word not in voc:
        oov.append(word)

print(len(oov))

from gensim.models import FastText 
wv_model = FastText.load_fasttext_format('wiki-news-300d-1M-subword.bin')
with open('wiki-news-300d-1M-subword.vec', 'a', encoding='utf-8') as f:
    for word in oov:
        lst = [word]
        lst += list(map(str, [round(x, 4) for x in wv_model[word].tolist()]))
        f.write('\n'+' '.join(lst))

    