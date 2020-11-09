from py_tec.tec_loader_spacy import TecLoader
loader = TecLoader(tokenize=True, augment=True)
dataset = loader.load_tec('./tec.txt')
import pickle
pickle.dump(dataset, open('tec_augment.pkl', 'wb'))
pickle.l