import json
import copy
import pickle

#isear
counts = json.load(open('RLANS/data_collection/py_tec/tec_feature_count.json'))
vocab80 = set()
vocab60 = set()
vocab40 = set()
for word, count in counts:
        c = copy.deepcopy(count)
        c.sort(reverse=True)
        if (c[0]+c[1])/sum(c) >= 0.8:
                vocab80.add(word)
        if (c[0]+c[1])/sum(c) >= 0.6:
                vocab60.add(word)
        if (c[0]+c[1])/sum(c) >= 0.4:
                vocab40.add(word)

print(vocab80)
pickle.dump(vocab80, open('RLANS/data_collection/py_tec/vocab80.pkl', 'wb'))
pickle.dump(vocab60, open('RLANS/data_collection/py_tec/vocab60.pkl', 'wb'))
pickle.dump(vocab40, open('RLANS/data_collection/py_tec/vocab40.pkl', 'wb'))