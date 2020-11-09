emotion = ['joy', 'fear', 'anger', 'sad', 'disgust', 'surprise', 'trust', 'anticipation']

import twint
import json

for e in emotion:
    c = twint.Config()
    c.Search = e
    c.Lang = "en"
    c.Store_json = True
    c.Limit = 10000
    c.Output = e+'.json'
    c.Hide_output = True
    c.Source = 'Twitter Web Client'

    twint.run.Search(c)


    fw = open(e+'_en.json', 'w', encoding='utf-8')
    with open(e+'.json', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d['language']=='en':
                fw.write(line)
    fw.close()