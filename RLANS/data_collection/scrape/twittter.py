import twint
import json

emotion = ['#anger']

for e in emotion:
    c = twint.Config()
    c.Search = e
    c.Lang = "en"
    c.Store_json = True
    c.Limit = 10
    c.Output = e+'1.json'
    c.Hide_output = True
    # c.Source = 'Twitter Web Client'

    twint.run.Search(c)


    fw = open(e+'_en.json', 'w', encoding='utf-8')
    with open(e+'.json', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d['language']=='en':
                fw.write(line)
    fw.close()