from langdetect import detect
f = open('tec.txt', 'w', encoding="utf8")
with open('Jan9-2012-tweets-clean.txt', 'r', encoding="utf8") as file:
    for line in file:
        if detect(line.split('\t')[1]) == 'en':
            f.write(line)
f.close()