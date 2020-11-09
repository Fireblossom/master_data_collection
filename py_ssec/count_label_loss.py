from ssec_loader_spacy import SsecLoader


def count_no_emotion(dataloader):
    count = 0
    for target in dataloader.get_target():
        if target == (0, 0, 0, 0, 0, 0, 0, 0):
            count += 1
    return count


loader = SsecLoader(True)
dataloader0 = loader.load_ssec('./ssec-aggregated/train-combined-0.0.csv')
dataloader3 = loader.load_ssec('./ssec-aggregated/train-combined-0.33.csv')
dataloader5 = loader.load_ssec('./ssec-aggregated/train-combined-0.5.csv')
dataloader6 = loader.load_ssec('./ssec-aggregated/train-combined-0.66.csv')
dataloader9 = loader.load_ssec('./ssec-aggregated/train-combined-0.99.csv')
print('total:',len(dataloader0.get_target()))
print(count_no_emotion(dataloader0), round(count_no_emotion(dataloader0) / len(dataloader0.get_target()), 2), '%')
print(count_no_emotion(dataloader3), round(count_no_emotion(dataloader3) / len(dataloader0.get_target()), 2), '%')
print(count_no_emotion(dataloader5), round(count_no_emotion(dataloader5) / len(dataloader0.get_target()), 2), '%')
print(count_no_emotion(dataloader6), round(count_no_emotion(dataloader6) / len(dataloader0.get_target()), 2), '%')
print(count_no_emotion(dataloader9), round(count_no_emotion(dataloader9) / len(dataloader0.get_target()), 2), '%')