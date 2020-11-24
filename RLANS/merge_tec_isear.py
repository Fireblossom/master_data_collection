from data import TEC_ISEAR_DataSet


def get_merged_dataset(Corpus_Dic, tokenize, level):
    """
    1 JOY
    2 FEAR
    3 ANGER
    4 SADNESS
    5 DISGUST
    6 SHAME
    7 GUILT
    """
    train_isear = TEC_ISEAR_DataSet('isear')
    test_isear = TEC_ISEAR_DataSet('isear')
    train_isear.load(dictionary=Corpus_Dic, train_mode=True, tokenize=tokenize, level=level)
    test_isear.load(dictionary=Corpus_Dic, train_mode=False, tokenize=tokenize, level=level)
    """
    ':: joy': 1,
    ':: fear': 2,
    ':: anger': 3,
    ':: sadness': 4,
    ':: disgust': 5,
    ':: surprise': 6
    """
    train_tec = TEC_ISEAR_DataSet('tec')
    test_tec = TEC_ISEAR_DataSet('tec')
    train_tec.load(dictionary=Corpus_Dic, train_mode=True, tokenize=tokenize, level=level)
    test_tec.load(dictionary=Corpus_Dic, train_mode=False, tokenize=tokenize, level=level)

    train_data = TEC_ISEAR_DataSet('merge')
    test_data = TEC_ISEAR_DataSet('merge')

    train_data.max_length = max(train_isear.max_length, train_tec.max_length)
    train_data.length = train_isear.length + train_tec.length
    test_data.max_length = max(test_isear.max_length, test_tec.max_length)
    test_data.length = test_isear.length + test_tec.length

    train_data.tokens, train_data.labels = train_isear.tokens, train_isear.labels
    test_data.tokens, test_data.labels = test_isear.tokens, test_isear.labels

    for t, l in zip(train_tec.tokens, train_tec.labels):
        train_data.tokens.append(t)
        if l == 6:
            train_data.labels.append(8)
        else:
            train_data.labels.append(l)

    for t, l in zip(test_tec.tokens, test_tec.labels):
        test_data.tokens.append(t)
        if l == 6:
            test_data.labels.append(8)
        else:
            test_data.labels.append(l)
    print(test_data[100])

    return Corpus_Dic, train_data, test_data