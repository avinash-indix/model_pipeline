import pandas as pd
import nltk
from nltk.stem import PorterStemmer
import re
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import datetime
from inputs import accessories_keywords
from inputs import data_size,prepare_data

# full_data_path = "/home/indix/search/bestseller/model/in_search_with_brand_and_normalizedConfidence.csv"

# only data with vrabds and not stores
full_data_path = "/home/indix/search/bestseller/model/brands_in_search_with_brand_and_normalizedConfidence.csv"
processed_file_path = "/home/indix/search/bestseller/model/brands_clean_querytitle.csv"

# full_data_path  = "/home/indix/search/bestseller/model/in_search_small_dataset.csv"


def stem_token(ps, token):
    # simplest case ---> remove the 's' at the end of token
    if len(token) >= 3 and token[-1] == 's':
        return token[:-1]
    else:
        return token


def clean_feature(query_and_title, use=True):
    """

    :param query_and_title:
    :return: cleaned up query_and_title ----> lowercase + remove special characters and single character tokens and stem
    """
    query_and_title = str(query_and_title).lower()
    query_and_title = "".join(c for c in query_and_title if c.isalnum() or c.isspace())
    query_and_title = query_and_title.strip()

    # stem the token
    query_and_title = " ".join(map(lambda token: stem_token(ps, token), query_and_title.split()))

    # clean_query =  clean_feature(query_and_title)
    # print query_and_title

    # if case/cases, cover/covers appear with phone, iphone, ipad etc remove them
    clean_query_tokens = query_and_title.split()
    if not use:
        return " ".join(clean_query_tokens)

    def contains(query, product):
        if query.__contains__(product):
            return query.index(product)
        else:
            return -1

    # this is a heuristic
    # if position of "case" in query is i , then the product it accesories will be i-1
    accessory_index = filter(lambda index: index >= 1, map(lambda product: contains(clean_query_tokens, product)
                                                           , accessories_keywords))
    if len(accessory_index) > 0:

        # do not process accessories in this model
        return ""

        # drop the token as position i-1 from clean_query_tokens
        _accessory_index_ = accessory_index[0]
        if len(clean_query_tokens) > 1:
            # print clean_query_tokens[_accessory_index_]
            clean_query_tokens = [clean_query_tokens[_accessory_index_]]

    return " ".join(clean_query_tokens)
    # return query_and_title


def clean_features(features):
    """

    :param features: query_and_title list
    :return: cleaned up query_and_title
    """
    return map(lambda query_and_title: clean_feature(query_and_title), features)


def topK(clf, X, k=3, query=[]):
    predicted = clf.predict_proba(X)
    shape = predicted.shape
    print shape[0], len(query)
    # assert 1==2
    classes = clf.classes_
    topkclasses = []

    # get the top prob from predicted
    for i in range(predicted.shape[0]):
        # print i
        prediction = predicted[i]
        q = query[i]
        # for prediction in predicted:

        # index of top k classes
        topK_indices = prediction.argsort()[-k:]

        # flip the array so that the class with largest prob is at the beginning
        tmp = np.flip(classes[topK_indices], axis=0).tolist()
        topK_probs = np.flip(prediction[topK_indices], axis=0).tolist()
        class_and_prob = zip(tmp, topK_probs)
        topkclasses.append([q, class_and_prob])
    return topkclasses


def test_topK(array, k=3, classes=np.array(['zero', 'one', 'two', 'three', 'four', 'five', 'six'])):
    topK = []
    for row in array:
        topK_indices = row.argsort()[-k:]
        tmp = np.flip(classes[topK_indices], axis=0).tolist()
        topK.append(tmp)
    return topK


ps = PorterStemmer()
if __name__ == '__main__':
    start = time.time()
    full_data = pd.read_csv(full_data_path, sep='\t',nrows=data_size)
    print len(full_data)
    print full_data.columns

    # assert 1==2
    # our feature will be query + title
    features = full_data['query'] + " " + full_data['title']

    X = clean_features(features)
    Y = map(lambda brandname : clean_feature(brandname, False),full_data['brandname'])
    assert len(X) == len(features)
    data_size = len(X)

    # we need to filter/train on query + title data for non accessories
    # the accessory keywords are in accessories_keywords
    temp_df = pd.DataFrame({'query_title':X, 'brandName':Y})
    df = temp_df[temp_df['query_title'] != ""]

    if prepare_data == True:
        clean_df = df[['query_title','brandName']]
        clean_df.to_csv(processed_file_path, sep=",",index=False)
        exit(0);
    X_train = df['query_title']
    Y_train = map(lambda y : str(y),df['brandName'])


    # Extracting feature vectors
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    # print X_train_counts.shape


    # normalizing the counts (not using idf for the time being)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print X_train_tf.shape

    # Training a classifier

    clf = MultinomialNB(fit_prior=False).fit(X_train_tf, Y_train)
    stop = time.time()

    # predictions
    new_query = ['neem face wash cream', 'lenovo a6600', 'samsung galaxy note 5', 'apple iphone 6s',
                 'samsung galaxy note case']
    # print count_vect.get_feature_names()

    X_new_counts = count_vect.transform(new_query)
    X_new_tf = tf_transformer.transform(X_new_counts)

    array = np.array([[4, 5, 1, -2], [14, 5, 21, -2], [4, 5, 19, -2], [4, 5, 1, 28]])
    print test_topK(array)
    assert test_topK(array) == [['one', 'zero', 'two'], ['two', 'zero', 'one'], ['two', 'one', 'zero'],
                                ['three', 'one', 'zero']]

    # print predicted
    print "data size:\t", data_size
    print topK(clf, X_new_tf, k=3, query=new_query)
    print "Total time taken:\t", stop - start

    # saving the model
    model_type = 'nb_' + str(data_size) + '_'
    model_directory_path = "/home/indix/search/bestseller/model/model_pkl/"

    timestamp = datetime.datetime.now().timetuple()
    yr = timestamp.tm_year
    month = timestamp.tm_mon
    day = timestamp.tm_mday
    hour = timestamp.tm_hour
    minute = timestamp.tm_min
    sec = timestamp.tm_sec
    _time_ = str(yr) + '_' + str(month) + '_' + str(day) + '_' + str(hour) + '_' + str(minute) + '_' + str(sec)
    model_name = model_type + _time_ + '.pkl'
    complete_path = model_directory_path + model_name

    # model_object = {'vectorizer' :count_vect, 'tf_transformer':tf_transformer, 'model':clf}

    # i dont think we need to save the transformer
    model_object = {'vectorizer': count_vect, 'model': clf}
    joblib.dump(model_object, complete_path)
    print complete_path
