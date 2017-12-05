# http://www.gevent.org/intro.html#monkey-patching
import gevent.monkey
gevent.monkey.patch_all()

import os
import time
import logging
import sys
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from requestlogger import WSGILogger, ApacheFormatter
import bottle
from sklearn.externals import joblib
from bottle import request
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nb_model_brand import topK
from nb_model_brand import clean_features, clean_feature
from nb_acc_model_brand import clean_acc_features, clean_feature as clean_acc_feature

# from CalibrationTreeLearner import CalibrationTreeLearner

model_dir = "/home/indix/search/bestseller/model/model_pkl/"
model_file = model_dir + 'nb_50000_2017_10_4_13_52_52.pkl'
acc_model_file = model_dir + 'nb_acc50000_2017_10_4_13_53_13.pkl'
log_dir = os.getenv('LOG_DIR', "/tmp/")
fmt = logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('classification_service.' + __name__)

count_vect = None
tf_transformer = None
clf = None

X_new_counts = None
X_new_tf = None

is_models_loaded = False


try:

    try:
        logger.info("Loading models from %s" % model_file)
        classifier = joblib.load(model_file)
        count_vect = classifier['vectorizer']
        # tf_transformer = classifier['tf_transformer']
        clf = classifier['model']

        logger.info("Loading acc models from %s" % acc_model_file)
        acc_classifier = joblib.load(acc_model_file)
        acc_count_vect = acc_classifier['vectorizer']
        # tf_transformer = classifier['tf_transformer']
        acc_clf = acc_classifier['model']
        print "loaded accessory model from %s" %acc_model_file

        is_models_loaded = True

    except Exception as e:
        logger.exception(e)
        logger.error("failed loading model from %s" % model_file)
        raise

    # calibration_model = joblib.load(calibration_model_file)
    logger.info("Completed loading models")

except Exception as e:
    logger.exception("Initializing classifier service failed")
    sys.exit(1)



app = bottle.Bottle()


@app.route('/api/status')
def status():
    return {'status': 'online', 'servertime': time.time()}


@app.route('/api/echo/<text>')
def echo(text):
    return text


@app.route('/api/models')
def models():
    return {'model_loaded': is_models_loaded}


def to_unicode_or_bust(
        obj, encoding='utf-8'):
    if isinstance(obj, basestring):
        if not isinstance(obj, unicode):
            obj = unicode(obj, encoding, errors='replace')
            obj = obj.replace(u'\ufffd', ' ')
    return obj


@app.route('/api/classify')
def classify():
    try:
        response = None
        logger.info(request.query)
        query = to_unicode_or_bust(request.query.get('q'))
        top_K = to_unicode_or_bust(request.query.get('k'))

        query = query or u''
        clean_query = clean_feature(query)
        acc_query = clean_acc_feature(query)
        accessory = False
        top_K = int(top_K or '3')
        print query
        print top_K

        if len(acc_query)==0:
            """
            it is not an accessory query
            """
            X_new_counts = count_vect.transform([clean_query])
            tf_transformer = TfidfTransformer(use_idf=False).fit(X_new_counts)
            X_new_tf = tf_transformer.transform(X_new_counts)
            result = topK(clf,X_new_tf,k=top_K,query=[clean_query])
        else:

            accessory=True
            X_new_counts = acc_count_vect.transform([acc_query])
            tf_transformer = TfidfTransformer(use_idf=False).fit(X_new_counts)
            X_new_tf = tf_transformer.transform(X_new_counts)
            result = topK(acc_clf,X_new_tf,k=top_K,query=[acc_query])

        print result
        logger.info(result)
        # response = result.__dict__
        response = {'query':"", 'result_brands':{},'Accessory':accessory}
        response['query'] = result[0][0]
        print response

        for i in range(top_K):
            brand = result[0][1][i]
            prob = result[0][1][i]
            response['result_brands'][i] = {'brand': brand}


        return response
    except Exception as e:
        logger.error("error with query")
        logger.error(request.query.get('q'))
        raise


@app.route('/cam/classify', method='POST')
def classifyService():
    return classify()


# Setting up log handler at the end so that the routes are available
log_file = log_dir + '/access.log'
logging.info("Setting up server to use log dir @ " + log_file)
handlers = [TimedRotatingFileHandler(log_file, 'd', 3)]
app = WSGILogger(app, handlers, ApacheFormatter())
app.logger.propagate = False

if __name__ == '__main__':
    from bottle import run

    run(app, host='0.0.0.0', port=8080,server='gevent')
