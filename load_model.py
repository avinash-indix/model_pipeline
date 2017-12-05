from sklearn.externals import joblib
from nb_model_brand import topK
import time

model_directory_path = "/home/indix/search/bestseller/model/model_pkl/"
model_file = 'nb_100_2017_9_26_12_46_49.pkl'
complete_path = model_directory_path+model_file

print "Loading model..."
start = time.time()
nb_clf_object = joblib.load(complete_path)
stop = time.time()
print "Done loading model in \t"+ str(stop-start)


#predictions
new_query = ['neem face wash cream','lenovo a6600','samsung galaxy note 5','apple iphone 6s','samsung galaxy note case']
# print count_vect.get_feature_names()

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
count_vect = nb_clf_object['vectorizer']
tf_transformer = nb_clf_object['tf_transformer']
clf = nb_clf_object['model']

X_new_counts = count_vect.transform(new_query)
X_new_tf = tf_transformer.transform(X_new_counts)

print topK(clf,X_new_tf,k=3,query=new_query)
