#from cz_data_pro_temp import *
import pandas as pd
from collections import Counter
from collections import defaultdict
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np 
from operator import itemgetter

#errors = process_all_as_dict({"L1":["ru","zh","ja","de",'fr']},"sentence")
errors_df = pd.read_csv("/Users/IlincaV/errorsdftest.csv")
#To be revised tho 
def get_vectors(errors_list = list(set(errors_df["errors"])) ,binary_bool = False ):
    data_counter = 0
    vectors_list = list()
    for i in set(errors_df["text_id"]): 
        d = defaultdict(list,{ k:0 for k in errors_list })
        rows = errors_df.loc[errors_df["text_id"] == i]
        
        for j in rows["errors"]:
            if j in d.keys():
                if binary_bool :
                    d[j] =1
                else:
                    d[j] +=1
        #print(len(d))
        if sum(d.values()) != 0:
            data_counter += 1
            vectors_list.append({"text_id":rows["text_id"],"label":rows["L1"].to_list()[0], "data":d})
    print(data_counter)
    return vectors_list

    

data_errors = get_vectors()
color_dict = {"ko":"blue", "de":"black", "en": "purple", "zh": "red", "fr": "brown", "ru": "green", "ja":"white"}

svc = SVC(kernel='linear')
model = OneVsOneClassifier(svc)


def array_format(data_errors):
    vectors = list()
    for i in data_errors:
        data = i
        data = data["data"].values()
        
        vectors.append(list(data))
    
    return vectors
labels = [data_errors[i]["label"] for i in range(len(data_errors))]

def svm_classify(vectors, labels):
    X_train, x_test, y_train, y_test = train_test_split(vectors, labels ,test_size=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred= model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(acc)
    confusion = confusion_matrix(y_test, y_pred, labels=list(set(labels)))
    sns.heatmap(confusion, annot=True, annot_kws={"size": 16}, xticklabels=set(labels), yticklabels= set(labels)) # font size
    plt.show()

    return confusion


vec_array= array_format(data_errors)
mutual_information = mutual_info_classif(vec_array, labels)


error_mapping = dict(zip(list(set(errors_df["errors"])),mutual_information))
error_mapping_ordered = dict(sorted(error_mapping.items(),reverse=True, key=lambda item: item[1]))

def downsample(data):
    labels_counter = Counter([i["label"] for i in data])
    labels_keys = list(labels_counter.keys())
    min_key, min_count = min(labels_counter.items(), key=itemgetter(1))
    new_list = list()
    for i in labels_keys:
        data_lang = [j for j in data if j["label"]==i]
        
        new_list.append(data_lang[0:min_count-1])
    return data_lang



        



def select_n_errors(n):
    print(error_mapping_ordered)
    sorted_keys = list(error_mapping_ordered.keys())[:n]
    print(sorted_keys)
    selected_errors = {key: value for key, value in zip(error_mapping.keys(), error_mapping.values()) if key in sorted_keys}

    return selected_errors

selected_errors = select_n_errors(2)

vectors_new = get_vectors(errors_list=list(selected_errors.keys()))
labels = [vectors_new[i]["label"] for i in range(len(vectors_new))]
vectors = array_format(vectors_new)
#labels = [vectors_new[i]["label"] for i in range(len(downsampled))]
svm_classify(vectors, labels)



