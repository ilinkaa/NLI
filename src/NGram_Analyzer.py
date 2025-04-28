from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
import os 
import json 


param_grid = {
    "estimator__C": [1, 10, 100, 1000],
    "estimator__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    "estimator__gamma": [0.001, 0.0001],
    
}
#n_grams ={'unigram':(1,1),'bigram':(2,2),'trigram':(3,3)}
n_grams ={'unigram':(1,1), "bigram":(2,2), "trigram":(3,3)}

"""
cur_dir = os.path.dirname(__file__)
parent_dir =  os.path.split(cur_dir)[0]
results_folder = "results"
results_path = os.path.join(parent_dir, results_folder)
if not os.path.exists(results_path):
    os.makedirs(results_path)
"""




class NGramAnalyser():
    def __init__(self, data, stopwords_path = None, dataset_name= "BAWE"):
        self.dataset_name = dataset_name
        self.data = data 
        self.n_grams = n_grams
        self.stopwords_path = stopwords_path
        self.label_encoder = LabelEncoder()
        self.labels = self.data["labels"]
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        self.estimators = {"svc_linear": SVC(kernel='linear')}
        self.vectorizer = CountVectorizer()
        self.tf_idf_vectorizer = TfidfVectorizer()
        self.measures =  {"accuracy":accuracy_score,'f1_macro':f1_score, 'precision_macro': precision_score, 'recall_macro': recall_score}
        self.dummy_classifier = DummyClassifier(strategy="most_frequent")
        
    
    def n_gram_analysis(self, tf_idf = False, variable = "sentences"):
        best_model_stats = list()
        n_gram_results = dict()
        
        out_file_name = self.dataset_name
        for i in self.n_grams.keys():
            n_gram = self.n_grams[i]
            self.vectorizer.n_gram_range = n_gram
            print(self.vectorizer.n_gram_range)
            if self.stopwords_path != None:
                out_file_name =+ "_with_stopwords_"
                self.vectorizer.stop_words_ = self.get_stop_words()

            if tf_idf:
                vector = self.tf_idf.fit_transform(self.data[variable])
                out_file_name+= "_tf_idf_"
            else:
                print("icic plislifjlsdqfhqsjfldksjHFSKQ")
                vector = self.vectorizer.fit_transform(self.data[variable])
            X_train, X_test, y_train, y_test = train_test_split(vector, self.labels, test_size=0.15, stratify= self.labels)
            
            dict_estimator_results = dict()
            for estimator in self.estimators.keys():

                est = self.estimators[estimator]
                ensemble_classifier = OneVsOneClassifier(estimator = est)
                for score in self.measures.keys():
                    
                    print(ensemble_classifier)
                    print(param_grid)
                    print(score)
                    sh = GridSearchCV(ensemble_classifier, param_grid=param_grid, scoring=score).fit(X_train, y_train)
                   
                    print("got til here")
                    #prediction = sh.predict(X_test)
                    #print(prediction)
                    """
                    prediction = [str(i) for i in prediction]
                    dict_results = dict()
                    
                    #labels_not_predicted = self.label_encoder.inverse_transform((set(self.labels) - set(prediction))).to_list()
                    #dict_results["labels not predicted"] = list(self.label_encoder.inverse_transform(labels_not_predicted))
                    dict_results["labels predicted"] =  list(set(prediction))
                    dict_results["labels not predicted"] =  set(self.labels)-set(prediction)
                
                    #dict_r
                    #esults["confusion matrix"] = confusion_matrix(y_test,prediction, labels= list(set(self.labels)))
                 
                    dict_results["confusion matrix"] = confusion_matrix(y_test,prediction, labels = list(set(self.labels)))
                    dict_results["metrics"] = dict()
                    for metric in self.measures.keys():
                        if metric == "accuracy":

                            test_predict_result = self.measures[metric](y_test, prediction)
                        else: 
                            test_predict_result = self.measures[metric](y_test, prediction, average = "weighted")
                        dict_results["metrics"][metric] = test_predict_result
                if len(best_model_stats) != 0 and self.measures["accuracy"] > best_model_stats[-1]:
                    best_model_stats.pop()
                    best_model_stats.append()
                   
            dict_estimator_results[estimator] = dict_results
            n_gram_results[i] = dict_estimator_results
      
        outname = os.path.join(results_path, out_file_name)
        
        with open(outname, 'w') as f:
            json.dump(n_gram_results, f)
       
        return n_gram_results
"""