from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
import os 
import json 
import pandas as pd 

stop_words_filename = "stopwords-cs.txt"
cur_dir = os.path.dirname(__file__)
def get_paths_czech_dataset():

    parent_dir =  os.path.split(cur_dir)[0]
    cz_file = os.path.join(parent_dir,stop_words_filename)
    with open(cz_file, "r", encoding="utf-8") as cz_stopwords:
        cs = cz_stopwords.read().splitlines()
    
    return cs 

temp_dir = os.path.split(cur_dir)[0]
out_dir = os.path.join(temp_dir,"out")

cs_stopwords = get_paths_czech_dataset()


param_grid = {
    "estimator__C": [1, 10, 100, 1000],
    "estimator__kernel": ["linear", "poly"],
    "estimator__gamma": [0.001, 0.0001],
    "estimator__degree":[1, 2]
    
}
param_grid = {
    "estimator__C": [1],
    "estimator__kernel": ["linear"],
    "estimator__gamma": [0.001],
    "estimator__degree":[1]
    
}

n_grams ={'unigram':(1,1),'bigram':(2,2),'trigram':(3,3)}
#n_grams ={'unigram':(1,1)}


class NGramAnalysertwo():
    def __init__(self, data, dataset_name= "BAWE"):
        self.dataset_name = dataset_name
        self.data = data 
        self.n_grams = n_grams
        
        self.label_encoder = LabelEncoder()
        self.labels = self.data["labels"]
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)
        self.svc = SVC()
        self.ensemble = OneVsOneClassifier(self.svc)
        self.vectorizer = CountVectorizer()
        self.vectorizers = {"Count":CountVectorizer(),"Tf-idf":TfidfVectorizer()}
        self.tf_idf_vectorizer = TfidfVectorizer()
        self.measures =  {"accuracy":accuracy_score,'f1_macro':f1_score, 'precision_macro': precision_score, 'recall_macro': recall_score}
        self.dummy_classifier = DummyClassifier(strategy="most_frequent")
        
    def n_gram_analysis(self, tf_idf= False,stopwords = False, variable ="sentences"):
        res_total = dict()
        self.tf_idf = tf_idf
        self.variable = variable
        self.stopwords = stopwords
        for i in list(n_grams.keys()):
            if tf_idf:
                vectorizer = self.vectorizers["Tf-idf"]
            else:
                vectorizer = self.vectorizers["Count"]
            if stopwords:
                if self.dataset_name=="BAWE":
                    vectorizer.stop_words="english"
                else:
                    vectorizer.stop_words= cs_stopwords
           
            vector =  vectorizer.fit_transform(self.data[variable])
            self.vectorizer.n_gram_range = n_grams[i]
            X_train, X_test, y_train, y_test = train_test_split(vector.toarray(), self.labels, test_size=0.15)
       
            sh = GridSearchCV(self.ensemble, param_grid=param_grid, scoring="accuracy")
               
            sh.fit(X_train, y_train)
            prediction = sh.predict(X_test)
            acc = round(accuracy_score(y_test, prediction),2)
            precision = round(precision_score(y_test, prediction,average = "weighted"),2)
            fone_score = round(f1_score(y_test, prediction, average = "weighted"),2)
            recall = round(recall_score(y_test, prediction, average = "weighted"),2)
            
            
            conf_matrix = confusion_matrix(y_test,prediction, labels = list(set(self.labels)))
            conf_matrix = pd.DataFrame(conf_matrix, index= list(set(self.labels)), columns= list(set(self.labels)))
            #conf_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, dis= list(set(self.labels)))

            results = {"dataset_name":self.dataset_name,'stopwords':str(stopwords),'tf-idf':str(tf_idf),"accuracy":acc, "precision":precision,'f1_score':fone_score, "recall":recall, "confusion_matrix":conf_matrix, "param_results":sh.best_params_}
            res_total[i] = results
        self.results = res_total
        return res_total

    def save_data(self):
        outfilename= self.variable +"_"+ self.dataset_name
        if self.tf_idf == True:
            outfilename= outfilename+"_tfidf"
        if self.stopwords == True:
            outfilename= outfilename+"_stopwords"
        outfilename = outfilename+ ".json"
        for i in list(n_grams.keys()):
            self.results[i]["confusion_matrix"] = self.results[i]["confusion_matrix"].to_dict()
        filename = os.path.join(out_dir, outfilename)
        with open(filename, 'w') as f:
            json.dump(self.results, f)

