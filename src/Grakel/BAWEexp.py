#Check if BAWE Corpus is more usable 
import os 
import xml.etree.ElementTree as ET
from collections import Counter
import ast
import re 
import conllu.exceptions
import pandas as pd 
from conllu import serializer
from sklearn.utils import Bunch
import json
from sklearn.metrics.pairwise import cosine_similarity 
import shutil
from sacremoses import MosesDetokenizer
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import conllu
from io import open
from conllu import parse_incr


#Need to change this
path_to_files =  "/Users/IlincaV/Documents/DataScienceProject/BAWE/CORPUS_ASCII/"
path_to_txt =  "/Users/IlincaV/Documents/DataScienceProject/BAWE/CORPUS_TXT/"
path_to_csvs = "/Users/IlincaV/Documents/DataScienceProject/BAWE/csv/BAWE_csv/"
path_to_conllu = "/Users/IlincaV/Documents/DataScienceProject/BAWE/conll/BAWE_conll/"

files = os.listdir(path_to_files)
detok = MosesDetokenizer(lang="en")
def files_per_lang(files): 
    dict_lang_file = dict()
    for i in files: 
        path_file = path_to_files + i
        try:

            tree= ET.parse(path_file)
            root = tree.getroot()
            langs = [target.text for target in root.findall('.//p[@n="first language"]')]
            if langs[0] not in dict_lang_file:
                dict_lang_file[langs[0]] = [i.replace(".xml", "")]
            else:
                dict_lang_file[langs[0]].append(i.replace(".xml", ""))
        
        except ET.ParseError: 
            print("Parse error")
    dict_lang_file[langs[0]] = i.replace(".xml","") 

    return dict_lang_file

files_dict = files_per_lang(files)

def get_texts(langs_list):
    text_coll  = []
    labels = []
    upos_list = []
    dep_rel_list = []
    lemmas = []
    file_id = list()
   
    for lang in langs_list:
        files_to_get = files_dict[lang] 
        for file in files_to_get:
            
            pathname = path_to_csvs+ file +".csv"
            df = pd.read_csv(pathname)
            file_id.append(file)
            sentences = df["word"].to_list()
            upos = df["pos"].to_list()
            deprel = df["dep"].to_list()
            lemma = df["lemma"].to_list()
            # Remove references
            if "-LRB-" in sentences and "-RRB-" in sentences:
                begin_ref= sentences.index("-LRB-")
                end_ref = sentences.index("-RRB-")
                sentences = sentences[0: begin_ref] + sentences[end_ref:len(sentences)]
          
                upos = upos[0: begin_ref] + upos[end_ref:len(sentences)]
                lemma = lemma[0: begin_ref] + lemma[end_ref:len(sentences)]
                deprel = deprel[0: begin_ref] + deprel[end_ref:len(sentences)]
            sentences = [str(i) for i in sentences]
            upos_tags = " ".join(upos)
            deprel_tags = " ".join(deprel)
            sentences_tokenized = detok.detokenize(sentences)
            text_coll.append(sentences_tokenized)
            labels.append(lang)
            upos_list.append(upos_tags)
            dep_rel_list.append(deprel_tags)
            lemmas.append(lemma)
          
    return file_id ,text_coll, labels , upos_list, dep_rel_list, lemmas

files,texts, labels, upos_list, deprel_list, lemmas = get_texts(["Japanese","French","Chinese Mandarin","Polish"])
def save_files_per_language(files_list,labels):
    new_dir = "/Users/IlincaV/Documents/data_langs"
    for i in files_list:
        file_name= path_to_txt+ i + ".txt"
        if os.path.exists(file_name):
            shutil.copy2(file_name, new_dir) # target filename is /dst/dir/file.ext

        else:
            labels.remove(labels[i])
            print("File not found")
    with open(new_dir+"/"+"labels.txt", "w") as labels_file:
        for lang in labels:
            labels_file.write(lang)
            labels_file.write("\n")
    print("files saved for BERT")


        

#For later processing with BERT, moves them to another file





def get_conllu_files(langs_list):
    dict_labels_and_files = dict()
    labels = list()
    deprel_indices = list()
    deprels = list()
    deprel_indices.append(0)
    total_files = list()

    for lang in langs_list:
        files_to_get = files_dict[lang] 
        for file in files_to_get:
            pathname = path_to_conllu+ file +".txt.conll"
            if os.path.isfile(pathname):
                total_files.append(file)
                with open(pathname, "r", encoding="utf8") as temp_file:
                    data = temp_file.read().split("\n\n")
                    #each dat = one sentence graph 
                    for dat in data[0:len(data)-1]:
                        dat_stock = list()
                        dat_split = dat.split("\n")
                        for i in dat_split:
                            dat_line = i.split("\t")
                            dat_line[4] = "_"
                            dat_line.insert(5,"_")
                            dat_line.append("_")
                            dat_line.append("_")
                            dat_line= "\t".join(dat_line)
                            dat_stock.append(dat_line)
                        sent_deprel = "\n".join(dat_stock)
                        deprels.append(conllu.parse(sent_deprel))
                        labels.append(lang)

    return (deprels, labels)

         
    
test_el = get_conllu_files(langs_list = ["Japanese", "Polish","French", "Greek"])

def format_for_grakel(deprels, labels):
  
    count_nodes = 1
    deprel_labels = dict()
    upos_labels= dict()
    label_encoder= LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    Gs= []
    #one sent
    for deprel in deprels:
        edges = set()
        nodes_dict = dict()
        edges_dict = dict()
        #token list format
        for tok in deprel:
            # different tokens
            for i in tok:
                if i["upos"] not in upos_labels.keys():
                        upos_labels[i["upos"]] = len(upos_labels)
                node_id = count_nodes
                if i["head"] != 0:
                    if i["deprel"] not in deprel_labels.keys():
                        deprel_labels[i["deprel"]] = len(deprel_labels)
                    
                    new_head_id = (count_nodes-i["id"]) + i["head"]
                    edges.add((new_head_id,node_id))
                    edges_dict[(new_head_id, node_id)] = deprel_labels[i["deprel"]]
                    nodes_dict[node_id] = upos_labels[i["upos"]]
                else:
                    nodes_dict[node_id] = upos_labels[i["upos"]]
                    

                    

               
                count_nodes = count_nodes+1
                
           
        Gs.append([edges, nodes_dict, edges_dict])
    
         
    return Bunch(data= Gs[1:len(Gs)], target= labels_encoded[1:len(labels_encoded)])



formatted_for_grakel= format_for_grakel(test_el[0], test_el[1])
vectorizer = CountVectorizer()
def cosine_similarity_langs():
    dict_cosine_results = dict()
    dict_results_final = dict()
    indexes = []
    labels_ind = dict(zip([i for i in range(0, len(labels))], labels))
    for i in list(set(labels)):
        labels_ind_group = list()
        for j in labels_ind.keys():
           
            if labels_ind[j] == i :
                labels_ind_group.append(j)
        indexes.append(labels_ind_group)
    dict_index_labels= dict(zip(list(set(labels)),indexes))
    labels_keys_copy = list(dict_index_labels.keys())
    vectors = vectorizer.fit_transform(texts)
    vectors = vectors.toarray()
    for i in dict_index_labels.keys():
        select_ind_lang = np.take(vectors, dict_index_labels[i], 0)
     
        labels_keys_copy.remove(i)
        for j in labels_keys_copy:
            select_ind_lang_two = np.take(vectors, dict_index_labels[j], 0)
            dict_cosine_results[i+"-"+j] = cosine_similarity(select_ind_lang, select_ind_lang_two)
    for res in dict_cosine_results.keys():
        dict_results_final[res] = np.mean(dict_cosine_results[res])
    
    return dict_results_final

    
cosine_results = cosine_similarity_langs()


#print(formatted_for_grakel.data[0])
# Per language, get average nb of sentence, avg length of sentence and average nb of words
# Get nb of students

def get_corpus_statistics(langs_list):
 
    dict_langs_statistics = dict()
    for lang in langs_list:
        files_to_get = files_dict[lang] 
        avg_sent_length = list()
        avg_sentences = list()
        dict_langs_statistics[lang] = dict()
        courses = list()
        for file in files_to_get:
            #Get Field
            pathname_xml = path_to_files + file + ".xml" 
            tree = ET.parse(pathname_xml)
            root = tree.getroot()
            course = [target.text for target in root.findall('.//p[@n="discipline"]')]
            # Clean name 
            course = re.sub(r"(BA | MA | MSc | Bsc)","",course[0])
            courses.append(course)
            pathname = path_to_csvs+ file +".csv"
            df = pd.read_csv(pathname)
            sentences = df["word"].to_list()
            # Remove references
            if "-LRB-" in sentences and "-RRB-" in sentences:
                begin_ref= sentences.index("-LRB-")
                end_ref = sentences.index("-RRB-")
                sentences = sentences[0: begin_ref] + sentences[end_ref:len(sentences)]
            sentence_length = len([str(i) for i in sentences])
            sentence_nb = df["sentence_number"].to_list()[-1]
            avg_sent_length.append(sentence_length)
            avg_sentences.append(sentence_nb)
        dict_langs_statistics[lang]["sentence_length"] = np.mean(avg_sent_length)
        dict_langs_statistics[lang]["number of sentences"] = np.mean(avg_sentences)
        dict_langs_statistics[lang]["courses"] = dict(Counter(courses))
        nb_students = len(set([re.sub(r"[a-z]", '', i) for i in files_to_get]))
        dict_langs_statistics[lang]["number of students"] = nb_students
     
    return dict_langs_statistics





def get_corpus_statistics_df(stats_dict):
    dict_reshaped= list()
    for i in stats_dict.keys():
        temp_dict_lang = {"Language":i}
        temp = stats_dict[i]
        temp.update(temp_dict_lang)
        for course in stats_dict[i]["courses"]:
            temp[course]= stats_dict[i]["courses"][course]
        del temp["courses"]
        dict_reshaped.append(temp)
    df_stats = pd.DataFrame(dict_reshaped)
    df_stats = df_stats.fillna(0)
    column_to_move = df_stats.pop("Language")

# insert column with insert(location, column_name, column_value)

    df_stats.insert(0, "Language", column_to_move)
    return df_stats





#get_corpus_statistics_df(corpus_stats)

param_grid = {'max_depth': [None,30,32,35,37,38,39,40],'min_samples_split': [2,150,170,180,190,200]}

class WordAnalyzerBAWE():
    def __init__(self, sentences= texts,deprel = deprel_list, upos_tags = upos_list, labels=labels, n_grams ={'unigram':(1,1),'bigram':(2,2),'trigram':(3,3)}):
        self.deprel = deprel
        self.upos = upos_tags
        #Fix this bc idk how 
        self.label_encoder = LabelEncoder()
        self.labels = labels
        self.labels_encoded = self.label_encoder.fit_transform(labels)
        self.sentences = sentences
        self.data = {"upos": self.upos, "words": self.sentences, "deprel": deprel_list}
        self.n_grams = n_grams
        #Add more classifiers (SVM)
        self.estimators = {"random-forest": RandomForestClassifier(),"svc_linear": OneVsOneClassifier(SVC(kernel="linear"))}
        
        self.vectorizer = CountVectorizer(stop_words="english")
        self.tf_idf = TfidfVectorizer(stop_words="english")
        self.measures =  {"accuracy":accuracy_score,'f1_macro':f1_score, 'precision_macro': precision_score, 'recall_macro': recall_score}
        self.dummy_classifier = DummyClassifier(strategy="most_frequent")
            
    
    def n_gram_analysis(self, tf_idf = False, category= "words"):
        data = self.data[category]
        n_gram_results = dict()
        for i in self.n_grams.keys():
            
            n_gram = self.n_grams[i]
            if tf_idf:
                self.tf_idf.n_gram_range = n_gram 
                vector = self.tf_idf.fit_transform(data)
            else:
                self.vectorizer.n_gram_range = n_gram
                vector = self.vectorizer.fit_transform(data)
            self.dummy_classifier.fit(vector, self.labels_encoded)
            self.dummy_classifier.predict(vector)

            score_dummy = self.dummy_classifier.score(vector, self.labels_encoded)
            
            X_train, X_test, y_train, y_test = train_test_split(vector, self.labels, test_size=0.15)
            dict_estimator_results = dict()
            dict_estimator_results["dummy classifier accuracy"] = score_dummy
            for estimator in self.estimators.keys():
                
                est = self.estimators[estimator]
                for score in self.measures.keys():
                    sh = HalvingGridSearchCV(est, param_grid=param_grid,cv=5, scoring=score,random_state=42,
                                      factor=2).fit(X_train, y_train)
                    best_est = sh.best_estimator_
                    best_est.fit(X_train, y_train)

                    prediction = best_est.predict(X_test)                 
                    labels_not_predicted = set(self.labels_encoded) - set(prediction)
                    
                    dict_results = dict()
                    dict_results["labels not predicted"] = list(labels_not_predicted)
                    dict_results["labels predicted"] =  list(set(prediction))
                    
                    dict_results["confusion matrix"] = list(confusion_matrix(y_test,prediction, labels= list(set(self.labels))))
                    dict_results["confusion matrix labels"]= self.labels
                    for metric in self.measures.keys():
                        if metric == "accuracy":        

                            test_predict_result = str(self.measures[metric](y_test, prediction))
                        elif metric =="f1":
                            test_predict_result = str(self.measures[metric](y_test, prediction, average = "weighted", labels = np.unique(prediction)))
                        else:
                            test_predict_result = str(self.measures[metric](y_test, prediction, average = "macro", labels = np.unique(prediction)))
                        dict_results[metric] = test_predict_result
                    
             
                dict_estimator_results[estimator] = dict_results
            n_gram_results[str(n_gram)] = dict_estimator_results
            print(n_gram_results)
        # Transform into df 
        #Add stuff for filename
        # Fix json problem
        
        if tf_idf:
            outname = "tf_n_grams_results1.json"
        else:
            outname = "n_grams1.json"
        with open(outname, 'w') as f:
            json.dump(n_gram_results, f)
        return n_gram_results
        
