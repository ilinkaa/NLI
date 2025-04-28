#Check if BAWE Corpus is more usable 
import os 
import xml.etree.ElementTree as ET
from collections import Counter
import ast
import re 
import random
import argparse
import matplotlib as plt

import pandas as pd 
from conllu import serializer
from sklearn.utils import Bunch
import json
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity 
import shutil
from sacremoses import MosesDetokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np 
import conllu
from io import open
from conllu import parse_incr




detok = MosesDetokenizer(lang="en")

def get_paths_eng_dataset():
    cur_file = os.path.basename(__file__)
    cur_dir = os.path.dirname(__file__)
    parent_dir =  os.path.split(cur_dir)[0]
    bawe_conll_path = os.path.join(parent_dir,"datasets","BAWE_copy","BAWE_conll")
    bawe_xml_path = os.path.join(parent_dir,"datasets","BAWE_copy","CORPUS_ASCII")
    bawe_csv_path = os.path.join(parent_dir,"datasets","BAWE_copy","BAWE_csv")
    bawe_conll_files = [os.path.join(bawe_conll_path,i) for i in os.listdir(bawe_conll_path)]
    bawe_xml_files = [os.path.join(bawe_xml_path,i) for i in os.listdir(bawe_xml_path)]
    bawe_csv_files = [os.path.join(bawe_csv_path,i) for i in os.listdir(bawe_csv_path)]
    return bawe_xml_files, bawe_csv_path, bawe_conll_path
    
    
txt_f, csv_path, conll_path = get_paths_eng_dataset()


def files_per_lang(files): 
    dict_lang_file = dict()
    for i in files: 
        try:

            tree= ET.parse(i)
            root = tree.getroot()
            langs = [target.text for target in root.findall('.//p[@n="first language"]')]
            nb_words= [target.text for target in root.findall('.//p[@n="number of words"]')]
            
            if langs[0] == "Chinese Mandarin":
                langs[0] = "Chinese"
            if langs[0] not in dict_lang_file:
                dict_lang_file[langs[0]] = [i.replace(".xml", "")]
            else:
                dict_lang_file[langs[0]].append(i.replace(".xml", ""))
        
        except ET.ParseError: 
            print("Parse error")
    dict_lang_file[langs[0]] = i.replace(".xml","") 

    return dict_lang_file




files_dict= files_per_lang(txt_f)

def get_languages_count():
    dict_temp = dict()
    for i in files_dict.keys():
        dict_temp[i] = len(files_dict[i])
    ordered_dict_languages= {k: v for k, v in sorted(dict_temp.items(), reverse = True, key=lambda item: item[1])}
    # Maybe add some plotting in there 

get_languages_count()


def downsample_english():
    # Remove half of English dataset 
    english_temp = files_dict["English"]
    eng_files_nb = len(files_dict["English"])
    indices_to_remove = random.sample(range(0, eng_files_nb),39)
    for i in sorted(indices_to_remove, reverse=True):
        del english_temp[i]
    files_dict["English"] = english_temp
    return files_dict
    


    

def get_texts(langs_list):
    text_coll  = []
    labels = []
    upos_list = []
    dep_rel_list = []
    lemmas_list = []
    file_id = list()
    
    for lang in langs_list:
        files_to_get = files_dict[lang] 
       
        for file in files_to_get:
            filename = os.path.split(file)[-1]
            csv_path_file = os.path.join(csv_path,filename+'.csv')
            df = pd.read_csv(csv_path_file)
            file_id.append(filename)
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
            lemma = [str(i) for i in lemma]
            lemmas = detok.detokenize(lemma)
            deprel_tags = " ".join(deprel)
            sentences_tokenized = detok.detokenize(sentences)
            text_coll.append(sentences_tokenized)
            labels.append(lang)
            upos_list.append(upos_tags)
            dep_rel_list.append(deprel_tags)
            lemmas_list.append(lemmas)
          
    return file_id ,text_coll, labels , upos_list, dep_rel_list, lemmas_list


def get_texts_new(langs_list):
    return None

#get_texts(["French", "German",'Japanese'])

def save_files_per_language(files_list,labels):
    
    cur_dir = os.path.dirname(__file__)
    parent_dir =  os.path.split(cur_dir)[0]
    dir_files_for_bert= "saved_files"
    path_to_file_dir = os.path.join(parent_dir, dir_files_for_bert) 
    
    os.mkdir(path_to_file_dir) 
    path_to_label= os.path.join(dir_files_for_bert, "labels.txt") 
    
    for i in files_list:
        for j in files_list[i]:
            file_name = os.path.split(j)[-1]
            new_file = os.path.join(path_to_file_dir, file_name+".txt")
    
            if os.path.exists(file_name):
                shutil.copy2(new_file, path_to_file_dir) # target filename is /dst/dir/file.ext

            else:
                labels.remove(labels[i])
                print("File not found")
        with open(path_to_label,"w", encoding="utf-8") as labels_file:
            for lang in labels:
                labels_file.write(lang)
                labels_file.write("\n")
        print("files saved for BERT")

#Test for save function idk 




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
            conll_f = os.path.join(conll_path,os.path.split(file)[-1]+".txt.conll")
            if os.path.isfile(conll_f):
                total_files.append(file)
                with open(conll_f, "r", encoding="utf8") as temp_file:
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



vectorizer = CountVectorizer()
"""
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

"""

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
            tree = ET.parse(file)
            root = tree.getroot()
            course = [target.text for target in root.findall('.//p[@n="discipline"]')]
            # Clean name 
            course = re.sub(r"(BA | MA | MSc | Bsc)","",course[0])
            courses.append(course)
            pathname = csv_path+ file +".csv"
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



def get_all_data_from_query(query):
    files,texts, labels, upos_list, deprel_list, lemmas = get_texts(query)
    test_el = get_conllu_files(langs_list = query)
    #If some files werent available in conllu format, increment the list to be able to have it in a dataframe
    
    formatted_for_grakel= format_for_grakel(test_el[0], test_el[1])
    return {"files":files, "sentences":texts, "labels":labels, "upos":upos_list, "deprel": deprel_list, "lemmas":lemmas,"conllu":test_el,"grakel_data":formatted_for_grakel.data}

#get_all_data_from_query(["French", "German",'Japanese', 'Polish','Chinese'])

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Provisory data collection script for the BAWE dataset',
                    description='Sets up files',
                    epilog='Text at the bottom of help')
    parser.add_argument("--languages", type=str)
    args = parser.parse_args()[0]
    dict_data = get_all_data_from_query(args)
"""