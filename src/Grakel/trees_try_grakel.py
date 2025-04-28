import stanza 
#from graphs import Graph 
import numpy as np 
from graph import Graph 
from czech_try import balanced
import os 
from sklearn.preprocessing import LabelEncoder 
from sklearn.utils import Bunch
from nltk.tokenize import WhitespaceTokenizer 
import re 
import ast 
import random
import pandas as pd 
     
# Create a reference variable for Class WhitespaceTokenizer 
tk = WhitespaceTokenizer() 
     

#Try to get dependency relations with read_data from grakel

#Get id of all the nodes


#Get first 100 columns /nodes 

#new_sample_df = balanced.loc[balanced['L1'].isin(["zh", "ja"])]
new_sample_df = balanced





re_majuscule = r"(\b[A-Z]+)"

nlp = stanza.Pipeline('czech')


#nlp_list = get_nlp_as_list(id_node_try)

# Correct retry this time 
#Fix first phrase problm ?

def preprocess_doc(sent): 
    split_title = re.split(re_majuscule,sent)
    if len(split_title)<2: 
        sent_correct= sent
    else: 	
        split_title = split_title[0] + split_title[1] + split_title[2]
        sent_no_title = sent.replace(split_title,"")
        sent_correct = split_title.strip() + "." + " " +sent_no_title
    #sentences = sent_correct.split(".")
    #sentences = [sentence.strip() for sentence in sentences][0: len(sentences)-1]
    return sent_correct
    




##Lets check if there is a 0 key in there

###Read data class from Grakel 
labels_encoder = LabelEncoder()

def process_data():
    labels= []
    count_nodes = 1
    deprel_labels = dict()
    upos_labels= dict()
    label_ident = dict()
    
    Gs= []
    for id, row in new_sample_df.iterrows(): 
      
        sentences = preprocess_doc(row["sentence"])
        sentences_nlp = nlp(sentences)
        
    
        # all sentences in document
        for j in sentences_nlp.sentences: 
            
           
            labels.append(row["levels"])
            #lab= random.choice(rand_list)
            #labels.append(lab)
            #This is going to be one tree

           
            edges = set()
            nodes_dict = dict()
            edges_dict = dict()
          
            # all the nodes for one sentence = one graph
            relative_node_position = dict()
            for dep_tree in j.dependencies:
                head_node = dep_tree[0]
                rel_node = dep_tree[2]
                

                new_head_node_id = head_node.id + count_nodes
                new_rel_node_id = rel_node.id +count_nodes
                if head_node not in relative_node_position.keys(): 
                    relative_node_position[head_node.id]= new_head_node_id
                if rel_node not in relative_node_position.keys(): 
                    relative_node_position[rel_node.id]= new_rel_node_id
                
                new_head_node_id = relative_node_position[head_node.id] 
                new_rel_node_id=relative_node_position[rel_node.id] 
                edges.add((new_head_node_id, new_rel_node_id))
                edges.add((new_rel_node_id, new_head_node_id))
                # get labels 
                if head_node.deprel not in deprel_labels: 
                    deprel_labels[head_node.deprel] = len(deprel_labels)
                if rel_node.deprel not in deprel_labels:
                    deprel_labels[rel_node.deprel] = len(deprel_labels)

                edges_dict[((new_head_node_id, new_rel_node_id))] = deprel_labels[head_node.deprel]
                edges_dict[((new_rel_node_id, new_head_node_id))] = deprel_labels[rel_node.deprel]
                if new_rel_node_id not in nodes_dict:
                    if rel_node.upos not in upos_labels:
                        upos_labels[rel_node.upos] = len(upos_labels)
                    nodes_dict[new_rel_node_id] = upos_labels[rel_node.upos]
                if new_head_node_id not in nodes_dict:
                    if head_node.upos not in upos_labels:
                        upos_labels[head_node.upos] = len(upos_labels)
                    nodes_dict[new_head_node_id] = upos_labels[head_node.upos]
            
                
                count_nodes = count_nodes +1
            Gs.append((edges, nodes_dict))
        labels_encoded = labels_encoder.fit_transform(labels)
    return Bunch(data= Gs, target= labels_encoded)


