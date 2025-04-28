#Read the data and collect some statistics
import xml.etree.ElementTree as ET
import pandas as pd 
import sys
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import resample
from sacremoses import MosesDetokenizer
import os 
import re 
import shutil 
import stanza

detok = MosesDetokenizer(lang="en")

nlp = stanza.Pipeline("czech")

detok = MosesDetokenizer(lang="cz")
# Build path 
filename = "2014-czesl-sgt-en-all-v2.xml"

language_names = {
    'ru': 'Russian',
    'zh': 'Chinese',
    'uk': 'Ukrainian',
    'ko': 'Korean',
    'en': 'English',
    'ja': 'Japanese',
    'kk': 'Kazakh',
    'de': 'German',
    'fr': 'French',
    'es': 'Spanish',
    'vi': 'Vietnamese',
    'ar': 'Arabic',
    'pl': 'Polish',
    'tr': 'Turkish',
    'it': 'Italian',
    'mn': 'Mongolian',
    'uz': 'Uzbek',
    'ky': 'Kyrgyz',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'be': 'Belarusian',
    'th': 'Thai',
    'bg': 'Bulgarian',
    'az': 'Azerbaijani',
    'nl': 'Dutch',
    'fi': 'Finnish',
    'mk': 'Macedonian',
    'mo': 'Moldovan',
    'sq': 'Albanian',
    'el': 'Greek',
    'pt': 'Portuguese',
    'he': 'Hebrew',
    'fa': 'Persian',
    'sv': 'Swedish',
    'ba': 'Bashkir',
    'lv': 'Latvian',
    'da': 'Danish',
    'sk': 'Slovak',
    'tl': 'Tagalog',
    'ka': 'Georgian',
    'sr': 'Serbian',
    'hy': 'Armenian',
    'hr': 'Croatian',
    '': 'Unknown',
    'hi': 'Hindi',
    'xal': 'Kalmyk',
    'kg': 'Kongo',
    'sl': 'Slovenian',
    'id': 'Indonesian',
    'sh': 'Serbo-Croatian',
    'tg': 'Tajik',
    'no': 'Norwegian',
    'la': 'Latin',
    'ms': 'Malay'
}



def get_paths_czech_dataset():
    cur_file = os.path.basename(__file__)
    cur_dir = os.path.dirname(__file__)
    parent_dir =  os.path.split(cur_dir)[0]
    cz_file = os.path.join(parent_dir,"datasets","czesl",filename)
    return cz_file

cz_file = get_paths_czech_dataset()

class XML_Basic_Processing():
    def __init__(self, filename):
        self.filename = filename
        self.doc = ET.parse(self.filename).getroot()
        self.divs = self.doc.findall(".//div")
    def clean_up(self):
        # Keep only texts for which there is a declared CEFR level and language
        self.divs_clean =[i for i in self.divs if not i.get("s_l1")=="" and not i.get("s_cz_CEF")== ""]
    def text_limit(self, word_limit):
        # Keep only texts which go beyond a word limit 
        divs_words = []
        for i in self.divs_clean:
            if int(i.get("t_words_count")) > word_limit:
                divs_words.append(i)
        #Also pls rename
        self.div_words = divs_words
    def values_to_keep(self, dict_attribs):
        for i in dict_attribs.keys():
            attrib_values = dict_attribs[i]
            self.div_words = [el for el in self.div_words if el.get(i) in attrib_values]


def process_to_pandas(divs):
    users = [i.get("s_id") for i in divs]
    languages = [i.get("s_L1") for i in divs]
    word_count = [i.get("t_words_count") for i in divs]
    text_ids = [i.get("t_id") for i in divs]
    levels = [i.get("s_cz_CEF") for i in divs]
    assert len(users) == len(languages) == len(word_count)
    # This gets all words without sentence boundaries
    words_temp = [i.findall(".//word") for i in divs]
    sentences_from_users = []
    sentences_corrected = []
    upos_ = []
    lemmas_ = []
    errors = []
    for i in words_temp:
        temp_words = []
        temp_words_correct = []
        temp_errors = []
        temp_lemmas = []
        for j in i:
            temp_words.append(j.text)
            temp_words_correct.append(j.get("word1"))
            temp_lemmas.append(j.get("lemma1"))
            # see what the deal with appending a empty list is 
            err = j.get("err")
            if "|" in err:
                err = [i for i in err.split("|")]
            temp_errors.append(err)
        errors.append(temp_errors)
        lemmas_.append(temp_lemmas)
        sentences_from_users.append(temp_words)
        sentences_corrected.append(temp_words_correct)
    sentences_users_concat = [detok.detokenize(i) for i in sentences_from_users]
    sentences_corrected_concat = [detok.detokenize(i) for i in sentences_corrected]
    lemmas_concat =  [" ".join(i) for i in lemmas_]
    assert len(sentences_from_users) == len(sentences_users_concat)
    assert len(sentences_from_users)== len(users) == len(sentences_corrected) == len(errors)
    new_errors = []
    for i in errors:
        temp_err = []
        for j in i:
           
            if "str" in str(type(j)) :
                if j =="":
                    pass
                else:
                    temp_err.append(j)
            elif "list" in str(type(j)):
                for z in j:
                    temp_err.append(z) 
        new_errors.append(temp_err)
    
    dict_for_frame = {"text_id":text_ids,"user":users,"levels": levels ,"L1":languages, "word_count":word_count, "sentence":sentences_users_concat, "sentence_corrected": sentences_corrected_concat, "errors": new_errors, "lemmas":lemmas_concat}
    frame = pd.DataFrame.from_dict(dict_for_frame)
    temp_labels = pd.Series(frame["L1"])
    temp_labels = temp_labels.map(language_names)
    
    frame["L1"] = list(temp_labels)
    #print(frame["L1"])
    return frame





#Note: maybe base it users present in the dataframe 
def frame_for_errors(divs):
    #might need some adjustements here 
    
    words_temp = [i.findall(".//word") for i in divs]
    assert len(words_temp) == len(divs)
    error_data_stock = []
    for i in range(len(divs)):
        user = divs[i].get("s_id")
        language = divs[i].get("s_L1")
        level = divs[i].get("s_cz_CEF")
        text_id = divs[i].get("t_id")
        errors_indices =[words_temp[i].index(j) for j in words_temp[i] if j.get("err")!=""]
        if len(errors_indices) !=0:

            for j in errors_indices:
                errors = words_temp[i][j].get("err")
                if "|" in errors:
                    errors = [e for e in errors.split("|")]
                else:
                    errors = [errors]
            
                for err in errors:
                    stock = (text_id,user,language,level,err, words_temp[i][j].text, words_temp[i][j].get("word1"))
                    error_data_stock.append(stock)

    list_of_columns = ["text_id","user", "L1", "level","errors", "word", "correction"]
    errors_dataframe = pd.DataFrame.from_records(error_data_stock, columns=list_of_columns)
    return errors_dataframe

def apply_criteria_to_df(df, dict_of_interest):
    for i in dict_of_interest.keys():
        attributes_to_keep = dict_of_interest[i]
        df = df.loc[df[i].isin(attributes_to_keep)]
 
    return df 



def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], rotation = 45)
 

def check_languages(df): 
    langs = Counter(df["L1"].to_list())
    return langs 


def selective_downsampling(x, arg1):
    if len(x) > arg1:
        return x.sample(arg1)
    else:
        return x 



def downsample_df(df, error_df, remove_extra):
    # note this balances them by having the same number of rows
    second_most_popular= Counter(df["L1"].to_list()).most_common()[1][1]

    g = df.groupby("L1", group_keys=False)
    balanced_df = pd.DataFrame(g.apply(selective_downsampling,second_most_popular ))

    balanced_df_index = list(set(balanced_df["text_id"].to_list()))
 
    error_matrix_balanced = error_df[error_df['text_id'].isin(balanced_df_index)]
    set_intersection = set(balanced_df_index)- set(error_matrix_balanced["text_id"].to_list())

    if remove_extra == True:
        balanced_df = balanced_df[~balanced_df['text_id'].isin(set_intersection)]



    return balanced_df, error_matrix_balanced

#NOTE: get dictionnary from arguments parse instead
def create_df(select_dict):

    #Modify to retrieve file properly 
    xml_file = XML_Basic_Processing(cz_file)
    xml_file.clean_up()
    xml_file.text_limit(0)
    res_processing =process_to_pandas(xml_file.div_words)
    errors_frame = frame_for_errors(xml_file.div_words)
    #Modify this to get the parameters from the arguments
    df_with_less_langs = apply_criteria_to_df(res_processing, select_dict)
    balanced, err_df = downsample_df(df_with_less_langs, errors_frame, True)
    return balanced, err_df


xml_file = XML_Basic_Processing(cz_file)
xml_file.clean_up()
xml_file.text_limit(0)
res_processing =process_to_pandas(xml_file.div_words)


def preprocessing_function(txt):
    punct_removed =" ".join(re.findall(r"[\w]+", txt))
    nb_removed = re.sub(r"\d+", "", punct_removed)
    nb_removed = re.sub(' +', ' ', nb_removed)
    low_case = nb_removed.lower()
    return low_case

def preprocess(df, attribute):
    df[attribute] = df[attribute].apply(preprocessing_function)
    return df[attribute].to_list()

def get_nlp(df, attribute):
    preprocs = preprocess(df, attribute)
    sentences_upos = list()
    sentences_deprel = list()
    for text in preprocs:
        nlp_text = nlp(text)
        upos_sent =  " ".join([i.upos for i in nlp_text.iter_words()])
        deprel_sent =  " ".join([i.deprel for i in nlp_text.iter_words()])
        sentences_upos.append(upos_sent)
        sentences_deprel.append(deprel_sent)

    return sentences_deprel, sentences_upos

# Also add code to save the error dataframe ig
# Column = argument for whether the classification is done with corrected sentences or not  
def process_all_as_dict(select_dict, column):
    balanced_df, error = create_df(select_dict)
    sentences = preprocess(balanced_df,column)
    deprel, upos = get_nlp(balanced_df,column)

    return error, {"sentences":sentences, "labels":balanced_df["L1"].to_list(),"deprel":deprel,"upos":upos,"lemmas":balanced_df["lemmas"]}




