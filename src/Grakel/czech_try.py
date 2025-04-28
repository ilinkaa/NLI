#Read the data and collect some statistics
import xml.etree.ElementTree as ET
import pandas as pd 
import sys
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import resample
from sacremoses import MosesDetokenizer
import os 
import shutil 

detok = MosesDetokenizer(lang="cz")

folder_name = "datasets_czech"


file_path = "/Users/IlincaV/Documents/Cz_project/2014-czesl-sgt-en-all-v2.xml"


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


xml_file = XML_Basic_Processing(file_path)
xml_file.clean_up()
xml_file.text_limit(0)

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
    return frame

res_processing =process_to_pandas(xml_file.div_words)


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


errors_frame = frame_for_errors(xml_file.div_words)

def apply_criteria_to_df(df, dict_of_interest):
    for i in dict_of_interest.keys():
        attributes_to_keep = dict_of_interest[i]
        df = df.loc[df[i].isin(attributes_to_keep)]
 
    return df 
def check_languages(df): 
    langs = Counter(df["L1"].to_list())
 
    return langs 
df_with_less_langs = apply_criteria_to_df(res_processing, {"levels":["A1","A2","B1", "B2", "C1"], "L1": ["ko","ru", "zh", "ja", "de", "en", "fr"]})



def select_elements(df, other_df):
    indexes = df.index
    other_df = other_df.loc[indexes]
    return df, other_df

more_languages_df , new_error_df = select_elements(df_with_less_langs, errors_frame)
 


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i], rotation = 45)
 

def check_languages(df): 
    langs = Counter(df["L1"].to_list())
    print(langs)
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
balanced, err_df = downsample_df(df_with_less_langs, errors_frame, True)

#NOTE: get dictionnary from arguments parse instead
def create_df(select_dict):

    #Modify to retrieve file properly 
    xml_file = XML_Basic_Processing(file_path)
    xml_file.clean_up()
    xml_file.text_limit(0)
    res_processing =process_to_pandas(xml_file.div_words)
    errors_frame = frame_for_errors(xml_file.div_words)
    #Modify this to get the parameters from the arguments
    df_with_less_langs = apply_criteria_to_df(res_processing, select_dict)
    balanced, err_df = downsample_df(df_with_less_langs, errors_frame, True)
    return balanced, err_df






def save_new_dfs(*args):
    # Change this cause that's kinda awkward 
    filenames = ["balanced_df.csv", "error_df.csv"]

    cwd= sys.path[0]

    path_to_folder = os.path.join(cwd, folder_name)
    if os.path.exists(path_to_folder):
        shutil.rmtree(path_to_folder)
    os.makedirs(path_to_folder)
    for i,j in zip(filenames, args):
        path_to_file = os.path.join(path_to_folder, i)
     
        j.to_csv(path_to_file)
    print("files saved")
 

save_new_dfs(balanced,err_df)



class StatisticalOverview():
    def __init__(self, dataframe, error_df):
        self.dataframe = dataframe
        self.columns = self.dataframe.columns
        self.languages = list(set(self.dataframe["L1"].to_list()))
        self.levels = self.dataframe["levels"].to_list()
        self.word_counts = self.dataframe["word_count"].to_list()
        self.error_df = error_df
        self.errors = error_df["errors"]
    # Plot some statistics in regards to languages / level / word counts

    def plot_languages_text(self):
        counter_languages = dict(Counter(self.languages))
        plt.bar(counter_languages.keys(), counter_languages.values(), color ='maroon', 
        width = 0.4)
        plt.xlabel("L1")
        plt.ylabel("Number of texts")
        plt.title("Number of texts available per language")
        plt.show()
    def plot_levels_text(self):
        counter_levels = dict(Counter(self.levels))
        plt.bar(counter_levels.keys(), counter_levels.values(), color ='maroon', 
        width = 0.4)
        addlabels(counter_levels.keys(), list(counter_levels.values()))
        plt.xlabel("Level")
        plt.ylabel("Number of texts")
        plt.title("Number of texts available per level")
        plt.show()
    def plot_most_common_level_per_language(self):
        most_common_level_per_lang = []
        for i in self.languages:
            temp_lang_df = self.dataframe[self.dataframe["L1"] == i]
            most_common_level =Counter(temp_lang_df["levels"].to_list()).most_common(1)[0][0]
            most_common_level_per_lang.append(most_common_level)
        dict_res = dict(zip(self.dataframe["L1"].to_list()), most_common_level_per_lang)
        # Still need to check how to display this 
            
    def given_lvl_per_language(self, level):
        query_df = self.dataframe.loc[self.dataframe["levels"] == level]
        counter_texts = dict(Counter(query_df["L1"].to_list()))
        plt.bar(counter_texts.keys(), counter_texts.values(), color ='maroon', 
        width = 0.4)
        plt.xlabel("Languages")
        plt.ylabel("Number of "+ level + "level texts")
        plt.title("Number of "+ level + "level texts per language")
        plt.show()
    def average_text_length_per_language(self):
        average_txt_length_per_lang = []
        for i in self.dataframe["L1"]:
            temp_lang_df = self.dataframe[self.dataframe["L1"] == i]
            avg_word_count =temp_lang_df["word_count"].to_list()
            temp_int_list  = list(map(int, avg_word_count))
            temp_avg = sum(temp_int_list)/len(temp_int_list)
            average_txt_length_per_lang.append(temp_avg)

        #Maybe re write this too bc it takes too long 
        average_txt_length_per_lang = [round(i, 0) for i in average_txt_length_per_lang]
        dict_plot = dict(zip(self.dataframe["L1"].to_list(), average_txt_length_per_lang))
        plt.bar(dict_plot.keys(), dict_plot.values(), color ='maroon', 
            width = 0.4)
        plt.xlabel("Languages")
        plt.ylabel("Average length of texts")
        plt.title("Average length of a text per language")
        plt.show()
        
    def errors_frequency(self):
        error_counter = dict(Counter(self.errors).most_common())
    
        plt.bar(error_counter.keys(), error_counter.values(), color ='maroon', 
            width = 0.2)
        addlabels(error_counter.keys(), list(error_counter.values()))
        plt.xlabel("Error")
        plt.xticks(rotation=90, fontsize = 'xx-small')


        plt.ylabel("Number of errors")
        plt.title("General distribution of errors")
        plt.show()     


    #Fix formatting or find a solution here 
    def most_common_error_per_language(self, languages_list):
        figure, axis = plt.subplots(len(languages_list))
        figure.tight_layout()
        for i in languages_list : 
            temp_lang_df = self.error_df[self.error_df["L1"] == i]
            errors_counter = dict(Counter(temp_lang_df["errors"].to_list()).most_common())
            axis[languages_list.index(i)].bar(list(errors_counter.keys()), list(errors_counter.values()))
            axis[languages_list.index(i)].set_title("Most common errors for  " + i)
            axis[languages_list.index(i)].tick_params(axis='x', labelrotation=45)
        plt.show()


    def nb_of_students_per_language(self):
        # Make it look better + add plot with number of texts to compare 
        user_count = []
        txt_count = []
        for i in self.languages:
            temp_lang_df = self.dataframe[self.dataframe["L1"] == i]
            temp_users = len(set(temp_lang_df["user"].to_list()))
            user_count.append(temp_users)
            #txt_count.append(len(temp_users["text_id"].to_list()))
        assert len(self.languages) == len(user_count)
        dict_to_plot = dict(zip(self.languages, user_count))
        dict_to_plot = {k: v for k, v in sorted(dict_to_plot.items(), key=lambda item: item[1])}

        
        plt.bar(dict_to_plot.keys(), dict_to_plot.values(), color ='maroon', 
            width = 0.2)
        plt.xlabel("Language")
        addlabels(dict_to_plot.keys(), list(dict_to_plot.values()))

        plt.ylabel("Number of writers")
        plt.title("Number of writers per language")
        plt.show()     
        

 




