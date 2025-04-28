from BAWEdata import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from cz_data_pro_temp import *
query = {"L1":["ru","zh","ja","de",'fr'],"levels":["B1","B2"]}
data_dict = process_all_as_dict(select_dict=query, column="sentence")

data_dict = pd.DataFrame.from_dict(data_dict)

def get_cosine(data_dict,category, feature):
    vectorizer = CountVectorizer()
    sentence_vector = vectorizer.fit_transform(data_dict[feature])
    #Dictionnary only containing vectorizable columns
    new_dict = dict((k, data_dict[k]) for k in data_dict.keys() if k not in ("conllu", "grakel_data"))

    balanced = pd.DataFrame.from_dict(new_dict)
    #print(balanced)

    sentence_vector = sentence_vector.toarray()
    cat_labels = list(set(balanced[category].to_list()))
    dict_results = dict()
    copy_cat = cat_labels
    for i in cat_labels:
        
       
        indexes_lang = list(balanced.loc[balanced[category]==i].index)
        select_ind_lang = np.take(sentence_vector, indexes_lang, 0)
        copy_cat.remove(i)
        for j in copy_cat: 
            indexes_lang = list(balanced.loc[balanced[category]==j].index)
            select_ind_other_lang = np.take(sentence_vector, indexes_lang, 0)
            dict_results[i+"-"+j] = cosine_similarity(select_ind_lang, select_ind_other_lang)
    for res in dict_results.keys():
        m = np.mean(dict_results[res])
        dict_results[res] = np.mean(dict_results[res])
    return dict_results

res_check = get_cosine(data_dict=data_dict, category="L1",feature="sentence")
print(res_check)

