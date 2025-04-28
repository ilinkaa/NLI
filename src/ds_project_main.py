#from cz_data_pro_temp import process_all_as_dict
from BAWEdata import *
from NGramnew import NGramAnalysertwo
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import time 
import argparse 



n_grams ={'unigram':(1,1),'bigram':(2,2),'trigram':(3,3)}

metrics = ["accuracy","precision","f1_score","recall"]

variable = "sentences"

query =["French", "German",'Japanese', 'Polish','Chinese']
data_dict = get_all_data_from_query(query)


variable = "lemmas"
metrics = ["accuracy","precision","f1_score","recall"]

from cz_data_pro_temp import process_all_as_dict

n_grams ={'unigram':(1,1),'bigram':(2,2),'trigram':(3,3)}

query = {"levels":["A1","A2","B1","B2","C1"], "L1":["French", "German",'Japanese', 'Polish','Chinese']}

errors,data_cz = process_all_as_dict(query, column="sentence_corrected")
analyzer = NGramAnalysertwo(data=data_dict, dataset_name="BAWE")
results=analyzer.n_gram_analysis(variable=variable, tf_idf=True, stopwords= True)

def format_parameters(param_dict):
    new_keys = list()
    for i in list(param_dict.keys()):
        i= i.replace("estimator__","")
        new_keys.append(i)
    new_dict = dict(zip(new_keys,list(param_dict.values())))
    str_ver = str(new_dict).replace("{","")
    str_ver = str_ver.replace("}","")
    str_ver = str_ver.replace("'","")
    return str_ver


def plot_results(res_dict):
    sns.color_palette("flare", as_cmap=True)

    for i in list(res_dict.keys()):

        print(i)
    f, (ax1, ax2 ,ax3) = plt.subplots(3)
    ax1.set_ylim(0,1.0)
    ax2.set_ylim(0,1.0)
    ax3.set_ylim(0,1.0)
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlepad'] = -14
    """
    uni_params= format_parameters(res_dict["unigram"]["param_results"])
    bi_params= format_parameters(res_dict["unigram"]["param_results"])
    tri_params= format_parameters(res_dict["unigram"]["param_results"])
    """
    ax1.set_title('\nUnigram\n',fontsize= 8)
    ax2.set_title('\nBigram\n',fontsize= 8)
    ax3.set_title('\nTrigram\n',fontsize= 8)
    sns.set_style("darkgrid")
    a1 =sns.barplot(ax = ax1,x = metrics, y= [res_dict["unigram"][i] for i in metrics])
    a1.bar_label(a1.containers[0])
    a2= sns.barplot(ax = ax2,x = metrics, y= [res_dict["bigram"][i] for i in metrics])
    a2.bar_label(a2.containers[0])
    a3 = sns.barplot(ax = ax3,x = metrics, y= [res_dict["trigram"][i] for i in metrics])
    a3.bar_label(a3.containers[0])
    #Make title and plot look better, with values and grid background, better colors
    title = variable.capitalize()+ " N-Grams " + "Dataset: CzEsl"
    f.suptitle(title, fontsize=10)
    

    plt.show()
    return f

def confusion_matrix(res_dict):
    confusion_unigram = res_dict["unigram"]["confusion_matrix"]
    confusion_bigram = res_dict["bigram"]["confusion_matrix"]
    confusion_trigram = res_dict["trigram"]["confusion_matrix"]
    fig, (ax1, ax2,ax3) = plt.subplots(nrows=3, figsize=(5,5))
    sns.heatmap(confusion_unigram, ax= ax1, annot=True)
    sns.heatmap(confusion_bigram, ax= ax2, annot=True)
    sns.heatmap(confusion_trigram, ax= ax3, annot=True)
    plt.show()
    return fig

# To do still : save results ! 
# Run all analysis duhh
#  



f = plot_results(results)
f = confusion_matrix(results)



#analyzer.save_data()
"""

variable = "upos"
results=analyzer.n_gram_analysis(variable=variable, tf_idf=False, stopwords= False)
f = plot_results(results)
f.savefig("Czesl_uposll_levels_corrected.png")
f.savefig("Czesl_uposll_levels_corrected.png")

f = confusion_matrix(results)
f.savefig("Czesl_upos_confll_levels_corrected.png")
#analyzer.save_data()

variable = "deprel"
results=analyzer.n_gram_analysis(variable=variable, tf_idf=False, stopwords= False)
f = plot_results(results)
f.savefig("Czesl_deprelll_levels_corrected.png")
f = confusion_matrix(results)
f.savefig("Czesl_deprem_confll_levels_corrected.png")
#analyzer.save_data()
"""
""""
parser = argparse.ArgumentParser(
                    prog='NGramAnalyzer',
                    description='N-gram based SVC classification based on different variables for two datasets',
                    epilog='UHM')
parser.add_argument('-d', '--dataset', default="BAWE")      # option that takes a value
parser.add_argument('-stopwords', '--stopwords',
                    default=True, type = bool)
parser.add_argument('-tf','--tfidf', default = False, type = bool)
parser.add_argument('-v','--variable', default="sentences")
parser.add_argument('-q','--query', nargs='+', default= ["French", "German",'Japanese', 'Polish','Chinese'])
parser.add_argument('-l','--levels', nargs='+', default= ["B1", "B2",'C1'])

args = parser.parse_args()

print(args.dataset)
query = args.query
stopwords = args.stopwords
tfidf = args.tfidf
variable = args.variable
if args.dataset =="BAWE":
    from BAWEdata import *
    data = get_all_data_from_query(query)
elif args.dataset =="czesl":
    from cz_data_pro_temp import process_all_as_dict
    data = process_all_as_dict(query, args.levels)

analyzer_ngram= NGramAnalysertwo(data =data)
analyzer_ngram.n_gram_analysis(tf_idf= tfidf, stopwords=stopwords, variable=variable)
plot_results()
confusion_matrix()
analyzer_ngram.save_data()


"""