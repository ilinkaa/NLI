from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score,f1_score,recall_score
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsOneClassifier
from weisfeiler_lehman import WeisfeilerLehman
from vertex_histogram import VertexHistogram
import seaborn as sns 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from BAWEexp import formatted_for_grakel
from trees_try_grakel import process_data
import pandas as pd 

#cz_data = process_data()
print("processing done")
param_grid = {
    "estimator__C": [1, 10, 100, 1000],
    "estimator__gamma": [0.001, 0.0001],
    "estimator__degree":[1, 2, 3, 4]
    
}
metrics = ["accuracy", "precision","recall","f1_score"]

gk = WeisfeilerLehman(n_iter=4, base_graph_kernel=VertexHistogram, normalize=True)
dummy_clf = DummyClassifier(strategy="most_frequent")
def weisfeiler_lehman_svm(data,dataset):
    G_train, G_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.1, random_state=42)
    K_train = gk.fit_transform(G_train)
    K_test = gk.transform(G_test)
    num_dummy_results = []
    
    dummy_clf.fit(G_train, y_train)
    d_pred = dummy_clf.predict(K_test)
    d_accuracy = round(accuracy_score(y_test, d_pred),2)
    d_precision = round(precision_score(y_test, d_pred, average = "weighted"),2)
    d_recall = round(recall_score(y_test, d_pred, average = "weighted"),2)
    d_fone_score = round(f1_score(y_test, d_pred, average = "weighted"),2)
    num_dummy_results.append(d_accuracy)
    num_dummy_results.append(d_precision)
    num_dummy_results.append(d_recall)
    num_dummy_results.append(d_fone_score)
    model_dummy = ["dummy_"+ dataset] * len(num_dummy_results)
    sh = OneVsOneClassifier(SVC(kernel="precomputed"))
    #sh = GridSearchCV(clf, param_grid=param_grid, scoring="accuracy")
    sh.fit(K_train, y_train)   
    y_pred = sh.predict(K_test)
    acc = accuracy_score(y_pred,y_test)
    num_results = []
    # Computes and prints the classification accuracy
    acc = accuracy_score(y_test, y_pred)
    precision = round(precision_score(y_test, y_pred,average = "weighted"),2)
    recall = round(recall_score(y_test, y_pred, average = "weighted"),2)
    fone_score = round(f1_score(y_test, y_pred, average = "weighted"),2)
    num_results.append(acc)
    num_results.append(precision)
    num_results.append(recall)
    num_results.append(fone_score)
    model = [dataset] * len(num_results)
    
    
    results = {"dataset":dataset,"accuracy":acc, "precision":precision,"recall":recall,"f1_score":fone_score}
    results_d = {"accuracy":d_accuracy, "precision":d_precision,"recall":d_recall,"f1_score":d_fone_score}
    results_dataframe= pd.DataFrame([model, metrics, num_results], index=['dataset', 'metric','score']).T
    dummy_results_dataframe= pd.DataFrame([model_dummy, metrics, num_dummy_results], index=['dataset', 'metric','score']).T
    return results_dataframe, dummy_results_dataframe

cz_data = process_data()

bawe_actual_results, baseline_bawe = weisfeiler_lehman_svm(formatted_for_grakel,"BAWE")
cz_actual_results, baseline_cz = weisfeiler_lehman_svm(cz_data,"CzEsl")

final_df = pd.concat([bawe_actual_results,cz_actual_results], sort=False)
g = sns.catplot(
    data=final_df, kind="bar",
    x="metric", y="score", hue="dataset",
    errorbar="sd", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("", "")
plt.ylim(0,1)
ax = g.facet_axis(0, 0)  # or ax = g.axes.flat[0]
plt.show()
# iterate through the axes containers



