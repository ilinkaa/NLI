# NLI
Short project looking at Native Language Identification for two datasets (BAWE, a corpus of English academic texts, and CzEsl, a L2 Czech learner's dataset), by using a simple N-Gram classification method on different variables. 


We examine different methods for native language identification for two different corpora, by making use of lexical, morphological and lexical data. 
By looking at two different corpora (BAWE, which regroups essays written by college students from different L1 backgrounds with advanced proficiency in English, and Czesl, which compiles short texts written by non-native speakers of Czech), we hope to explore the way these different methods apply to two different languages while conducting an analysis on different corpora. 
Our three objectives are: 
Run a simple N-gram analysis for different variables on both the dataset and see how they perform.
Evaluate the impact of the dataset on the method. Our corpora are not only different in terms of the language they represent, but in terms of level: while the BAWE dataset contains longer essays written by advanced learners, the Czesl texts come from language classes from people still actively learning the language. While we take inspiration from Tydlitátová (2016) for this study, we note that there was no consideration of how the selected levels might have impacted the results. 
We try to not only make use of n-grams, but also syntactic data, by trying out graph kernels on the dependency relations we obtained. 
Finally, we also look at the errors annotation provided in the Czesl dataset and attempt SVM classification of L1 based on them.

For the L1 classification task, we selected our target languages by taking into account how they overlap in both of the datasets (choosing languages with a significant number of documents), while aiming to encompass different language families. 


With this in mind, we settled on Mandarin Chinese, Japanese, Polish, French and German. 
After downsampling in order to ensure balance across the dataset, we process the data so as to obtain words, lemmas, POS tags, and dependency relations, and look at unigrams, bigrams and trigrams. For the BAWE dataset, these were available with minimal processing, while for Czesl, we used the Stanza tool to obtain dependency relations. 

After obtaining the N-grams, we perform both count and tf-idf based vectorization before feeding the data to a SVM classifier. Inspired by Tydlitátová’s thesis, we run a grid search in order to obtain the best performing parameters, by experimenting with a variety of kernels, gamma and C parameter values. 

However, N-grams do not capture the long-term dependencies reflected in syntactic information. We propose a novel approach, which attempts classification based on dependency relations using the graph kernels. We convert the dependency relations into a tree format and use the precomputed Weisfeiler-Leman kernel from the GraKel library, which was initially designed to capture isomorphism between two graphs by making use of both the node and the edge data. By using this, we could in theory base the classification on the similarity between types of sentence structures and see how it relates to L1. 


## Results

### BAWE dataset:

The best performing N-gram model across all types of n-grams is the Tf-IDF lemmas N-gram model which includes stopwords removal. Interestingly, Tf-IDF vectorization seems to help with the bigram accuracy for all types of of n-grams, whereas for count-based methods, the bigram accuracy tends to be lower than that of the unigram and bigram.The one exception is the count-based word analysis which includes stopwords, in which accuracy values for the bigram model are higher. 
We note that for most models, similar kernel parameters were chosen, resulting in most models using a linear kernel, with the exception of the UPOS model, which relies on the poly kernel instead. 

## CzEsl dataset:

We select texts at the B1-B2 level under the assumption that they are more likely to contain a higher type to token ratio and more varied structures. However, we find that the classifiers perform significantly better at the word level when including data from lower levels as well. Moreover, during some exploratory work, we saw that the level is much easier to predict than the L1 by using the same methods. Ideally, this claim could be supplemented by a more in-depth analysis of the type to token ratio of the dataset and the way it is distributed among the different levels / languages. 
Our second find is that using the corrected sentences which are provided by the corpora actually improve the metrics on the word and lemma analysis, while the numbers of the upos and dependency labels analysis mostly stays the same. Moreover, as opposed to the BAWE dataset, metrics seem to increase with the n-gram range. 
Additionally, the confusion matrix plots show that prediction errors generally do not coincide with closeness between languages, and reveals that our models have trouble generalizing to smaller classes. 

Overall, this confirms that the classic N-gram approach using lemmas remains the most efficient one. Moreover, for the CzEsl dataset, accuracy tends to be lower when not using the corrected version of the sentence, probably because errors are not that common in the dataset (and are mostly annotated for spelling related mistakes, instead of providing grammatical information), which might create more outliers and make it harder for the model to generalize. 

In terms of dataset comparison, similar methods do well in both datasets despite them being different in terms of content. The higher scores on the BAWE dataset might be simply due to the fact the data points available reflect longer texts and thus provide more information. 

## Weisfeiler-Leman Kernel

Graph kernels allow to classify graph-like structures by examining their similarity. While there have been attempts at designing graph kernels for linguistic structures (Suzuki et al., 2003 proposed their HDAG kernel, which accounts for hierarchical dependency relations), these do not simultaneously take the structure and the node and the edge labels into account. We propose to use the Weisfeiler-Leman kernel for classification purposes






## Sources:

Malmasi, S., Evanini, K., Cahill, A., Tetreault, J., Pugh, R., Hamill, C., ... & Qian, Y. (2017, September). A report on the 2017 native language identification shared task. In Proceedings of the 12th Workshop on Innovative Use of NLP for Building Educational Applications (pp. 62-75).

Suzuki, J., Hirao, T., Sasaki, Y., & Maeda, E. (2003, July). Hierarchical directed acyclic graph kernel: Methods for structured natural language data. In Proceedings of the 41st Annual Meeting of the Association for Computational Linguistics (pp. 32-39).

Tydlitátová, L. (2016). Native Language Identification of L2 Speakers of Czech.
