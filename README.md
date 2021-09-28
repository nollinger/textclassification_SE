# textclassification_SE

'''
@author: noll_richard

'''



'''

1. creating functions for preprocessing/word embedding of data.

'''

from string import punctuation
from string import digits

def remove_umlaut(text): 
    
    """
    Replace umlauts for a given text
    
    :param text: text as string
    :return: manipulated text as str
    """
    
    tempVar = text # local variable
    
    # Using str.replace() 
    
    tempVar = tempVar.replace('ä', 'ae')
    tempVar = tempVar.replace('ö', 'oe')
    tempVar = tempVar.replace('ü', 'ue')
    tempVar = tempVar.replace('Ä', 'Ae')
    tempVar = tempVar.replace('Ö', 'Oe')
    tempVar = tempVar.replace('Ü', 'Ue')
    tempVar = tempVar.replace('ß', 'ss')
    tempVar = tempVar.replace('ãÿ', 'ss')
    tempVar = tempVar.replace('ÃŸ', 'ss')
    tempVar = tempVar.replace('Ã¶', 'oe')
    tempVar = tempVar.replace('Ã¼', 'ue')
    tempVar = tempVar.replace('ã¼', 'ue')
    tempVar = tempVar.replace('Ã¤', 'ae')
    
    return tempVar



def punctuation_number(text):
    
    """
    Delete punctuation '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    and digits '0-9'
    
    :return: text without punctuation and digits
    """
    
    remove_pun = str.maketrans('', '', punctuation)
    text_wo_pun = text.translate(remove_pun)
    remove_digits = str.maketrans('', '', digits) #using native string function
    text_wo_num_pun = text_wo_pun.translate(remove_digits)
    return text_wo_num_pun


# import stop words and tokenization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

text_file = open("C:\\Users\\nollr\\Desktop\\stop_words_german.txt", "r")
stopword = text_file.read().split()
text_file.close()

# tokenize data: (1, 1) means only unigrams, (1, 2) means unigrams and bigrams, and (2, 2) means only bigrams
vects = CountVectorizer(stop_words=stopword, ngram_range=(1, 1)) 

tf_idf = TfidfTransformer(use_idf = False) #tf idf 


'''

2. import of the training data

'''


import pandas as pd
import os
import numpy as np


training_lst=[] #list with all the training data
disease=[] #list with disease names

for root, dirs, files in os.walk("C:\\Users\\nollr\\Desktop\\SD_DATEN\\SEMD"): 
   #path with the training-data for both diseases.
   for name in files:
      if name.endswith(".txt"): # data is saved as a txt file.
          file = os.path.join(root, name)
          file= open(file, "r")
          lines= file.read()
          lines = remove_umlaut(lines) #replacing the 'umlaute'
          lines = punctuation_number(lines) #removing punctuation and numbers
          training_lst.append(lines)
          if name.startswith('CF+'):
              disease.append('CF+')
          elif name.startswith('SD+'):
              disease.append('SD+')
          file.close()
          
#print (training_lst[0])
#print (disease[0:10])


'''

3. transform data into csv file and create data matrix.

'''

dict = {
        'text': training_lst,
        'SE': disease
        }

df = pd.DataFrame(dict)
df.to_csv('SE1.csv',index=False,header=True) # store data in csv with two columns.
#df

data = pd.read_csv('SE1.csv') #text in column 1, classifier in column 2.
numpy_array = data.as_matrix() #create data matrix
X = numpy_array[:,0] #text
Y = numpy_array[:,1] #classifier



'''

4. introducing and creating a pipeline for different classifiers

'''



from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB #import of classifiers
from sklearn.linear_model import SGDClassifier

def classifier(classifier):
    
    '''
    
    different classifier
    
    '''
    
    if classifier == 'NB':
        classifier = MultinomialNB(alpha=0.001)
    elif classifier == 'SVM':
        classifier = SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, max_iter=10, random_state=42)
    
    return classifier
    

text_clf = Pipeline([('vect', vects),
 ('tfidf', tf_idf),
 ('clf', classifier('NB')) # choose classifier
])

    
    
'''

5. Using cross validation

'''


from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit

scoring = ['precision_macro', 'recall_macro', 'f1_macro']
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0) #validation splits and test size
scores = cross_validate(text_clf, X, Y, cv=cv, scoring=scoring, return_estimator=False)
#sorted(scores.keys())

print('f1-score: ', scores['test_f1_macro'].mean())
print('precision: ', scores['test_precision_macro'].mean())
print('recall: ', scores['test_recall_macro'].mean())

#text_clf.get_params()



'''

6. learning curve about the effect of different training sizes on validation scores.


'''




import matplotlib.pyplot as plt 
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_size=np.linspace(.1, 1.0, 5)):
    


    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_size, scoring ='f1_macro'
                       )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")
    
    
    return plt, train_scores_mean, test_scores_mean


fig, axes = plt.subplots(3, 2, figsize=(10, 15))

title = "Learning Curves (Naive Bayes)"

plot_learning_curve(text_clf, title, X, y=Y, axes=axes[0], ylim=(0.7, 1.01),
                    cv=cv, n_jobs=3)

plt.show()
