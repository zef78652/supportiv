# supportiv
# Sentence Relevancy classification Using BERT and SVM

This Jupyter notebook demonstrates an approach for classifying whether the two sentences are relevant or not, using Sentence Transformers library for encoding and various metrics, followed by training and testing an SVM model on the generated feature set.


## Contents

1. **Data loading and preprocessing**
    * Load dataset from 'assignment_A.csv'
    * Check and remove NaN values
    * Pre-process the text data (lower case, remove punctuations, tokenize and lemmatize)
    * Map the text preprocessing onto the dataset

2. **Feature extraction**
    * Use Sentence Transformers (BERT) to get sentence embeddings, used the bert-base-nli-mean-tokens model since its fine-tuned on NLI task which is somewaht similar to our task
    * Concatenate embeddings for sentence pairs
    * Calculate cosine similarity, Jaccard similarity, and Levenshtein distance for each sentence pair
    * Add these similarity metrics to the dataset
    * Used similarity metrics as additional features
    
3. **Dataset preparations**
    * Split the dataset into train and test sets (80% / 20%)
    * Combine embeddings of both sentences as an input feature
    * Perform feature scaling with StandardScaler

4. **SVM model training and testing**
    * Perform grid search with cross-validation to find the best parameters
    * Train an SVM model with RBF kernel and degree 2 (best parameters)
    * Predict labels and evaluate the performance on the test set

5. **Other Experiments**
    *Chose SVM, over Logistic regression, decision trees as it captures higher complexity and non-linear decision boundaries: In general, logistic regression models are limited by their linear nature.
    *Experiments around passing embeddings and distance features were experimented, but since cosine sim was being produced by emebeddings themselves, later decided to go forward with only cosine and jaccard distance as two features to train the SVM on.
    
6. **Potential Improvements**
    *More sophisticated embeddings like embeddings from OpenAI could have been used. Maybe simpler embedding techniques like Word2vec, or maybe even tf-idf could have been experimented with. 
    *More complex models like Neural Networks or Boosting based techniques could have been experimented with. Maybe an ensemble of models could have been done.
    * Could have fine-tuned BERT model itself on the data for classificiation rather than just using the emeddings.


