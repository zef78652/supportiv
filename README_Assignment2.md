# Tag classification using BERT Embedding and Linear SVC 


## Steps

1. Import the required libraries.
2. Read the dataset as a pandas DataFrame.
3. Perform data preprocessing -tokenize, lemmatize, lowercasing.
4. Check the class distribution,  we merge miscelleneuos tags with minority class as very few examples are available.
5. Pre-process the sentences in the dataset.
6. Drop rows with missing data.
7. Encode the preprocessed text with SentenceTransformer with all-MiniLM-L6-v2'.
8. Encode the tags using LabelEncoder.
9. Since the class is highly imablanced we attempt to make them uniform with SMOTE and train-test split.
10. Scale the input features.
11. Train various classifiers:
    - Support Vector Classifier (SVC)
    - LinearSVC with one vs rest classifier
    - XGBoost
12. Perform hyperparameter tuning with GridSearchCV and kfold validation to avoid overfitting.
13. Evaluate these classifiers and record their accuracy and classification report, we focus on model with higher f1 wieghted score instead of accuracy.
14. Create a function to predict the topic of a given input text.

## Potential Improvement

*More sophisticated embeddings could have been used.

*A Neural Network with cross entropy loss could be used to train a multi-class classifier.

*A bert based model could also be fine-tuned.

## Usage

After setting up and running the notebook, you can evaluate the input with the `predict_topic` function. For example:

```python
input_text = "financial loss, unable to survive"
predicted_label = predict_topic(input_text)
print("Predicted Label:", label_encoder.inverse_transform(predicted_label))

input_text = "hectic work unbearable stress"
predicted_label = predict_topic(input_text)
print("Predicted Label:", label_encoder.inverse_transform(predicted_label))
```

This will return the predicted label for the given input text.
