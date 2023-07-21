# Implementation-of-Bidirectional-Model
This project is implemented using python in which Bidirectional model is Implemented on sentiment140-subset.csv dataset.Link of the Dataset is :: https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/sentiment-analysis-is-bad/data/sentiment140-subset.csv.zip
#                            Implementation of bidirectional model on Sentiment-140-Subset Dataset
The code provided implements a bidirectional LSTM model for sentiment analysis using TensorFlow. Let's break down the steps:
1.	Importing the Dataset: The code downloads the dataset from a URL, unzips it, and loads it into a pandas DataFrame. The dataset contains two columns: 'polarity' (0 or 1) indicating negative or positive sentiment, and 'text' containing the text of the tweets.
2.	Tokenization: The text data is tokenized using the Tokenizer class from TensorFlow. This step converts the text into sequences of integers, and the vocabulary size is limited to the max_features variable (set to 4000 in this case).
3.	Creating the Bidirectional LSTM Model: The code defines a sequential model in TensorFlow. The model consists of an embedding layer, a spatial dropout layer, a bidirectional LSTM layer, and a dense layer with a softmax activation function. The bidirectional LSTM layer allows the model to consider both past and future information while processing the input data, which helps capture long-range dependencies in the text.
4.	Train-Test Splitting: The dataset is split into training and testing sets for model evaluation. The 'polarity' column is one-hot encoded to convert the target labels into a binary representation.
5.	Training the Model: The model is trained on the training data using the fit function. It runs for 20 epochs with a batch size of 500, and the training process is monitored to track the loss and accuracy metrics.
6.	Model Evaluation: The trained model is evaluated on the test set using the evaluate function, and the test accuracy is calculated.
7.	Sentiment Prediction: A new tweet, "I do not recommend this product," is tokenized, padded, and then fed into the trained model for sentiment prediction. The model predicts the sentiment as "Negative."
Overall, this code provides a simple implementation of a bidirectional LSTM model for sentiment analysis on the provided dataset. The model achieves an accuracy of approximately 74% on the test set, as indicated in the evaluation step.

