# Digits-Classification-in-a-Distributed-Enviroment
## Description
This project uses PySpark to perform image classification on the MNIST dataset. Various preprocessing techniques and machine learning models are applied to classify the images. The application was implemented using Google Cloud.
## Code Structure
### Functions

- **readAssembleDataset(spark)**:
  - Reads the train and test data from Google Cloud Storage via HDFS.
  - Assembles pixel columns into a single feature vector.

- **normalize(train, test)**:
  - Normalizes the features using MinMaxScaler.

- **PCAVector(train, test, component)**:
  - Reduces the dimensionality of the features using PCA.

- **SVCSpark(train, test)**:
  - Trains and evaluates an SVM model with One-vs-Rest.

- **RandomForestSpark(train, test)**:
  - Trains and evaluates a Random Forest model.

- **MLPSpark(train, test, layers)**:
  - Trains and evaluates a Multilayer Perceptron model.
