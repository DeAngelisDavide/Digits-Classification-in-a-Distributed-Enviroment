
from pyspark.ml.feature import VectorAssembler
def readAssembleDataset(spark):
  #Load Dataset
  train_mnist_df = spark.read.csv('hdfs:///user/davide_deangelis001/mnist_train_small.csv', header=True, inferSchema=True)
  test_mnist_df = spark.read.csv('hdfs:///user/davide_deangelis001/mnist_test.csv', header=True, inferSchema=True)
  pixel_columns = train_mnist_df.columns[1:]
  pixel_columns_test = test_mnist_df.columns[1:]
  #Assemble pixel columns into a single vector
  assembler_train = VectorAssembler(inputCols=pixel_columns, outputCol="features")
  assembled_train_df = assembler_train.transform(train_mnist_df).drop(*pixel_columns)
  assembler_test = VectorAssembler(inputCols=pixel_columns_test, outputCol="features" )
  assembled_test_df = assembler_test.transform(test_mnist_df).drop(*pixel_columns_test)
  return assembled_train_df, assembled_test_df

from pyspark.ml.feature import MinMaxScaler
def normalize(train, test):
  #Initializes the MinMaxScaler
  scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
  #Applay the normalizations
  scaler_model = scaler.fit(train)
  scaled_train_df = scaler_model.transform(train).drop('features')
  scaler_model = scaler.fit(test)
  scaled_test_df = scaler_model.transform(test).drop('features')
  #rename the label to the original one
  scaled_train_df = scaled_train_df.withColumnRenamed('scaled_features', 'features')
  scaled_test_df = scaled_test_df.withColumnRenamed('scaled_features', 'features')
  return scaled_train_df, scaled_test_df

from pyspark.ml.feature import PCA
def PCAVector(train, test, component):
  pca = PCA(k = component, inputCol="features", outputCol='pca features')
  model = pca.fit(train)
  trainPCA = model.transform(train).drop("features")
  testPCA = model.transform(test).drop("features")
  #rename the label to the original one
  trainPCA = trainPCA.withColumnRenamed('pca features', 'features')
  testPCA = testPCA.withColumnRenamed('pca features', 'features')
  return trainPCA, testPCA

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LinearSVC, OneVsRest
def SVCSpark(train, test):
  #Initializes the LinearSVC
  svm = LinearSVC(labelCol="labels", featuresCol="features", maxIter=10)
  #Initializes the One-vs-Rest
  ovr = OneVsRest(classifier=svm, labelCol="labels", featuresCol="features")
  #Application of One-vs-Rest
  ovr_model = ovr.fit(train)
  ovr_predictions = ovr_model.transform(test)
  #Evaluation of the model
  evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
  svm_accuracy = evaluator.evaluate(ovr_predictions)
  return ovr_predictions, ovr_model, svm_accuracy

from pyspark.ml.classification import RandomForestClassifier
def RandomForestSpark(train, test):
  #Initializes the RandomForestClassifier
  rf = RandomForestClassifier(labelCol="labels", featuresCol="features", numTrees=100)
  #Application of Random Forest
  rf_model = rf.fit(train)
  rf_predictions = rf_model.transform(test)
  #Evaluation of the model
  evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction")
  rf_accuracy = evaluator.evaluate(rf_predictions, {evaluator.metricName: "accuracy"})
  return rf_predictions, rf_model, rf_accuracy

from pyspark.ml.classification import MultilayerPerceptronClassifier
def MLPSpark(train, test, layers):
  #Initializes the MLP
  mlp = MultilayerPerceptronClassifier(layers=layers, labelCol="labels", featuresCol="features", maxIter=100, blockSize=128, seed=1234)
  #Application of MLP
  mlp_model = mlp.fit(train_df)
  mlp_predictions = mlp_model.transform(test_df)
  #Evaluation of the model
  evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")
  mlp_accuracy = evaluator.evaluate(mlp_predictions)
  return mlp_predictions, mlp_model, mlp_accuracy


from pyspark.sql import SparkSession
import time
if __name__ == "__main__":
  #Setup Spark
  spark = SparkSession.builder.appName("sparkML").getOrCreate()
  assembled_train_df, assembled_test_df = readAssembleDataset(spark)
  train_df, test_df = normalize(train = assembled_train_df, test = assembled_test_df)
  train_df = train_df.withColumnRenamed('6', 'labels')
  test_df = test_df.withColumnRenamed('7', 'labels')
  train_df, test_df = PCAVector( train_df, test_df, 200)
  start = time.time()
  svc_predictions, svc_model, svc_accuracy = SVCSpark(train_df, test_df)
  end = time.time()
  print(f"Accuracy of SVM: {svc_accuracy}, Execution time: {end-start}")

  start = time.time()
  rf_prefictions, rf_model, rf_accuracy = RandomForestSpark(train_df, test_df)
  end = time.time()
  print(f"Accuracy of Random Forest: {rf_accuracy}, Execution time: {end-start}")

  start = time.time()
  mlp_predictions, mlp_model, mlp_accuracy = MLPSpark(train_df, test_df, [200, 128, 64, 10])
  end = time.time()
  print(f"Accuracy of MLP: {mlp_accuracy}, Execution time: {end-start}")
