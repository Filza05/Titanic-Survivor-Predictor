from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib as plt
import tensorflow as tf
from tensorflow import feature_column as fc

dftrain = pd.read_csv('train.csv')
dfeval = pd.read_csv('eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_function(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df)) 
    #dict(data_df) converts the data_df DataFrame into a dictionary
    #where each column becomes a key-value pair, with the column name
    # as the key and the column values as the values. 
    
    # dict(data_df) converts the data_df DataFrame into a dictionary 
    # where each column becomes a key-value pair, with the column name 
    # as the key and the column values as the values. 
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function

train_input_fn = make_input_function(dftrain,y_train)
eval_input_fn = make_input_function(dfeval,y_eval,num_epochs=1,shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = list(linear_est.predict(eval_input_fn))
print(dftrain.loc[10])
print(y_eval.loc[10])
print(result[10]['probabilities'])