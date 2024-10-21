import sys
import random
import numpy
import pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data_set_size = 1000
x_range = 2000
y_range = 2000
train_batches = 10

x_list = []
y_list = []
outcome = []
# let's assume that all pairs of numbers for which each number in a pair is within the 1-1000 range should return True
for i in range(0, int(data_set_size/4)):
  x = random.randint(0, int(x_range/2))
  x_list.append(x)
  y = random.randint(0, int(y_range/2))
  y_list.append(y)
  outcome.append(True)

# now let's assume that all other pairs of numbers should return False
for i in range(0, int(data_set_size/4)):
  x = random.randint(0, int(x_range/2))
  x_list.append(x)
  y = random.randint(int(y_range/2), y_range)
  y_list.append(y)
  outcome.append(False)

for i in range(0, int(data_set_size/4)):
  x = random.randint(int(x_range/2), x_range)
  x_list.append(x)
  y = random.randint(0, int(y_range/2))
  y_list.append(y)
  outcome.append(False)

for i in range(0, int(data_set_size/4)):
  x = random.randint(int(x_range/2), x_range)
  x_list.append(x)
  y = random.randint(int(y_range/2), y_range)
  y_list.append(y)
  outcome.append(False)

# merging x_list, y_list and outcome into a 2-dim array which will be our dataset
data = numpy.vstack((x_list, y_list,outcome)).T

train_data, test_data = train_test_split(data, test_size=0.2)
print("Number of samples in training set: " + str(train_data.shape[0]))
print("Number of samples in test set: " + str(test_data.shape[0]))

# split test data into independent and dependent values
y_test = test_data[:,-1:]
X_test  = numpy.delete(test_data, [-1], axis=1)

# define the model
model = GaussianNB()

# remove last N rows that will not fit into a full batch
rows_to_remove = train_data.shape[0] % train_batches
train_data = train_data[:-rows_to_remove or None]

# train the model in batches
batch_no = 1
for train_data_batch in numpy.split(train_data, train_batches):
  # split train batch data into independent and dependent values
  y_train_batch = train_data_batch[:,-1:]
  x_train_batch = numpy.delete(train_data_batch, [-1], axis=1)
  model.partial_fit(x_train_batch, y_train_batch, numpy.unique(y_train_batch))
  print("Training batch " + str(batch_no) + "/" + str(train_batches) + " completed.")

  # test the model after every batch
  y_predicted = model.predict(X_test)

  # Compare the predicted values for the test set (y_predicted) against the expected values (y_test)
  print("Classification Report:")
  print(classification_report(y_test, y_predicted))

  batch_no += 1
  
# test the model with sample cases

X_samples = []
X_samples.append([500, 500])	# True
X_samples.append([450, 120])	# True
X_samples.append([0, 200])	# True
X_samples.append([50, 900])	# True
X_samples.append([999, 999])	# True
X_samples.append([1500, 700])	# False
X_samples.append([700, 1500])	# False
X_samples.append([1500, 1500])	# False
X_samples.append([1001, 1001])	# False
X_samples.append([2000, 2000])	# False
print(model.predict(X_samples))