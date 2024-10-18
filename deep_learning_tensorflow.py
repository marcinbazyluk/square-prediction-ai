import random
import numpy
import tensorflow
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data_set_size = 1000
x_range = 2000
y_range = 2000

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
data = numpy.vstack((x_list, y_list)).T

X_train, X_test, y_train, y_test = train_test_split(data, outcome, test_size=0.2)
print("Number of samples in training set: " + str(X_train.shape[0]))
print("Number of samples in test set: " + str(X_test.shape[0]))

# converting the data and outcomes into numpy arrays (requirement of tensorflow.keras)
X_train = numpy.array(X_train)
X_test = numpy.array(X_test)
y_train = numpy.array(y_train)
y_test = numpy.array(y_test)

# define the model
tensorflow.random.set_seed(10) # seed for reproducible result
tensorflow.keras.utils.set_random_seed(10)

model = tensorflow.keras.Sequential([
    tensorflow.keras.layers.Dense(20, activation='relu', input_shape=(2,)), # there are 2 input parameters, that's why this value for input_shape
    tensorflow.keras.layers.Dense(10, activation='relu'),
    tensorflow.keras.layers.Dense(2, activation='softmax')
])


# compile the model and define the loss function, the optimizer, and the training epochs.

model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 1000

# train the model
model.fit(X_train, y_train, epochs=epochs, verbose=0.5)

print("Model training complete")

# test the model
y_predicted_probabilities = model.predict(X_test)
y_predicted = tensorflow.argmax(y_predicted_probabilities, axis=1)

# Compare the predicted values for the test set (y_predicted) against the expected values (y_test)
print("Classification Report:")
print(classification_report(y_test, y_predicted))

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

outputs = (False, True)
def predict(samples):
    prediction_probabilities = model.predict(numpy.array(samples), verbose=0)
    # argmax gets the index of the maximum value in an array
    predictions = [outputs[numpy.argmax(p)] for p in prediction_probabilities]
    return predictions

predictions = predict(X_samples)
print(predictions)
