import random
import numpy
import torch
import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data_set_size = 50
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

# Encode the data as PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# define the model
torch.manual_seed(20) # seed for reproducible result
class MyModel(torch.nn.Module):
    def __init__(
        self,
        num_input_features=2,
        num_neurons_layer1=20,
        num_neurons_layer2=10,
        num_targets=2
    ):
        super().__init__()
        # Define the neural network layers
        self.layer1 = torch.nn.Linear(num_input_features, num_neurons_layer1)
        self.layer2 = torch.nn.Linear(num_neurons_layer1, num_neurons_layer2)
        self.out = torch.nn.Linear(num_neurons_layer2, num_targets)

    def forward(self, X):
        # pass the data through the layers
        x = torch.nn.functional.relu(self.layer1(X))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.out(x)

model = MyModel()

# == Backward Propagation Configuration ==
# CrossEntropyLoss is a common loss function for classifcation
loss_function = torch.nn.CrossEntropyLoss()
# Use the Adam optimizer with a learning rate of 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 1000

# train the model
for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = loss_function(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Epoch: {i}. Loss: {loss.item()}")

print("Model training complete")

# test the model
y_predicted = []
with torch.no_grad():
    for i, data in enumerate(X_test):
        predictions = model(data)
        y_predicted.append(predictions.argmax())

# Compare the predicted values for the test set (y_predicted)
# against the expected values (y_test)
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
    inputs_dataframe = pandas.DataFrame(samples)
    inputs_tensor = torch.FloatTensor(inputs_dataframe.values)
    predictions = []
    for case in inputs_tensor:
        predictions_tensor = model(case)
        prediction_index = predictions_tensor.argmax().item()
        predictions.append(outputs[prediction_index])
    return predictions

predictions = predict(X_samples)
print(predictions)
