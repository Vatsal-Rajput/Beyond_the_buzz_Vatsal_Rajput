import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_data = pd.read_csv('/content/drive/MyDrive/dataset/train.csv')

# Split the training data into features and labels
X_train = train_data.drop('VERDICT', axis=1)
y_train = train_data['VERDICT']

# Load the test data
test_data = pd.read_csv('/content/drive/MyDrive/dataset/test.csv')

# Remove the ID column from the test data
test_data = test_data.drop('Id', axis=1)

# Create a multi-layer perceptron classifier
model = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(test_data)

# Save the predictions to a CSV file
predictions = pd.DataFrame(y_pred, columns=['VERDICT'])
predictions.to_csv('/content/drive/MyDrive/dataset/predictions.csv', index=False)

# Print the accuracy score of the model on the training data
train_accuracy = accuracy_score(y_train, model.predict(X_train))
print("Training accuracy:", train_accuracy)

