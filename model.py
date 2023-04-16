import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the training data
train_data = pd.read_csv('/content/drive/MyDrive/dataset/train.csv')

# Split the training data into features and labels
X_train = train_data.drop('VERDICT', axis=1)
y_train = train_data['VERDICT']

# Load the test data
test_data = pd.read_csv('/content/drive/MyDrive/dataset/train.csv')

# Split the test data into features and labels
X_test = test_data.drop('VERDICT', axis=1)
y_test = test_data['VERDICT']

# Create a multi-layer perceptron classifier
model = MLPClassifier(hidden_layer_sizes=(50, 25), activation='relu', solver='adam', random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the performance of the model on the test data
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create a new DataFrame to store the predictions
predictions = pd.DataFrame(y_pred, columns=['VERDICT'])

# Save the predictions to a CSV file
predictions.to_csv('/content/drive/MyDrive/dataset/predictions.csv', index=False)