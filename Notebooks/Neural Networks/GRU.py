import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU
from sklearn.preprocessing import LabelEncoder

results = ''
iterations = 5 # accuracy 94

# Load the dataset
data = pd.read_csv("./Datasets/Crop_Recommendation_Dataset.csv")
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Convert string columns to numeric type
numeric_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data.dropna(inplace=True)

# Separate features and labels
X = data.drop('label', axis=1)
Y = data['label']

# Normalize the features using Min-Max scaling
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

for i in range(iterations):
    # Splitting Dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size=0.15, random_state=42)

    # Reshape features for RNN input
    X_train_rnn = X_train.reshape(-1, 1, X_train.shape[1])
    X_test_rnn = X_test.reshape(-1, 1, X_test.shape[1])

    # Define the GRU model
    model = Sequential([
        GRU(64, input_shape=(1, X_train.shape[1])),  # GRU layer with 64 units
        Dense(32, activation='relu'),  # Dense layer with 32 units and ReLU activation
        Dense(len(Y.unique()), activation='softmax')  # Output layer with softmax activation for multi-class classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_rnn, Y_train, epochs=45, batch_size=32, validation_split=0.2)

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(X_test_rnn, Y_test)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    results += f'Test Loss = {test_loss} and Test accuracy = {test_accuracy} \n'

print(results)
