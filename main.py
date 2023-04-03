import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load and preprocess the dataset
data = pd.read_csv('insurance_claims.csv')
data['claim_date'] = pd.to_datetime(data['claim_date'])
data = data.sort_values(by='claim_date')

# Scale the numerical features
scaler = MinMaxScaler()
data[['claim_amount']] = scaler.fit_transform(data[['claim_amount']])

# Prepare the input and output sequences
sequence_length = 10  # The number of previous claims to consider for prediction
X = []
y = []

for i in range(len(data) - sequence_length):
    X.append(data.iloc[i:i + sequence_length, 1:6].values)
    y.append(data.iloc[i + sequence_length, -1])

X = np.array(X)
y = np.array(y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Save the model for future use
model.save('fraud_detection_lstm.h5')
