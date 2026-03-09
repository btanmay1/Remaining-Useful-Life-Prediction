
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from preprocessing.data_loader import load_data, create_sequences
from models.lstm_model import build_model

X, y = load_data("data/battery_dataset.csv")
X_seq, y_seq = create_sequences(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42
)

model = build_model((X_seq.shape[1], X_seq.shape[2]))

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test)
)

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)

print("RMSE:", rmse)
print("MAE:", mae)

plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label="validation")
plt.legend()
plt.title("Training Loss")
plt.show()
