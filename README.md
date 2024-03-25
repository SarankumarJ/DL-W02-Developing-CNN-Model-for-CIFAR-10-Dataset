# DL-W02-Developing-CNN-Model-for-CIFAR-10-Dataset
### Name : Sarankumar J
### Reg No : 212221230087

## Program
```py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Step 2: Preprocess the data
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert class vectors to binary class matrices (one-hot encoding)
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Step 3: Define a more complex CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001),  # Lowering the learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the model for more epochs
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Plot iteration vs accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Iteration vs Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Iteration vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 5: Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Check if test accuracy is more than 70%
if test_acc > 0.70:
    print("Congratulations! Model achieved more than 70% accuracy.")
else:
    print("Model failed to achieve 70% accuracy.")
```

## Output
### Plot iteration vs accuracy and loss

![image](https://github.com/SarankumarJ/DL-W02-Developing-CNN-Model-for-CIFAR-10-Dataset/assets/94778101/0a1dfa21-f44d-4e51-958e-7418c9852691)


![image](https://github.com/SarankumarJ/DL-W02-Developing-CNN-Model-for-CIFAR-10-Dataset/assets/94778101/b6c3bab8-7636-4e64-9276-8da5cc88af98)

### Training the model to get more than 70% accuracy

![image](https://github.com/SarankumarJ/DL-W02-Developing-CNN-Model-for-CIFAR-10-Dataset/assets/94778101/4df13731-1986-46b2-9756-7fd47ffd50db)

## Github URL
https://github.com/SarankumarJ/DL-W02-Developing-CNN-Model-for-CIFAR-10-Dataset/
