import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from google.colab import drive
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2

import os
drive.mount('/content/drive')
train_dir = f'/content/drive/My Drive/mini_project/data'
test_dir =  f'/content/drive/My Drive/mini_project/test'
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)  # Apply same rescaling to test set
target_width=128
target_height=128
batch_size=32
# Load images from directories (adjust target size and batch size as needed)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(target_width, target_height),
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical mode for multi-class classification
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(target_width, target_height),
    batch_size=batch_size,
    class_mode='categorical'
)
for images, labels in train_generator:
  # View sample images (uncomment if desired)
    plt.imshow(images[0])
    plt.show()
    break  # Only view one batch

print("Image data generators are ready for training!")
model = Sequential()

# Convolutional layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(target_width, target_height, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional layer 2
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional layer 3
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the output for feeding into fully connected layers
model.add(Flatten())

# Dense layer 1
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
num_classes=36
# Classification layer (output layer)
model.add(Dense(num_classes, activation='softmax'))


num_classes=36

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint_path = "training_checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1, save_freq='epoch')

# If there is a checkpoint, load it
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest:
    model.load_weights(latest)

# Train the model using train_generator
history=model.fit(train_generator, epochs=10, validation_data=test_generator)
# Save the model


print("Model training complete!")
test_loss, test_acc = model.evaluate(test_generator)

print('Test accuracy:', test_acc)
# Get a sample image path
from google.colab import drive
drive.mount('/content/drive')

img_path ="/content/drive/My Drive/mini_project/test/4/1.jpg"
# Preprocess the image (same preprocessing as data generators)
img = image.load_img(img_path, target_size=(target_width, target_height))
img = image.img_to_array(img)
img = img / 255.0  # Rescale
img = np.expand_dims(img, axis=0)  # Add a batch dimension

# Predict on the preprocessed image
predictions = model.predict(img)

# Decode the predicted class ( categorical encoding)
predicted_class = np.argmax(predictions[0])

# Get the class labels from the test data generator
class_labels = test_generator.class_indices
invert_class_labels = dict((value, key) for key, value in class_labels.items())
predicted_class_name = invert_class_labels[predicted_class]

print("Predicted class:", predicted_class_name)


#from tenserflow.keras.models import load_model
model.save('/content/drive/My Drive/mini_project/improvedmodel.hdf5')

!pip install gradio
import gradio as gr
import numpy as np
from PIL import Image # Import the Image module for resizing

image = gr.Image()  # No shape parameter here

def predict_image(img):
    """Predicts the class for an uploaded image."""
    if img is None:
        return "No image uploaded"

    # Resize the image using PIL
    img = Image.fromarray(img).resize((128, 128))
    img = np.asarray(img)

    img = img / 255.0  # Rescale
    img = np.expand_dims(img, axis=0)  # Add a batch dimension

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])

    class_labels = test_generator.class_indices
    invert_class_labels = dict((value, key) for key, value in class_labels.items())
    predicted_class_name = invert_class_labels[predicted_class]

    print(f"Predicted class: {predicted_class_name}")
    return predicted_class_name

interface = gr.Interface(fn=predict_image, inputs=image, outputs="text")
interface.launch(debug="True")
