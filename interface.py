from google.colab import drive
drive.mount('/content/drive')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
my_model = load_model('/content/drive/My Drive/mini_project/improvedmodel.hdf5')
my_model.summary()
!pip install gradio
import gradio as gr
import numpy as np
from PIL import Image
image = gr.Image()  # No shape parameter here
test_dir =  f'/content/drive/My Drive/mini_project/test'
test_datagen = ImageDataGenerator(rescale=1./255)
target_width=128
target_height=128
batch_size=32
test_generator = test_datagen.flow_from_directory(
test_dir,
target_size=(target_width, target_height),
batch_size=batch_size,
class_mode='categorical'
)

#test_loss, test_acc = my_model.evaluate(test_generator)
#print('Test accuracy:', test_acc)
def predict_image(img):
    """Predicts the class for an uploaded image."""
    if img is None:
        return "No image uploaded"

    img = Image.fromarray(img).resize((128, 128))
    img = np.asarray(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = my_model.predict(img)
    predicted_class = np.argmax(predictions[0])
    test_dir =  f'/content/drive/My Drive/mini_project/test'
    test_datagen = ImageDataGenerator(rescale=1./255)
    target_width=128
    target_height=128
    batch_size=32
    test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(target_width, target_height),
    batch_size=batch_size,
    class_mode='categorical'
)

    class_labels = test_generator.class_indices
    invert_class_labels = dict((value, key) for key, value in class_labels.items())
    predicted_class_name = invert_class_labels[predicted_class]

    print(f"Predicted class: {predicted_class_name}")
    return predicted_class_name

interface = gr.Interface(fn=predict_image, inputs=image, outputs="text")
interface.launch(debug="True")
