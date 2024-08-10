### Project Summary: Image Classification Model with Gradio Interface

This project focuses on building and deploying a Convolutional Neural Network (CNN) for multi-class image classification. The model is trained on a dataset of images stored in Google Drive, where it learns to categorize images into 36 distinct classes. The project leverages TensorFlow and Keras libraries for model building, training, and evaluation, with data preprocessing handled by the `ImageDataGenerator` class. The model includes three convolutional layers, each followed by batch normalization, max pooling, and dropout to enhance performance and prevent overfitting.

After training, the model's performance is evaluated on a test dataset. The final trained model is saved to Google Drive for future use. Additionally, a Gradio interface is implemented to allow users to upload images and receive real-time predictions based on the trained model. This project is suitable for use cases requiring quick and accurate image classification, such as in medical imaging, automated quality control, or other areas requiring image-based predictions.

### Key Features:
- **CNN Architecture**: Three convolutional layers with batch normalization and dropout for improved accuracy and reduced overfitting.
- **Data Preprocessing**: Automated rescaling and batching of images using TensorFlow’s `ImageDataGenerator`.
- **Model Training**: Includes model checkpoints for saving progress and avoiding data loss.
- **Real-Time Prediction**: A Gradio-powered web interface for interactive image classification.
- **Deployment-Ready**: The final model is saved in HDF5 format, enabling easy deployment and further fine-tuning.
