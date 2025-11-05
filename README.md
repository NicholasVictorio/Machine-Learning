# Corrosion Detection Using Deep Learning

## Comparison of Corrosion vs. Non-Corrosion Image Classification Using MobileNetV2, ResNet50, and Basic CNN

Developed an automated image classification system to detect corrosion vs. non-corrosion on metal surfaces using deep learning techniques. The project aimed to provide a faster, more accurate, and consistent alternative to manual visual inspections in industries like construction, oil & gas, and transportation.

## 🎯 Overview

This project implements three deep learning models (Basic CNN, MobileNetV2, and ResNet50) to classify images of metal surfaces as either corroded or non-corroded. The models are trained, validated, and tested on a comprehensive dataset, with ResNet50 achieving the best performance.

## 🗂️ Dataset

The dataset consists of **1,819 labeled images** of corroded and non-corroded metal surfaces, split into:

- **Training Set**: 1,273 images
  - CORROSION: 693 images
  - NOCORROSION: 580 images
- **Validation Set**: 364 images
  - CORROSION: 198 images
  - NOCORROSION: 166 images
- **Testing Set**: 182 images
  - CORROSION: 99 images
  - NOCORROSION: 83 images

## ⚙️ Data Preprocessing

The following preprocessing steps were applied to the images:

- **Resizing**: All images resized to 224×224 pixels
- **Normalization & Rescaling**: Pixel values normalized to [0, 1] range
- **Data Augmentation** (for training set):
  - Rotation: ±20 degrees
  - Zoom: ±10%
  - Width/Height Shift: ±10%
  - Horizontal Flip: Random horizontal flipping

## 🤖 Models Implemented

### 1. Basic CNN (Baseline Model)
A custom convolutional neural network with:
- 3 convolutional layers (32, 64, 128 filters)
- Max pooling layers
- Dense layers for classification
- Binary output (corrosion/no-corrosion)

### 2. MobileNetV2 (Efficient Lightweight Model)
- Pre-trained MobileNetV2 as feature extractor
- Transfer learning approach
- Optimized for mobile and edge devices
- Fast inference with lower computational requirements

### 3. ResNet50 (Deep Residual Network)
- Pre-trained ResNet50 as feature extractor
- Deep residual architecture for high accuracy
- Transfer learning with fine-tuning
- Best performance among all models

## 🧪 Results

**ResNet50** achieved the best performance with:

- **Accuracy**: 95%
- **F1-Score**: 0.96
- **Precision**: 
  - CORROSION: 94%
  - NOCORROSION: 97%
- **Recall**: 
  - CORROSION: 93%
  - NOCORROSION: 98%

### Model Comparison

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Basic CNN | 83% | 0.83 | 83% | 83% |
| MobileNetV2 | 91% | 0.91 | 91% | 91% |
| **ResNet50** | **95%** | **0.96** | **96%** | **95%** |

## 💻 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Libraries

Install the required packages using pip:

```bash
pip install streamlit tensorflow Pillow numpy matplotlib scikit-learn seaborn
```

For Jupyter notebooks, also install:

```bash
pip install jupyter notebook
```

## 🚀 How to Run

### Running the Jupyter Notebooks

1. **Training Models**:
   - Open `Basic_CNN.ipynb`, `MobileNetV2.ipynb`, or `ResNet50.ipynb` in Jupyter Notebook
   - Ensure the dataset is in the correct directory structure (see Dataset section)
   - Run all cells to train the model
   - The trained model will be saved as `.keras` file in the same directory

2. **Model Evaluation**:
   - Each notebook includes evaluation metrics and visualizations
   - Classification reports, confusion matrices, and ROC curves are generated

### Running the Streamlit Web Application

1. **Prepare the Model**:
   - Ensure `model_mobilenetv2_corrosion.keras` is in the same directory as `app.py`
   - The app currently uses MobileNetV2, but you can modify `app.py` to use ResNet50 or Basic CNN

2. **Launch the App**:
   ```bash
   streamlit run app.py
   ```

3. **Using the App**:
   - The browser will automatically open at `http://localhost:8501`
   - View example images in the sidebar (Corrosion and No Corrosion)
   - Upload an image using the file uploader (JPG, JPEG, or PNG format, max 200MB)
   - View the prediction results with confidence score
   - Download the annotated image with prediction overlay

### Loading Saved Models

To load and use a saved model in your own code:

```python
import tensorflow as tf

# Load any of the available models
model = tf.keras.models.load_model('model_resnet50_corrosion.keras')

# Preprocess image
img = tf.keras.preprocessing.image.load_img('image.jpg', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) / 255.0

# Make prediction
prediction = model.predict(img_array)
probability = prediction[0][0]
label = 'CORROSION' if probability > 0.5 else 'NOCORROSION'
```

## 📝 Notes

- The dataset used in this project is a custom compiled dataset from multiple sources
- Pre-trained model weights are downloaded automatically from TensorFlow/Keras when using transfer learning models
- All three model files (Basic CNN, MobileNetV2, and ResNet50) are saved in `.keras` format for easy loading and deployment
- Model files are saved in the same directory where the notebooks are executed
