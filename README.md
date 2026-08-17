# 🌱 AgroTrack — AI-Powered Plant Disease Detection & Monitoring System

AgroTrack is an AI-based agricultural monitoring system designed to help identify **tomato leaf diseases at an early stage** using image classification. The system allows users to capture a leaf image through a camera or upload an existing image, analyze it using a **DenseNet121 deep learning model**, determine the predicted disease and confidence level, and generate an appropriate risk alert.

The system also maintains a log of analyzed images and their results, providing a simple monitoring dashboard through **Streamlit**.

---

## 🎯 Project Objectives

* Detect tomato leaf diseases using deep learning.
* Provide an easy-to-use interface for farmers/users.
* Support both **camera capture** and **image upload**.
* Classify tomato leaves into different disease/health categories.
* Generate a confidence score for each prediction.
* Categorize detected conditions into **Low, Medium, and High risk**.
* Generate alerts based on the detected risk.
* Maintain a record of previous predictions for monitoring and analysis.

---

## 🚀 Key Features

### 📷 Image Capture & Upload

Users can either:

* Capture a tomato leaf image using a laptop camera.
* Upload an existing `.jpg`, `.jpeg`, or `.png` image.

### 🧠 AI-Based Disease Detection

The uploaded/captured image is processed and analyzed using a **DenseNet121** deep learning model.

### 🔍 Disease Classification

The current model supports three classes:

| Class                                   | Description                     |
| --------------------------------------- | ------------------------------- |
| `Tomato_Bacterial_spot`                 | Bacterial Spot detected         |
| `Tomato__Tomato_YellowLeaf__Curl_Virus` | Yellow Leaf Curl Virus detected |
| `Tomato_healthy`                        | Healthy tomato leaf             |

### 📊 Confidence Score

The system displays the confidence associated with the predicted class.

### ⚠️ Risk & Alert Generation

| Leaf Condition         | Risk Level | Alert           |
| ---------------------- | ---------- | --------------- |
| Healthy                | Low        | No Alert        |
| Bacterial Spot         | High       | Immediate Alert |
| Yellow Leaf Curl Virus | High       | Immediate Alert |
| Other/Uncertain        | Medium     | Warning Alert   |

### 📋 Prediction Logging

Each analysis is stored in `logs.csv` with information such as:

* Timestamp
* Image path
* Leaf status
* Predicted class
* Confidence
* Risk level
* Alert
* Model name

### 📈 Monitoring Dashboard

The Streamlit interface displays the current analysis result as well as previously recorded prediction logs.

---

## 🏗️ System Workflow

```text
                 ┌─────────────────────┐
                 │     Leaf Image       │
                 └──────────┬──────────┘
                            │
                  ┌─────────▼─────────┐
                  │ Camera / Upload   │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ Image Preprocessing│
                  │ 224 × 224 RGB     │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │    DenseNet121     │
                  │ Deep Learning Model │
                  └─────────┬─────────┘
                            │
                  ┌─────────▼─────────┐
                  │ Disease Prediction │
                  └─────────┬─────────┘
                            │
             ┌──────────────┴──────────────┐
             │                             │
     ┌───────▼────────┐           ┌────────▼────────┐
     │   Confidence   │           │  Risk Assessment │
     └───────┬────────┘           └────────┬────────┘
             │                             │
             └──────────────┬──────────────┘
                            │
                   ┌────────▼────────┐
                   │ Alert Generation│
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  Save to Logs   │
                   │    logs.csv     │
                   └─────────────────┘
```

---

## 🧠 Deep Learning Model

AgroTrack uses **DenseNet121** for tomato leaf image classification.

---

## 🔄 Image Processing Pipeline

Before prediction, each image goes through the following process:

```text
Input Image
     ↓
Read using OpenCV
     ↓
Convert BGR → RGB
     ↓
Resize to 224 × 224
     ↓
Convert to Float32
     ↓
DenseNet121 Preprocessing
     ↓
Add Batch Dimension
     ↓
Model Prediction
     ↓
Select Highest Probability Class
```

The predicted class is determined using the class with the highest model output probability.

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* DenseNet121

### Computer Vision

* OpenCV
* NumPy

### Web Interface

* Streamlit

### Data Handling

* Pandas
* CSV logging

### Development Tools

* Git
* GitHub
* VS Code

---

## 📁 Project Structure

```text
AgroTrack/
│
├── app.py
├── capture_and_analyze.py
├── alerts.py
│
├── train_densenet.py
├── test_model.py
│
├── densenet_leaf_model.keras
├── labels.json
├── requirements.txt
├── logs.csv
│
├── dataset/
│   ├── Tomato_Bacterial_spot/
│   ├── Tomato__Tomato_YellowLeaf__Curl_Virus/
│   └── Tomato_healthy/
│
└── captured/
```

### File Description

| File                        | Purpose                                                  |
| --------------------------- | -------------------------------------------------------- |
| `app.py`                    | Main Streamlit application and user interface            |
| `capture_and_analyze.py`    | Camera capture, image preprocessing and model prediction |
| `alerts.py`                 | Stores prediction results in `logs.csv`                  |
| `train_densenet.py`         | Trains the DenseNet121 classification model              |
| `test_model.py`             | Tests the trained model using a sample image             |
| `densenet_leaf_model.keras` | Trained DenseNet121 model                                |
| `labels.json`               | Mapping between class indexes and class names            |
| `requirements.txt`          | Python dependencies                                      |
| `logs.csv`                  | Prediction and alert history                             |
| `dataset/`                  | Training and validation images                           |
| `captured/`                 | Captured/uploaded leaf images                            |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AgroTrack.git
cd AgroTrack
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

After selecting an image, AgroTrack analyzes the leaf and displays the prediction, confidence, risk level, alert, and model used.

---

## 🧪 Testing the Model

The repository also includes `test_model.py` for testing an individual leaf image.

Place a sample image in the expected location and run:

```bash
python test_model.py
```

The script loads the trained model, preprocesses the image, performs prediction, and prints the predicted class and confidence.

---

## 🏋️ Training the Model

To train the DenseNet121 model using the dataset:

```bash
python train_densenet.py
```

The training script:

1. Loads images from the dataset directory.
2. Creates training and validation datasets.
3. Resizes images to 224 × 224.
4. Uses an 80/20 training-validation split.
5. Loads DenseNet121 with ImageNet weights.
6. Freezes the pretrained base model.
7. Adds a custom classification head.
8. Trains the model.
9. Uses early stopping and model checkpointing.
10. Saves the trained model as:

```text
densenet_leaf_model.keras
```

and the class mapping as:

```text
labels.json
```

---


For a detected disease, the system can generate a high-risk result and an immediate alert.

---

## 🔔 Alert & Logging System

AgroTrack automatically records prediction results.

This makes it possible to maintain a history of analyzed leaf images and their corresponding predictions.

---

## 🔐 Error Handling

The application includes handling for situations such as:

* Camera unavailable
* Image capture failure
* Invalid image
* Image reading failure
* Prediction errors
* Missing or unreadable log files

This helps make the application more robust during real-world usage.

---

## 🌾 Applications

AgroTrack can be used as a prototype for:

* Smart agriculture systems
* Crop disease monitoring
* Early disease identification
* Agricultural research
* AI-assisted farming
* Plant health monitoring
* Digital agriculture solutions

---

## 🔮 Future Scope

The current system can be extended with:

* 🌱 Support for additional crops and diseases
* 📡 IoT-based soil and environmental sensor integration
* ☁️ Cloud-based data storage
* 📱 Mobile application integration
* 🌡️ Temperature and humidity monitoring

## ⚠️ Limitations

* The current image classification model supports a limited number of tomato leaf classes.
* Prediction quality depends on image quality and the characteristics of the training dataset.
* The current implementation is primarily a prototype for AI-based leaf disease detection.
* Risk levels are rule-based mappings associated with the predicted classes rather than a separate risk-prediction model.
* The current repository implementation does not by itself demonstrate a complete IoT sensor deployment.

---

## 🎓 Project Contribution

AgroTrack combines **computer vision, deep learning, image classification, automated alert generation, and data logging** into a single agricultural monitoring prototype.

The project demonstrates how AI can be applied to assist with early identification of plant health conditions and provide a structured monitoring workflow.

---

## 👨‍💻 Team

**AgroTrack Project**

* **Domain:** Agriculture / Artificial Intelligence
* **Application:** Plant Disease Detection & Monitoring
* **Primary Model:** DenseNet121
* **Interface:** Streamlit

### My Contribution

**Role:** Testing Lead / Project Management

Key responsibilities included:

* Designing and executing test cases
* Testing disease prediction functionality
* Validating image input and prediction results
* Testing confidence and risk-level outputs
* Validating alert generation
* Testing prediction logging
* Identifying and resolving application issues
* Verifying overall system functionality

---

## 📜 License

This project is developed for **academic and educational purposes**.

---

## ⭐ Acknowledgement

This project demonstrates the application of **Artificial Intelligence and Deep Learning in agriculture**, with the goal of making plant disease identification more accessible and supporting data-driven crop monitoring.

If you find this project useful, consider giving the repository a ⭐ on GitHub.
