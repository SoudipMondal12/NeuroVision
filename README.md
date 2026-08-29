# 🧠 NeuroVision — Dementia Prediction Using CNN

NeuroVision is an AI-powered medical imaging application that uses a **Convolutional Neural Network (CNN)** to classify brain MRI images for dementia-related categories.

The project provides an easy-to-use **Streamlit web interface** where users can upload an MRI image and receive an AI-based prediction along with the model's confidence score.

> **⚠️ Medical Disclaimer:** NeuroVision is an educational and research project and is **not intended for medical diagnosis or clinical decision-making**. Predictions should not replace evaluation by a qualified healthcare professional.

---

## 🚀 Features

* 🧠 **Brain MRI Image Classification**
* 🤖 CNN-based deep learning model
* 📤 Upload MRI images directly through the web interface
* 📊 Displays predicted class and confidence score
* 🌐 Interactive Streamlit web application
* ⚡ Fast inference using a trained PyTorch model
* 💻 Simple and user-friendly interface
* 📦 Large PyTorch model managed using **Git LFS**

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Deep Learning

* PyTorch
* Convolutional Neural Networks (CNN)

### Image Processing

* PIL / Pillow
* NumPy

### Web Application

* Streamlit

### Development Tools

* Git
* GitHub
* Git LFS
* VS Code
* Google Colab / Jupyter Notebook

---

## 🧠 How It Works

The NeuroVision pipeline follows these steps:

```text
MRI Image
    ↓
Image Upload
    ↓
Image Preprocessing
    ↓
CNN Model
    ↓
Feature Extraction
    ↓
Classification
    ↓
Predicted Dementia Category
    ↓
Confidence Score
```

The uploaded MRI image is preprocessed into the format expected by the trained CNN model. The model then extracts visual features and predicts the corresponding class.

---

## 📁 Project Structure

```text
NeuroVision/
│
├── app.py
├── best_model.pth
├── requirements.txt
├── .gitattributes
├── .gitignore
├── README.md
│
└── ...
```

### Important Files

| File               | Description                |
| ------------------ | -------------------------- |
| `app.py`           | Main Streamlit application |
| `best_model.pth`   | Trained PyTorch CNN model  |
| `requirements.txt` | Python dependencies        |
| `.gitattributes`   | Git LFS configuration      |
| `.gitignore`       | Files excluded from Git    |
| `README.md`        | Project documentation      |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/SoudipMondal12/NeuroVision.git
```

Navigate into the project:

```bash
cd NeuroVision
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📤 Using NeuroVision

1. Open the NeuroVision Streamlit application.
2. Upload a supported brain MRI image.
3. The image is automatically preprocessed.
4. The trained CNN model performs inference.
5. The predicted dementia category is displayed.
6. The application also provides the model confidence for the prediction.

---

## 🤖 Model

The project uses a trained **PyTorch CNN model** stored in:

```text
best_model.pth
```

Because the trained model is large, it is managed using **Git Large File Storage (Git LFS)** rather than standard Git storage.

To obtain the model when cloning the repository, make sure Git LFS is installed:

```bash
git lfs install
```

Then pull the LFS files:

```bash
git lfs pull
```

---

## 🌐 Streamlit Deployment

NeuroVision can be deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Push the project to GitHub.
2. Make sure `best_model.pth` is tracked with Git LFS.
3. Open Streamlit Community Cloud.
4. Connect your GitHub account.
5. Select:

```text
Repository: SoudipMondal12/NeuroVision
Branch: main
Main file: app.py
```

6. Deploy the application.

Streamlit will install the dependencies from:

```text
requirements.txt
```

and launch the application using:

```text
app.py
```

---

## 📌 Git LFS

The trained model is approximately **300 MB**, so it is stored using Git LFS.

To verify that the model is tracked by Git LFS:

```bash
git lfs ls-files
```

You should see:

```text
best_model.pth
```

When cloning the repository:

```bash
git clone https://github.com/SoudipMondal12/NeuroVision.git
cd NeuroVision
git lfs pull
```

---

## 🔬 Project Goal

The goal of NeuroVision is to demonstrate how **deep learning and computer vision** can be applied to medical image classification.

The project focuses on building an accessible AI interface that connects:

**Medical Imaging → Deep Learning → Image Classification → Interactive Web Application**

---

## 📈 Future Improvements

* Improve model accuracy through additional training and data augmentation
* Experiment with transfer learning using modern CNN architectures
* Add model explainability techniques such as Grad-CAM
* Improve preprocessing and image normalization
* Add prediction history
* Provide more detailed visual analysis
* Optimize inference performance
* Improve deployment scalability

---

## 👨‍💻 Author

**Soudip Mondal**

GitHub: [@SoudipMondal12](https://github.com/SoudipMondal12)

---

## ⭐ Support

If you find this project interesting or useful, consider giving the repository a ⭐ on GitHub.

---

## ⚠️ Disclaimer

NeuroVision is developed for **educational, experimental, and research purposes only**.

It should **not** be used as a substitute for professional medical advice, diagnosis, or treatment. Medical decisions should always be made by qualified healthcare professionals.
