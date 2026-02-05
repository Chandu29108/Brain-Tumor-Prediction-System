# Brain Tumor Detection Using Machine Learning

This project presents an end-to-end **machine learning-based system** for detecting brain tumors from MRI images. The model classifies images into two categories:

* **No Tumor**
* **Pituitary Tumor**

The system leverages classical machine learning techniques with image preprocessing and dimensionality reduction to achieve high classification performance.

## Objective

To build a reliable, interpretable, and efficient classification pipeline that can assist medical professionals in identifying brain tumors from MRI scans using machine learning.

## Methodology

1. **Data Preprocessing**

   * Converted images to grayscale
   * Resized all images to **200 × 200 pixels**
   * Flattened images into feature vectors

1) **Feature Engineering**

   * Standardized pixel values using **StandardScaler**
   * Applied **PCA (98% variance retained)** for dimensionality reduction

2) **Model Training**

   * Trained and compared:

     * Logistic Regression
     * Support Vector Machine (SVM with RBF kernel)

3) **Model Evaluation**

   * Used:

     * Accuracy
     * Precision
     * Recall
     * F1-score
     * Confusion Matrix

## Results

| Model               | Training Accuracy | Testing Accuracy |
| ------------------- | ----------------- | ---------------- |
| Logistic Regression | 1.0000            | **0.9986**       |
| SVM (RBF Kernel)    | 0.9993            | **0.9986**       |

### Selected Model: **Support Vector Machine (SVM)**

SVM was chosen due to:

* Strong generalization ability
* Better handling of non-linear patterns in MRI images
* Robust decision boundary via maximum-margin principle

## Tech Stack

* **Python**
* **NumPy, Pandas**
* **OpenCV**
* **Scikit-learn**
* **PCA**
* **Logistic Regression**
* **SVM**
* **Matplotlib, Seaborn**

## Dataset

The dataset contains MRI brain images organized into:

Data/
│── Training/
│   ├── notumor/
│   └── pituitary/
│
│── Testing/
│   ├── notumor/
│   └── pituitary/

## How to Run the Project

### Step 1 — Install dependencies

pip install -r requirements.txt

### Step 2 — Train the model

Run your training script:

python train.ipynb

This will generate:

* `svm_model.pkl`
* `scaler.pkl`
* `pca.pkl`

> **Note:** Trained model pca.pkl file is not uploaded to GitHub due to size limits. Please generate them locally.

## Future Enhancements

* Implement **CNN-based deep learning model**
* Add **Grad-CAM visualization for explainability**
* Deploy as a **web app using Streamlit or Flask**
* Extend to multi-class tumor classification

## Author

**Chandu Vanja**
B.Tech (ECE), NIT Durgapur


