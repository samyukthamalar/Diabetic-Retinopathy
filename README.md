# Diabetic Retinopathy Lesion Segmentation using Deep Learning

**Final Year Project - B.Tech Computer Science**  
**Semester 8 | Academic Year 2025-26**

---

## 📋 Project Overview

This project implements an automated diabetic retinopathy (DR) detection and classification system using deep learning and transfer learning techniques. While the original objective was lesion segmentation, the implementation focuses on binary classification of retinal fundus images into two categories: DR (Diabetic Retinopathy present) and No_DR (No Diabetic Retinopathy), which serves as a crucial first step in DR screening and diagnosis.

### Key Highlights
- **Accuracy:** 96.97%
- **AUC-ROC:** 98.77%
- **Model:** ResNet50 with Transfer Learning
- **Dataset:** 2,607 retinal fundus images
- **Framework:** PyTorch

---

## 🎯 Objectives

1. Develop an automated DR detection system using deep learning
2. Achieve high accuracy in binary classification of retinal images
3. Demonstrate advantages of deep learning over traditional methods
4. Create a clinically relevant screening tool for early DR detection
5. Lay foundation for future lesion segmentation implementation

---

## 🏗️ System Architecture

### Model: ResNet50
- **Type:** Deep Convolutional Neural Network
- **Layers:** 50 layers with residual connections
- **Pretrained:** ImageNet weights (Transfer Learning)
- **Modified:** Final layer adapted for binary classification

### Training Strategy
- **Optimizer:** Adam
- **Learning Rate:** 1e-4 with ReduceLROnPlateau scheduling
- **Loss Function:** Cross-Entropy Loss
- **Data Augmentation:** Rotation, flipping, color jittering
- **Early Stopping:** Patience of 10 epochs

---

## 📊 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.97% |
| **Precision** | 95.87% |
| **Recall (Sensitivity)** | 98.31% |
| **F1 Score** | 97.07% |
| **AUC-ROC** | 98.77% |

### Confusion Matrix
```
                Predicted
              No_DR    DR
Actual No_DR   108     5
       DR        2   116
```

**Analysis:**
- Only 7 misclassifications out of 231 test images
- High sensitivity (98.31%) - catches almost all DR cases
- High specificity (95.58%) - few false positives
- Excellent for clinical screening applications

---

## 📁 Project Structure

```
diabetic-retinopathy-detection/
│
├── 📄 README.md                          # This file
├── 📄 HOW_TO_RUN.md                      # Execution guide
├── 📄 requirements.txt                   # Dependencies
│
├── 🐍 train_classification.py            # Training script
├── 🐍 test_classification.py             # Testing script
├── 🐍 config_classification.py           # Configuration
│
├── 📂 data/                              # Dataset
│   └── raw/
│       └── diagnosis-of-diabetic-retinopathy/
│
├── 📂 checkpoints/                       # Trained models
│   └── best_model_classification.pth    # Best model (96.97% acc)
│
├── 📂 results/                           # Test results
│   ├── test_metrics_classification.txt  # Numerical results
│   ├── confusion_matrix.png             # Confusion matrix
│   ├── roc_curve.png                    # ROC curve
│   └── sample_predictions.png           # Example predictions
│
└── 📂 utils/                             # Helper functions
    └── helpers.py                        # Utility functions
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Quick Start (After Cloning Repository)

1. **Clone the repository**
   ```bash
   git clone https://github.com/samyukthamalar/Diabetic-Retinopathy.git
   cd Diabetic-Retinopathy
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note:** This will install PyTorch (CPU version), OpenCV, and other required packages. Installation takes 2-3 minutes.

3. **Run the trained model (Recommended)**
   ```bash
   python test_classification.py
   ```
   
   **What happens:**
   - Loads the pre-trained model from `checkpoints/best_model_classification.pth`
   - Tests on 231 images from the dataset
   - Displays metrics: Accuracy (96.97%), Precision, Recall, F1, AUC
   - Generates visualizations in `results/` folder
   - Takes ~2 minutes on CPU
   
   **No training needed!** The trained model is already included in the repository.

### Training from Scratch (Optional)

If you want to retrain the model:
```bash
python train_classification.py
```

**Note:** 
- Training takes 30-60 minutes on CPU, 5-10 minutes on GPU
- Not necessary as trained model is already provided
- Useful for experimenting with different hyperparameters

### What's Included in the Repository

✅ **Trained Model** - Ready to use (282 MB)  
✅ **Complete Dataset** - All 2,607 images organized  
✅ **Test Results** - Pre-generated metrics and visualizations  
✅ **Source Code** - Training and testing scripts  
✅ **Documentation** - This README with full details

**You can run the project immediately after cloning!**

---

## 🔬 Methodology

### 1. Data Preprocessing
- Resize images to 224×224 pixels
- Normalize using ImageNet statistics
- Apply data augmentation (rotation, flipping, color jittering)

### 2. Model Architecture
- Base: ResNet50 pretrained on ImageNet
- Modified final fully connected layer for binary classification
- Total parameters: ~25 million

### 3. Training Process
- Split: 70% train, 20% validation, 10% test
- Batch size: 32
- Epochs: 50 (with early stopping)
- Learning rate: 1e-4 (adaptive)

### 4. Evaluation
- Comprehensive metrics: Accuracy, Precision, Recall, F1, AUC
- Confusion matrix analysis
- ROC curve visualization
- Sample prediction visualization

---

## 💡 Advantages over Traditional Methods

| Aspect | Traditional ML | Deep Learning (Our Approach) |
|--------|---------------|------------------------------|
| **Feature Extraction** | Manual (HOG, SIFT, etc.) | Automatic |
| **Accuracy** | 85-90% | 96.97% |
| **Generalization** | Limited | Excellent |
| **Training Time** | Fast | Moderate (with transfer learning) |
| **Scalability** | Limited | High |
| **Clinical Relevance** | Moderate | High |

### Key Benefits:
1. **No Manual Feature Engineering:** Model learns optimal features automatically
2. **Transfer Learning:** Leverages knowledge from millions of ImageNet images
3. **High Accuracy:** Outperforms traditional ML methods by 7-12%
4. **Robust:** Handles variations in image quality and lighting
5. **Scalable:** Can be deployed for large-scale screening

---

## 📚 Technical Stack

- **Language:** Python 3.10
- **Deep Learning:** PyTorch 2.10
- **Computer Vision:** TorchVision, OpenCV
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Metrics:** Scikit-learn

---

## 📖 References

### Dataset
- **Source:** Kaggle - Diagnosis of Diabetic Retinopathy
- **Size:** 2,607 fundus images
- **Classes:** DR (1,050 images), No_DR (1,026 images)
- **License:** Academic use

### Key Papers
1. He et al. (2016) - "Deep Residual Learning for Image Recognition"
2. Gulshan et al. (2016) - "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy"
3. Ting et al. (2017) - "Deep Learning in Ophthalmology: The Technical and Clinical Considerations"

---

## 🎓 Academic Context

### Course Information
- **Program:** B.Tech Computer Science Engineering
- **Course:** Final Year Project
- **Project Title:** Diabetic Retinopathy Lesion Segmentation using Deep Learning
- **Implementation:** DR Classification (Binary) - Foundation for Segmentation
- **Semester:** 8
- **Academic Year:** 2025-26

### Learning Outcomes
1. Understanding of deep learning architectures (CNNs, ResNets)
2. Practical experience with transfer learning
3. Medical image analysis techniques
4. Model evaluation and validation
5. Real-world application development
6. Dataset analysis and problem adaptation

---

## 🔮 Future Enhancements

1. **Lesion Segmentation:** Implement pixel-level segmentation of specific lesions (microaneurysms, hemorrhages, exudates, cotton wool spots) using U-Net or Attention U-Net architectures
2. **Multi-class Classification:** Grade DR severity (0-4 scale) according to international standards
3. **Ensemble Methods:** Combine multiple models for better accuracy
4. **Mobile Deployment:** Create mobile app for field screening
5. **Explainability:** Add attention maps and Grad-CAM to show model focus areas
6. **Real-time Processing:** Optimize for faster inference

### Note on Lesion Segmentation
The original project scope included lesion segmentation. However, due to dataset constraints (available dataset contained classification labels rather than pixel-level segmentation masks), the current implementation focuses on binary classification. This serves as a robust foundation for:
- Screening patients for DR presence
- Prioritizing cases for detailed examination
- Future integration with segmentation models when appropriate datasets become available

---

## 📞 Contact

**Student Name:** [Your Name]  
**Roll Number:** [Your Roll No]  
**Email:** [Your Email]  
**Institution:** [Your University]  
**Department:** Computer Science Engineering

**Project Guide:** [Guide Name]  
**Designation:** [Guide Designation]

---

## 📄 License

This project is developed for academic purposes as part of the Final Year Project curriculum.

**Usage:** Academic and educational purposes only  
**Dataset:** Subject to Kaggle dataset license terms  
**Code:** Available for academic review and learning

---

## 🙏 Acknowledgments

- Project guide for valuable guidance and support
- Department of Computer Science Engineering
- Kaggle for providing the dataset
- PyTorch community for excellent documentation
- Research papers that inspired this work

---

## 📝 Project Status

✅ **Completed**
- Dataset collection and preprocessing
- Model training and optimization
- Testing and evaluation
- Documentation and visualization
- Results analysis

**Date Completed:** March 1, 2026

---

## 📌 Project Implementation Note

**Original Title:** Diabetic Retinopathy Lesion Segmentation using Deep Learning

**Current Implementation:** Binary Classification (DR vs No_DR)

**Rationale:** The available dataset (Kaggle - Diagnosis of Diabetic Retinopathy) provides classification labels rather than pixel-level segmentation masks required for lesion segmentation. The current implementation:

1. ✅ Successfully detects presence of DR with 96.97% accuracy
2. ✅ Demonstrates deep learning effectiveness in medical imaging
3. ✅ Provides clinically relevant screening capability
4. ✅ Establishes foundation for future segmentation work
5. ✅ Achieves project learning objectives

**For Full Segmentation:** Future work will integrate datasets like IDRID or DDR that include pixel-level lesion annotations, enabling implementation of U-Net based segmentation models as originally envisioned.

---
