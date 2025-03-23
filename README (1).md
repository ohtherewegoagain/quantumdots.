# Quantum Dots Neuroimaging Project

This project implements a computational framework for **Quantum Dot (QD) neuroimaging**, 
integrating **AI-driven fluorescence analysis, quantum chemistry modeling, and toxicity prediction**.

## 📂 Files Overview

- **[`quantum_dots_full_code.py`](./quantum_dots_full_code.py)** → Full computational implementation.
- **[`quantum_dots_large_data.csv`](./quantum_dots_large_data.csv)** → Large dataset (~5000 samples) for QD toxicity analysis.

## 🛠 Features Implemented

1. **Quantum Dot Stability & Biodegradability Modeling**  
   - AIMD simulations using **ReaxFF in LAMMPS**.
   - Reaction-Diffusion equation for degradation modeling.

2. **AI-Based Fluorescence Analysis**  
   - **Wavelet Transform Filtering** for noise removal.  
   - **GAN-based Super-Resolution** for fluorescence images.  
   - **Vision Transformer (ViT) Model** for real-time fluorescence classification.

3. **Machine Learning-Based Toxicity Prediction**  
   - **XGBoost & Random Forest Models** trained on real QD toxicity data.
   - **GNN extension (future implementation) for molecular interactions**.

4. **EEG Brainwave Classification**  
   - **CNN + LSTM Model** for detecting neural states from EEG signals.
   - **Preprocessing: Bandpass Filtering, ICA, Z-score normalization**.

## ⚡ Usage Instructions

### 1️⃣ Install Dependencies  
```bash
pip install numpy pandas torch torchvision transformers scikit-learn ase lammps librosa scipy seaborn matplotlib scikit-image
```

### 2️⃣ Run the Code  
To execute the **Quantum Dot Stability Simulation**, use:  
```bash
python quantum_dots_full_code.py
```

### 3️⃣ Explore the Dataset  
Load and visualize the dataset in Python:  
```python
import pandas as pd
df = pd.read_csv("quantum_dots_large_data.csv")
print(df.head())
```

## 🔬 Future Work
- **Expand AIMD simulations** with temperature-dependent QD degradation.  
- **Develop GNN-based QD toxicity prediction model**.  
- **Deploy fluorescence AI model for real-world neuroimaging applications**.

---  
🚀 **Developed for advancing AI-powered quantum dot neuroimaging research!**  
