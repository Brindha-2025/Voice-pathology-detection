# Voice-pathology-detection
Voice Pathology Detection Using Scalogram Images and Hybrid Deep Learning Network

This repository provides the MATLAB implementation for detecting pathological voices using a DNN-SVM hybrid model trained on scalogram images derived from:

- Speech signals  
- Electroglottograph (EGG) signals  
- Combined speech + EGG signals  

Scalograms are generated using Continuous Wavelet Transform (CWT), providing rich time-frequency representations of voice signals. A Deep Neural Network (DNN) extracts features, and a Support Vector Machine (SVM) performs the final classification.

Early detection of vocal disorders is crucial for diagnosis and treatment. This research presents a MATLAB-based system for automated voice pathology detection using:

- Scalogram images for feature-rich signal representation  
- Deep Neural Network (DNN) for feature learning  
- Support Vector Machine (SVM) for classification  


  - MATLAB R2022a or newer
  - Toolboxes Required:
  - Deep Learning Toolbox
  - Signal Processing Toolbox
  - Wavelet Toolbox
  - Statistics and Machine Learning Toolbox
  - Parallel Computing Toolbox for faster training

Saarbruecken voice database. Available from: http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4.
