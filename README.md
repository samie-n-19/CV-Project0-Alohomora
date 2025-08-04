# CV-Project0-Alohomora

_WPI RBE-549 Coursework | Computer Vision_

This repository contains my solutions for Project 0 ("Alohomora") as part of the RBE-549 Computer Vision course at Worcester Polytechnic Institute. The project is divided into two major phases, showcasing both classical and deep learning approaches for image analysis.

---

## Phase 1: Boundary Detection with Pb-lite

- **Implemented a Pb-lite edge detection pipeline** by integrating custom Difference of Gaussian (DoG), Leung-Malik (LM), and Gabor filter banks (all filters coded from scratch).
- **Extracted texture (texton), brightness, and color features** using K-means clustering to generate separate feature maps.
- **Computed per-pixel gradients** via half-disc masks and chi-square distance, merging these with Sobel and Canny edge detection baselines for robust boundary maps.
- **Achieved improved boundary detection** compared to classical methods by leveraging multi-scale and multi-orientation texture and color information.

---

## Phase 2: Deep Learning for Image Classification

- **Developed and evaluated several neural network architectures for CIFAR-10 image classification:**
  - Custom CNN, ResNet18, ResNeXt50, and DenseNet121 (all implemented from scratch; no use of built-in models).
- **Applied advanced training strategies** including data augmentation, batch normalization, and learning rate scheduling for better generalization.
- **Assessed each model** on training/testing accuracy, loss curves, confusion matrices, parameter counts, and inference time.
- **DenseNet121 yielded the best trade-off** between model accuracy and parameter efficiency.

---

## Highlights

- All key components—filter bank generation, Pb-lite algorithm, and CNN architectures—are implemented without prohibited third-party tools or pre-built utilities.
- The project is modular, well-documented, and contains visualizations, code structure, and clear run instructions.

---

## 📄 Documentation & Results

- **See `Report.pdf`** for detailed technical explanations, algorithm visualizations, results, and performance analysis.
- **See `README.md`** for project structure and run instructions.

---

