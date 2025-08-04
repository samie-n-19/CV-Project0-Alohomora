# CV-Project0-Alohomora
WPI RBE-549 Course work

CV-Project0-Alohomora
This project is part of my coursework for RBE-549 (Computer Vision) at Worcester Polytechnic Institute. It is divided into two phases, showcasing both classical and deep learning approaches for image analysis.

Phase 1: Boundary Detection with Pb-lite
Implemented a Pb-lite edge detection pipeline, integrating custom DoG, Leung-Malik, and Gabor filter banks (all self-coded).

Extracted texture (texton), brightness, and color features using K-means clustering.

Computed per-pixel gradients with half-disc masks and chi-square distance, combining these with Sobel and Canny baselines for robust boundary maps.

Achieved notable improvement over classic edge detection by leveraging multi-scale, multi-orientation texture and color cues.

Phase 2: Deep Learning for Image Classification
Developed and compared multiple neural network architectures for CIFAR-10:

Custom CNN, ResNet18, ResNeXt50, DenseNet121 (all implemented from scratch, not using prebuilt models).

Applied data augmentation, batch normalization, and learning rate scheduling to improve generalization.

Evaluated models on accuracy, loss, confusion matrices, and number of parameters.

DenseNet121 achieved the best trade-off between accuracy and model size.

Highlights
All key algorithms—including filter bank generation and deep network architectures—are implemented without using prohibited third-party utilities.

Results and findings are detailed in the included report, with visualizations, code structure, and run instructions provided in the repository.

For more, see the Report.pdf and README.md for step-by-step instructions, results, and analysis.
