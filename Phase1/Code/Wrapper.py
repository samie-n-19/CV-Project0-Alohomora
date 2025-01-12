#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def create_gaussian_kernel(size, sigma):
    k = cv2.getGaussianKernel(size, sigma)
    gaussian_kernel = np.outer(k, k)
    return gaussian_kernel

def create_dog_filter(size, sigma1, sigma2):
    g1 = create_gaussian_kernel(size, sigma1)
    g2 = create_gaussian_kernel(size, sigma2)
    dog = g1 - g2
    return dog

def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

def create_log_filter(size, sigma):
    g = create_gaussian_kernel(size, sigma)
    log = cv2.Laplacian(g, cv2.CV_64F)
    return log

def create_derivative_filters(size, sigma, order):
    g = create_gaussian_kernel(size, sigma)
    if order == 1:
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        return gx, gy
    elif order == 2:
        gxx = cv2.Sobel(g, cv2.CV_64F, 2, 0, ksize=3)
        gyy = cv2.Sobel(g, cv2.CV_64F, 0, 2, ksize=3)
        return gxx, gyy

def generate_lm_filter_bank(scales, orientations):
    size = 31  # Size of the filter
    filters = []

    # First and second order derivatives of Gaussians
    for scale in scales:
        for order in [1, 2]:
            gx, gy = create_derivative_filters(size, scale, order)
            for i in range(orientations):
                angle = i * (360 / orientations)
                filters.append(rotate_image(gx, angle))
                filters.append(rotate_image(gy, angle))

    # Laplacian of Gaussian filters
    for scale in scales:
        filters.append(create_log_filter(size, scale))
        filters.append(create_log_filter(size, scale * 3))

    # Gaussian filters
    for scale in scales:
        filters.append(create_gaussian_kernel(size, scale))

    return filters

def create_gabor_kernel(size, sigma, theta, lambd, gamma, psi):
    sigma_x = sigma
    sigma_y = sigma / gamma

    nstds = 3  # Number of standard deviations to include in the kernel
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)

    gabor = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi * rotx / lambd + psi)
    return gabor

def create_half_disk_masks(radius, orientations):
    masks = []
    for orientation in range(orientations):
        angle = np.pi * orientation / orientations
        mask = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)
        cv2.ellipse(mask, (radius, radius), (radius, radius), np.degrees(angle), 0, 180, 1, -1)
        masks.append(mask)
        masks.append(np.flip(mask, axis=1))
    return masks

def generate_texton_map(image, filter_bank):
    responses = []
    for filter in filter_bank:
        response = cv2.filter2D(image, -1, filter)
        responses.append(response)
    responses = np.stack(responses, axis=-1)
    responses = responses.reshape(-1, responses.shape[-1])
    kmeans = KMeans(n_clusters=64).fit(responses)
    texton_map = kmeans.labels_.reshape(image.shape[:2])
    return texton_map

def generate_brightness_map(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray_image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=16).fit(gray_image)
    brightness_map = kmeans.labels_.reshape(image.shape[:2])
    return brightness_map

def generate_color_map(image):
    image_reshaped = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=16).fit(image_reshaped)
    color_map = kmeans.labels_.reshape(image.shape[:2])
    return color_map

def chi_square_distance(map, masks):
    gradients = np.zeros((map.shape[0], map.shape[1], len(masks) // 2))
    for i in range(0, len(masks), 2):
        left_mask = masks[i]
        right_mask = masks[i + 1]
        for bin_val in range(np.max(map) + 1):
            bin_map = (map == bin_val).astype(np.float32)
            g_i = cv2.filter2D(bin_map, -1, left_mask)
            h_i = cv2.filter2D(bin_map, -1, right_mask)
            chi_sqr = ((g_i - h_i) ** 2) / (g_i + h_i + 1e-10)
            gradients[:, :, i // 2] += chi_sqr
    return gradients

def main():

    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png,
    use command "cv2.imwrite(...)"
    """
    size = 31  # Size of the filter
    scales = [1, 2]  # Different scales
    orientations = 16  # Number of orientations

    filters = []
    for scale in scales:
        dog_filter = create_dog_filter(size, scale, scale * 1.6)
        for i in range(orientations):
            angle = i * (360 / orientations)
            rotated_filter = rotate_image(dog_filter, angle)
            filters.append(rotated_filter)

    # Display and save the filters
    fig, axes = plt.subplots(len(scales), orientations, figsize=(20, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[i], cmap='gray')
        ax.axis('off')
    plt.savefig('DoG.png')
    plt.show()

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png,
    use command "cv2.imwrite(...)"
    """

    scales_lms = [1, np.sqrt(2), 2, 2 * np.sqrt(2)]
    scales_lml = [np.sqrt(2), 2, 2 * np.sqrt(2), 4]
    orientations = 6

    filters_lms = generate_lm_filter_bank(scales_lms, orientations)
    filters_lml = generate_lm_filter_bank(scales_lml, orientations)

    # Display and save the filters
    fig, axes = plt.subplots(8, 6, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters_lms[i], cmap='gray')
        ax.axis('off')
    plt.savefig('LMS.png')
    plt.show()

    fig, axes = plt.subplots(8, 6, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters_lml[i], cmap='gray')
        ax.axis('off')
    plt.savefig('LML.png')
    plt.show()
    
    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    
    size = 31  # Size of the filter
    scales = [1, 2, 3]  # Different scales
    orientations = 8  # Number of orientations
    lambd = 10  # Wavelength of the sinusoidal factor
    gamma = 0.5  # Spatial aspect ratio
    psi = 0  # Phase offset

    filters = []
    for scale in scales:
        sigma = scale
        for i in range(orientations):
            theta = i * (np.pi / orientations)
            gabor_filter = create_gabor_kernel(size, sigma, theta, lambd, gamma, psi)
            filters.append(gabor_filter)

    # Display and save the filters
    fig, axes = plt.subplots(len(scales), orientations, figsize=(20, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[i], cmap='gray')
        ax.axis('off')
    plt.savefig('Gabor.png')
    plt.show()


    image_path = '/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/Images/1.jpg'  
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Generate Half-disk masks
    radius = 15
    orientations = 8
    masks = create_half_disk_masks(radius, orientations)
    
    # Display and save Half-disk masks
    fig, axes = plt.subplots(2, orientations, figsize=(20, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(masks[i], cmap='gray')
        ax.axis('off')
    plt.savefig('HDMasks.png')
    plt.show()

    # Generate Texton Map
    filter_bank = [create_gaussian_kernel(31, sigma) for sigma in [1, 2, 3]]
    texton_map = generate_texton_map(gray_image, filter_bank)
    
    # Display and save Texton Map
    plt.imshow(texton_map, cmap='nipy_spectral')
    plt.axis('off')
    plt.savefig('TextonMap_ImageName.png')
    plt.show()

    # Generate Brightness Map
    brightness_map = generate_brightness_map(image)
    
    # Generate Brightness Gradient (Bg)
    brightness_gradient = chi_square_distance(brightness_map, masks)
    
    # Display and save Brightness Gradient
    plt.imshow(np.sum(brightness_gradient, axis=-1), cmap='hot')
    plt.axis('off')
    plt.savefig('Bg_ImageName.png')
    plt.show()

    # Generate Color Map
    color_map = generate_color_map(image)
    
    # Generate Color Gradient (Cg)
    color_gradient = chi_square_distance(color_map, masks)
    
    # Display and save Color Gradient
    plt.imshow(np.sum(color_gradient, axis=-1), cmap='hot')
    plt.axis('off')
    plt.savefig('Cg_ImageName.png')
    plt.show()

    """
    Read Sobel Baseline
    use command "cv2.imread(...)"
    """
    sobel_baseline = cv2.imread('/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/SobelBaseline/1.png', cv2.IMREAD_GRAYSCALE)
    if sobel_baseline is None:
        raise FileNotFoundError("Sobel baseline image not found.")
    
    """
    Read Canny Baseline
    use command "cv2.imread(...)"
    """
    canny_baseline = cv2.imread('/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/CannyBaseline/1.png', cv2.IMREAD_GRAYSCALE)
    if canny_baseline is None:
        raise FileNotFoundError("Canny baseline image not found.")
    
    # Normalize baselines
    sobel_baseline = sobel_baseline / 255.0
    canny_baseline = canny_baseline / 255.0
    
    """
    Combine responses to get pb-lite output
    Display PbLite and save image as PbLite_ImageName.png
    use command "cv2.imwrite(...)"
    """
    Tg = np.sum(chi_square_distance(texton_map, masks), axis=-1)
    Bg = np.sum(brightness_gradient, axis=-1)
    Cg = np.sum(color_gradient, axis=-1)
    
    feature_strength = (Tg + Bg + Cg) / 3.0
    w1, w2 = 0.5, 0.5
    pb_lite = feature_strength * (w1 * canny_baseline + w2 * sobel_baseline)
    pb_lite = cv2.threshold(pb_lite, 0.5, 1, cv2.THRESH_BINARY)[1]
    
    # Display and save PbLite
    plt.imshow(pb_lite, cmap='grey')
    plt.axis('off')
    plt.savefig('PbLite_ImageName.png')
    plt.show()
    print("PbLite image saved as PbLite_ImageName.png")

if __name__ == '__main__':
    main()