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
from scipy.ndimage import rotate
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

def create_gaussian_kernel(size, sigma):
    k = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(k, k)
    gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    gaussian_kernel /= np.sum(gaussian_kernel)
    return gaussian_kernel

def sobel_kernel():
    sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return sobel_kernel

def create_dog_filter(size, sigma1, sigma2):
    G1 = create_gaussian_kernel(size, sigma1) 
    G2 = create_gaussian_kernel(size, sigma2)
    dog1 = cv2.filter2D(G1, -1, sobel_kernel())
    dog2 = cv2.filter2D(G2, -1, sobel_kernel())
    dog = []
    for angle in np.linspace(0, 360, 16):
        center_point = (dog1.shape[0] // 2, dog1.shape[1] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, 1.0)
        dog1_rotated = cv2.warpAffine(dog1, rotation_matrix, dog1.shape)
        dog.append(dog1_rotated)

    for angle in np.linspace(0, 360, 16):
        center_point = (dog2.shape[0] // 2, dog2.shape[1] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center_point, angle, 1.0)
        dog2_rotated = cv2.warpAffine(dog2, rotation_matrix, dog2.shape)
        dog.append(dog2_rotated)             
    return dog

def gaussian(size, scale):
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)
    g = np.exp(-(X**2 + Y**2) / (2 * scale**2))/ (2 * np.pi * (scale**2))
    return g / g.sum()

def first_derivative(size, scale):
    x = np.linspace(-size // 2, size // 2, size)
    gx = -(x / scale**2) * np.exp(-x**2 / (2 * scale**2)) / (np.sqrt(2 * np.pi) * scale)
    return gx / np.abs(gx).sum()

def second_derivative(size, scale):
    x = np.linspace(-size // 2, size // 2, size)
    gxx = ((x**2 - scale**2) / scale**4) * np.exp(-x**2 / (2 * scale**2))/((np.sqrt(2 * np.pi) * scale))
    return gxx / np.abs(gxx).sum()

def laplacian_of_gaussian(size, scale):
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2
    log = -(1 / (np.pi * scale**4)) * (1 - r2 / (2 * scale**2)) * np.exp(-r2 / (2 * scale**2))
    return log / np.abs(log).sum()

def generate_filters(scales, orientations, size=49, elongation=3):
    filters = []

    for scale in scales:
        # Generate 1D Gaussian derivatives
        y = np.linspace(-size // 2, size // 2, size)
        x,y=np.meshgrid(y,y)
        gx = first_derivative(size, scale)
        gxx = second_derivative(size, scale)
        gy =  np.exp(-y**2 / (2 * (3*scale)**2)) / (np.sqrt(2 * np.pi) * (3*scale))
        gy= gy / np.abs(gy).sum()

        # Create 2D filters using the outer product
        gx_2d = gx*gy
        gxx_2d = gxx*gy

        # Generate oriented filters by rotating the 2D convolutions
        for orientation in range(orientations):
            angle = orientation * (180.0 / orientations)
            gx_rot = rotate(gx_2d, angle, reshape=False, order=1, mode='constant', cval=0)
            gxx_rot = rotate(gxx_2d, angle, reshape=False, order=1, mode='constant', cval=0)

            # Ensure all rotated filters are 2D with the proper shape
            if gx_rot.shape != (size, size):
                gx_rot = cv2.resize(gx_rot, (size, size))
            if gxx_rot.shape != (size, size):
                gxx_rot = cv2.resize(gxx_rot, (size, size))

            filters.append(gx_rot)
            filters.append(gxx_rot)

    return filters


def lm_filter_bank(filter_type="LMS"):
    if filter_type == "LMS":
        scales = [1, np.sqrt(2), 2, 2 * np.sqrt(2)]  # LMS scales 
    elif filter_type == "LML":
        scales = [np.sqrt(2), 2, 2 * np.sqrt(2), 4]  # LML scales 
    else:
        raise ValueError("Invalid filter_type. Choose 'LMS' or 'LML'.")

    orientations = 6
    size = 49  # Filter size

    # Generate first and second derivative filters (36 filters)
    derivative_filters = generate_filters(scales[:3], orientations, size=size, elongation=3)

    # Generate Laplacian of Gaussian (LoG) filters (8 filters)
    log_filters = []
    for scale in scales:  # 4 scales
        log_filters.append(laplacian_of_gaussian(size, scale))       # Original scale
        log_filters.append(laplacian_of_gaussian(size, 3 * scale))   # Scaled by 3

    # Generate Gaussian filters (4 filters)
    gaussian_filters = [gaussian(size, scale) for scale in scales]  # 4 scales

    # Combine all filters
    filter_bank = derivative_filters + log_filters + gaussian_filters
    return filter_bank


def display_filters(filter_bank, title="Filter Bank", n_rows=4, n_cols=12, save_path="/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/Code"):
    """Display filters in a filter bank in a 4-row, 12-column grid."""
    n_filters = len(filter_bank)

    plt.figure(figsize=(15, 15))  # Adjust the figure size if necessary
    for i, f in enumerate(filter_bank):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(f / np.max(np.abs(f)), cmap='gray') 
        plt.axis('off')
    plt.suptitle(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
  
def create_gabor_kernel(size, sigma, theta, lambd, gamma, psi):

    # Calculate the dimensions of the Gaussian function
    sigma_x = sigma
    sigma_y = sigma / gamma  # Apply the spatial aspect ratio
    x, y = np.meshgrid(np.linspace(-size // 2, size // 2, size), np.linspace(-size // 2, size // 2, size))

    # Apply the rotation to the filter
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)

    # Gabor filter equation (Gaussian multiplied by sinusoidal wave)
    gabor = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi * rotx / lambd + psi)
    return gabor

def generate_gabor_filter_bank(scales, orientations):
    size = 49  # Size of the filter (31x31)
    gamma = 0.5  # Spatial aspect ratio
    psi = 0  # Phase offset of the sinusoid (0 or pi)

    gabor_filters = []

    # Loop through each scale and orientation to generate the Gabor filters
    for scale in scales:
        sigma = scale  # The standard deviation of the Gaussian function
        lambd = 1.25 * sigma
        for i in range(orientations):
            theta = i * (np.pi / orientations)  # Calculate the angle for orientation
            gabor_filter = create_gabor_kernel(size, sigma, theta, lambd, gamma, psi)
            gabor_filters.append(gabor_filter)
    
    return gabor_filters

def create_half_disc_mask(size, scale, theta, is_left=True):
    # Define the center of the image
    center = size // 2
    y, x = np.indices((size, size))
    
    # Distance from center
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    
    # Create a binary mask for a full disc (radius = scale)
    mask = (distance <= scale).astype(float)
    
    # Rotate the mask according to theta
    x_rot = (x - center) * np.cos(theta) + (y - center) * np.sin(theta)
    y_rot = -(x - center) * np.sin(theta) + (y - center) * np.cos(theta)
    
    # Create the left or right half-disc mask based on rotation and orientation
    if is_left:
        half_disc_mask = mask * (x_rot <= 0)
    else:
        half_disc_mask = mask * (x_rot >= 0)
    
    return half_disc_mask

def generate_half_disc_bank(scales, orientations, size=31):
    half_disc_masks = []
    
    for scale in scales:
        for i in range(orientations):
            # Calculate the orientation in radians
            theta = i * (np.pi / orientations)
            
            # Generate the left and right half-disc masks for each orientation
            left_mask = create_half_disc_mask(size, scale, theta, is_left=True)
            right_mask = create_half_disc_mask(size, scale, theta, is_left=False)
            
            # Append the left and right masks
            half_disc_masks.append(left_mask)
            half_disc_masks.append(right_mask)
    
    return half_disc_masks

def display_half_discs(half_disc_masks, n_cols=8, save_path="/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/Code"):
    n_filters = len(half_disc_masks)
    n_rows = (n_filters + n_cols - 1) // n_cols  

    plt.figure(figsize=(15, 15))
    for i, mask in enumerate(half_disc_masks):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
    
    plt.suptitle("Half-Disc Mask Bank", fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def generate_texton_map(image, filter_bank):
    responses = []
    for filter in filter_bank:
        response = convolve2d(image, filter, mode='same')
        responses.append(response)
    responses = np.stack(responses, axis=-1)
    responses = responses.reshape(-1, responses.shape[-1])
    kmeans = KMeans(n_clusters=64).fit(responses)
    texton_map = kmeans.labels_.reshape(image.shape[:2])
    return texton_map

def generate_brightness_map(image):
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    gray_image = gray_image.reshape(-1, 1)
    kmeans = KMeans(n_clusters=16).fit(gray_image)
    brightness_map = kmeans.labels_.reshape(image.shape[:2])
    return brightness_map

def generate_color_map(image):
    image_reshaped = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=16).fit(image_reshaped)
    color_map = kmeans.labels_.reshape(image.shape[:2])
    return color_map

def chi_square_distance(img, masks, num_bins):
    height, width = img.shape
    num_orientations = len(masks) // 2  # Each orientation has two masks (left and right)
    gradients = np.zeros((height, width, num_orientations), dtype=np.float32)

    # Iterate through each pair of masks
    for i in range(0, len(masks), 2):
        left_mask = masks[i]
        right_mask = masks[i + 1]
        orientation_idx = i // 2  # Index for storing gradients for this orientation

        # Iterate through bins in the map
        for bin_val in range(num_bins):
            # Create a binary map for the current bin
            bin_map = (img == bin_val).astype(np.float32)

            # Compute histogram values for left and right masks using cv2.filter2D
            g_i = cv2.filter2D(bin_map, -1, left_mask, borderType=cv2.BORDER_REFLECT)
            h_i = cv2.filter2D(bin_map, -1, right_mask, borderType=cv2.BORDER_REFLECT)

            # Compute χ² distance for this bin and update gradients
            chi_sqr = ((g_i - h_i) ** 2) / (g_i + h_i + 1e-10)  # Add small value to avoid division by zero
            gradients[:, :, orientation_idx] += chi_sqr

    return gradients


def main():
    """
    Generate Difference of Gaussian Filter Bank: (DoG)
    Display all the filters in this filter bank and save image as DoG.png
    """
    size = 31  # Size of the filter
    scales = [1, 2]  # Different scales

    dog_filters = []
    for scale in scales:
        dog_filter = create_dog_filter(size, scale, scale * 1.6)
        dog_filters.extend(dog_filter)

    # Display and save the filters
    fig, axes = plt.subplots(len(scales), 16, figsize=(20, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(dog_filter[i], cmap='gray')
        ax.axis('off')
    plt.savefig('DoG.png')
    plt.show()

    """
    Generate Leung-Malik Filter Bank: (LM)
    Display all the filters in this filter bank and save image as LM.png
    """
    # Generate LMS filter bank
    lms_filters = lm_filter_bank(filter_type="LMS")
    print(f"Total filters in LMS: {len(lms_filters)}")

    # Generate LML filter bank
    lml_filters = lm_filter_bank(filter_type="LML")
    print(f"Total filters in LML: {len(lml_filters)}")

    # Display LMS filter bank
    display_filters(lms_filters, title="LMS Filter Bank", save_path="LMS.png")

    # Display LML filter bank
    display_filters(lml_filters, title="LML Filter Bank", save_path="LML.png")
    
    """
    Generate Gabor Filter Bank: (Gabor)
    Display all the filters in this filter bank and save image as Gabor.png,
    use command "cv2.imwrite(...)"
    """
    
    scales = [5,6,7,8,9]  # Different scales
    orientations = 8  # Number of orientations
    gabor_filters = generate_gabor_filter_bank(scales, orientations)

    # Display and save the filters
    fig, axes = plt.subplots(len(scales), orientations, figsize=(20, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gabor_filters[i], cmap='gray')
        ax.axis('off')
    plt.savefig('Gabor.png')
    plt.show()

    image_path = '/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/BSDS500/Images/1.jpg'  
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at {image_path}")

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    scales = [5, 10, 15]  # Three scales
    orientations = 8  # Eight orientations
    
    # Generate the half-disc filter bank
    masks = generate_half_disc_bank(scales, orientations)
    
    # Display the half-disc masks
    display_half_discs(masks, save_path='HalfDiscMasks.png')

    # Generate Texton Map
    filter_bank = dog_filters + lms_filters + lml_filters + gabor_filters
    texton_map = generate_texton_map(gray_image, filter_bank)
    
    # Display and save Texton Map
    plt.imshow(texton_map, cmap='viridis')
    plt.axis('off')
    plt.savefig('TextonMap_ImageName.png')
    plt.show()

    # Generate Brightness Map
    brightness_map = generate_brightness_map(image)
    
    # Display and save Brightness Map
    plt.imshow(brightness_map, cmap='viridis')
    plt.axis('off')
    plt.savefig('BrightnessMap_ImageName.png')
    plt.show()

    # Generate Color Map
    color_map = generate_color_map(image)
    
    # Display and save Color Map
    plt.imshow(color_map, cmap='viridis')
    plt.axis('off')
    plt.savefig('ColorMap_ImageName.png')
    plt.show()
    Tg = chi_square_distance(texton_map, masks, num_bins=64)
    Bg = chi_square_distance(brightness_map, masks, num_bins=16)
    Cg = chi_square_distance(color_map, masks, num_bins=16)
    
    # Display and save Texton Gradient
    plt.imshow(np.sum(Tg, axis=-1), cmap='viridis')
    plt.axis('off')
    plt.savefig( "/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/Code/Tg_Image.png")
    plt.show()

    # Generate Brightness Gradient (Bg)
    #brightness_gradient = chi_square_distance(brightness_map, masks)
    
    # Display and save Brightness Gradient
    plt.imshow(np.sum(Bg, axis=-1), cmap='viridis')
    plt.axis('off')
    plt.savefig("/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/Code/Bg_ImageName.png")
    plt.show()

    # Generate Color Gradient (Cg)
    #color_gradient = chi_square_distance(color_map, masks)
    
    # Display and save Color Gradient
    plt.imshow(np.sum(Cg, axis=-1), cmap='viridis')
    plt.axis('off')
    plt.savefig("/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/Code/Cg_ImageName.png")
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
    Tg=np.max(Tg,axis=2)
    Bg=np.max(Bg,axis=2)
    Cg=np.max(Cg,axis=2)
    feature_strength = (Tg + Bg + Cg) / 3.0
    w1, w2 = 0.5, 0.5
    pb_lite = feature_strength * (w1 * canny_baseline + w2 * sobel_baseline)
    pb_lite = cv2.threshold(pb_lite, 0.5, 1, cv2.THRESH_BINARY)[1]
    
    # Display and save PbLite
    plt.imshow(pb_lite, cmap='grey')
    plt.axis('off')
    plt.savefig( "/home/samruddhi/Downloads/YourDirectoryID_hw0/Phase1/Code/PbLite_ImageName.png")
    plt.show()
    print("PbLite image saved as PbLite_ImageName.png")

if __name__ == '__main__':
    main()