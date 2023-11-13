import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_image(file_path):
    """Read the image and convert it to RGB."""
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image):
    """Reshape the image to a 2D array of pixels and 3 color values (RGB) and convert to float."""
    pixel_values = image.reshape((-1, 3))
    return np.float32(pixel_values)

def perform_kmeans_clustering(pixel_values, k=3):
    """Perform k-means clustering on the pixel values."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return compactness, labels, np.uint8(centers)

def create_segmented_image(pixel_values, labels, centers):
    """Create a segmented image using the cluster centroids."""
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image.shape)

def minify_image(image, crop_box, size): 
    if (crop_box is not None):
        cropped_image = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    downscaled_image = cv2.resize(cropped_image, (size[0], size[1]))
    return downscaled_image

def create_dotcloud_coord(pixels):
    x = pixels[:, :, 0].flatten()  
    y = pixels[:, :, 1].flatten()  
    z = pixels[:, :, 2].flatten() 
    return x, y, z 

def create_centers_coord(centers):
    x = centers[:, 0]
    y = centers[:, 1]
    z = centers[:, 2]

    return x, y, z

def display_image(image):
    """Display the image using matplotlib."""
    plt.imshow(image)
    plt.show()


def display_images_with_dotcloud(original_image, segmented_image, k, centroids):
    fig = plt.figure(figsize=(12, 4))
    print(centroids)
    ax1 = fig.add_subplot(131)
    ax1.imshow(original_image)
    ax1.set_title('Image Originale')

    ax2 = fig.add_subplot(132)
    ax2.imshow(segmented_image)
    ax2.set_title(f'Image Segmentée ({k} clusters)')

    ax3 = fig.add_subplot(133, projection='3d')
    # minified_image = minify_image(original_image, (0,0,350,350), (70,70))
    minified_image = minify_image(original_image, None, (70,70))
    pixels = np.array(minified_image)

    x, y, z = create_dotcloud_coord(pixels)
    height, width, channels = pixels.shape
    colors = pixels.reshape((height * width, channels)) / 255.0
    ax3.scatter(x, y, z, c=colors, marker='.', s=1)

    centroids_x, centroids_y, centroids_z = create_centers_coord(centroids)
    ax3.scatter(centroids_x, centroids_y, centroids_z, c='blue', marker='o', s=50, label='Centroids')

    ax3.set_xlabel('Rouge')
    ax3.set_ylabel('Vert')
    ax3.set_zlabel('Bleu')
    ax3.set_title('Nuage de Points')

    plt.show()

def display_k_means(image):
    fig = plt.figure(figsize=(12, 4))

    pixel_values2 = preprocess_image(image)
    _, labels2, centers2 = perform_kmeans_clustering(pixel_values, 2)
    segmented_image2 = create_segmented_image(pixel_values2, labels2, centers2)
    ax1 = fig.add_subplot(131)
    ax1.imshow(segmented_image2)
    ax1.set_title('K means 2')

    pixel_values8 = preprocess_image(image)
    _, labels8, centers8 = perform_kmeans_clustering(pixel_values, 8)
    segmented_image8 = create_segmented_image(pixel_values8, labels8, centers8)
    ax1 = fig.add_subplot(132)
    ax1.imshow(segmented_image8)
    ax1.set_title('K means 8')

    pixel_values32 = preprocess_image(image)
    _, labels32, centers32 = perform_kmeans_clustering(pixel_values, 32)
    segmented_image32 = create_segmented_image(pixel_values32, labels32, centers32)
    ax1 = fig.add_subplot(133)
    ax1.imshow(segmented_image32)
    ax1.set_title('K means 32')

    plt.show()



if __name__ == "__main__":
    image_path = sys.argv[1]
    k = int(sys.argv[2])
    # read the image
    image = read_image(image_path)
    # preprocess the image
    pixel_values = preprocess_image(image)
    # compactness is the sum of squared distance from each point to their corresponding centers
    compactness, labels, centers = perform_kmeans_clustering(pixel_values, k)
    # create the segmented image
    segmented_image = create_segmented_image(pixel_values, labels, centers)
    # display the image
    display_images_with_dotcloud(image, segmented_image, k, centers)
    # display_k_means(image)
