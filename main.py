import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def read_image(file_path):
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def preprocess_image(image):
    pixel_values = image.reshape((-1, 3))
    return np.float32(pixel_values)

def perform_kmeans_clustering(pixel_values, k=3):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    compactness, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return compactness, labels, np.uint8(centers)

def create_segmented_image(pixel_values, labels, centers):
    segmented_image = centers[labels.flatten()]
    return segmented_image.reshape(image.shape)

def minify_image(image, crop_box, size): 
    if (crop_box is not None):
        image = image[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    downscaled_image = cv2.resize(image, (size[0], size[1]))
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

def get_center_density(k, labels):
    freq = [0 for _ in range(k)]
    for i in labels: 
        freq[i[0]] += 1
    max_freq = max(freq)
    min_freq = min(freq)
    freq = [(i - 0.5*min_freq) / max_freq * 500 for i in freq]
    return freq

def display_images_with_dotcloud(original_image, minify_im, segmented_image, k, centroids, labels):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(original_image)
    ax1.set_title('Image Originale')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(segmented_image)
    ax2.set_title(f'Image Segmentée ({k} clusters)')

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    pixels = np.array(minify_im)

    x, y, z = create_dotcloud_coord(pixels)
    height, width, channels = pixels.shape
    colors = pixels.reshape((height * width, channels)) / 255.0
    ax3.scatter(x, y, z, c=colors, marker='.', s=1)

    freq = get_center_density(k, labels)
    centroids_x, centroids_y, centroids_z = create_centers_coord(centroids)
    ax3.scatter(centroids_x, centroids_y, centroids_z, c='blue', marker='o', s=freq, label='Centroids')

    ax3.set_xlabel('Rouge')
    ax3.set_ylabel('Vert')
    ax3.set_zlabel('Bleu')
    ax3.set_title('Nuage de Points')

    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    segmented_colors = np.array([minify_centers[i[0]] for i in labels]).reshape((height * width, channels)) / 255.0
    print(segmented_colors[0])
    ax4.scatter(x, y, z, c=segmented_colors, marker='.', s=1)

    ax4.scatter(centroids_x, centroids_y, centroids_z, c='blue', marker='o', s=freq, label='Centroids')

    ax4.set_xlabel('Rouge')
    ax4.set_ylabel('Vert')
    ax4.set_zlabel('Bleu')
    ax4.set_title('Nuage de Points')

    plt.show()

def display_k_means(image):
    fig = plt.figure(figsize=(12, 4))

    pixel_values2 = preprocess_image(image)
    _, labels2, centers2 = perform_kmeans_clustering(minify_pixel_values, 2)
    segmented_image2 = create_segmented_image(pixel_values2, labels2, centers2)
    ax1 = fig.add_subplot(131)
    ax1.imshow(segmented_image2)
    ax1.set_title('K means 2')

    pixel_values8 = preprocess_image(image)
    _, labels8, centers8 = perform_kmeans_clustering(minify_pixel_values, 8)
    segmented_image8 = create_segmented_image(pixel_values8, labels8, centers8)
    ax1 = fig.add_subplot(132)
    ax1.imshow(segmented_image8)
    ax1.set_title('K means 8')

    pixel_values32 = preprocess_image(image)
    _, labels32, centers32 = perform_kmeans_clustering(minify_pixel_values, 32)
    segmented_image32 = create_segmented_image(pixel_values32, labels32, centers32)
    ax1 = fig.add_subplot(133)
    ax1.imshow(segmented_image32)
    ax1.set_title('K means 32')

    plt.show()



if __name__ == "__main__":
    image_path = sys.argv[1]
    k = int(sys.argv[2])
    image = read_image(image_path)
    minify_im = minify_image(image, None, (70,70))
    pixel_values = preprocess_image(image)
    minify_pixel_values = preprocess_image(minify_im)
    compactness, labels, centers = perform_kmeans_clustering(pixel_values, k)
    minify_compactness, minify_labels, minify_centers = perform_kmeans_clustering(minify_pixel_values, k)
    segmented_image = create_segmented_image(pixel_values, labels, centers)
    display_images_with_dotcloud(image, minify_im, segmented_image, k, minify_centers, minify_labels)
