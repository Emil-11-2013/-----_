# --------------------------------------------------------------
# Import necessary libraries for:
# 1. Image processing (OpenCV)
import cv2
# 2. Numerical operations (NumPy)
import numpy as np
# 3. Displaying images (Matplotlib)
import matplotlib.pyplot as plt
# --------------------------------------------------------------

# --------------------------------------------------------------
# Define a utility function to display images using Matplotlib.
# 1. Set up the figure size.
def display_image(image, title="Image"):
    plt.figure(figsize=(8, 6))
    if len(image.shape) == 2:  # Grayscale image
        plt.imshow(image, cmap='gray')
    else:  # Color image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

def interactive_edge_detection(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image.")
        return

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    display_image(edges, title="Edge Detection")

    print("Select an option:")
    print("1. Sobel Edge Detection")
    print("2. Canny Edge Detection")
    print("3. Laplacian Edge Detection")
    print("4. Gaussian Smoothing")
    print("5. Median Filtering")
    print("6. Exit")

    
    while True: 
        choice = input("Enter your choice (1-6): ")
        if choice == '1':
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
            sobel_combined = cv2.magnitude(sobelx, sobely)
            display_image(sobel_combined.astype(np.uint8), title="Sobel Edge Detection")
        elif choice == '2':
            edges = cv2.Canny(image, 100, 200)
            display_image(edges, title="Canny Edge Detection")
        elif choice == '3':
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            display_image(laplacian.astype(np.uint8), title="Laplacian Edge Detection")
        elif choice == '4':
            gaussian = cv2.GaussianBlur(image, (5, 5), 0)
            display_image(gaussian, title="Gaussian Smoothing")
        elif choice == '5':
            median = cv2.medianBlur(image, 5)
            display_image(median, title="Median Filtering")
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")
interactive_edge_detection('photo.jpg')