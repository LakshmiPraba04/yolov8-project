import cv2
import os

# Specify the image name and path
image_name = '2.png'  # Ensure this path is correct
print("Loading image from:", image_name)

if not os.path.isfile(image_name):
    print(f"Error: The file {image_name} does not exist.")
else:
    # Read the image
    image = cv2.imread(image_name)

    # Grayscale Conversion
    def convert_to_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = convert_to_grayscale(image)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)

    # Resize Image
    def resize_image(image, width=None, height=None):
        if width is None and height is None:
            return image
        dim = None
        (h, w) = image.shape[:2]
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized

    resized_image = resize_image(gray_image, width=600)
    cv2.imshow('Resized Image', resized_image)

    # Noise Reduction
    def reduce_noise(image):
        return cv2.fastNlMeansDenoising(image, None, 30, 7, 21)

    noise_reduced_image = reduce_noise(resized_image)
    cv2.imshow('Noise Reduced Image', noise_reduced_image)

    # Histogram Equalization (Contrast Adjustment)
    def equalize_histogram(image):
        return cv2.equalizeHist(image)

    equalized_image = equalize_histogram(noise_reduced_image)
    cv2.imshow('Equalized Image', equalized_image)

    # Edge Detection
    def detect_edges(image):
        return cv2.Canny(image, 50, 150)

    edges = detect_edges(equalized_image)
    cv2.imshow('Edge Detected Image', edges)

    # Binarization
    def binarize_image(image):
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary_image

    binary_image = binarize_image(edges)
    cv2.imshow('Binarized Image', binary_image)

    # Save the final processed image
    processed_image_name = 'processed_image.jpg'  # Name to save the processed image
    cv2.imwrite(processed_image_name, binary_image)

    # Display the final processed image
    cv2.imshow('Final Processed Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()