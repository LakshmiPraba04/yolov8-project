import cv2
import numpy as np

def preprocess_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Failed to read {image_path}")
        return None
    
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply adaptive thresholding
    binary_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to remove small noise and connect text parts
    kernel = np.ones((3,3), np.uint8)
    morph_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the text
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    
    # Apply the mask to the original image
    text_only = cv2.bitwise_and(img, img, mask=mask)
    
    # Invert the text to make it black and the background white
    preprocessed_img = cv2.bitwise_not(text_only)
    
    return preprocessed_img

# Example usage:
image_path = r"C:\Users\laksh\Documents\sample\test\6.jpeg"  # Change this to your image path
preprocessed_img = preprocess_image(image_path)

if preprocessed_img is not None:
    # Display the image using OpenCV (optional)
    cv2.imshow('Preprocessed Image', preprocessed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the preprocessed image (optional)
    output_path = r"C:\Users\laksh\Documents\sample\preprocessed\6.jpeg"  # Change this to your output path
    cv2.imwrite(output_path, preprocessed_img)
    print(f"Preprocessed image saved to {output_path}")
else:
    print("Image preprocessing failed.")
