import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

# Custom Cutout Class (if you still need it)
class Cutout(A.ImageOnlyTransform):
    def __init__(self, num_holes=8, max_h_size=16, max_w_size=16, fill_value=0, always_apply=False, p=0.5):
        super(Cutout, self).__init__(always_apply=always_apply, p=p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def apply(self, image, **params):
        h, w, c = image.shape
        for _ in range(self.num_holes):
            hole_h_size = np.random.randint(1, self.max_h_size)
            hole_w_size = np.random.randint(1, self.max_w_size)
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            x1 = np.clip(x - hole_w_size // 2, 0, w)
            x2 = np.clip(x + hole_w_size // 2, 0, w)
            y1 = np.clip(y - hole_h_size // 2, 0, h)
            y2 = np.clip(y + hole_h_size // 2, 0, h)
            image[y1:y2, x1:x2, :] = self.fill_value
        return image

# Load Original Image
image = cv2.imread('7.jpeg')

# Check if image was successfully loaded
if image is None:
    raise ValueError("Image not found or the file path is incorrect")

# Rotate the image 90 degrees to the left (counter-clockwise) to straighten it
image_rotated_left = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Data Augmentation Pipeline without RandomRotate90 and other unnecessary transformations
augmentations = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=0, p=0.5),  # Disable rotation here
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    Cutout(num_holes=8, max_h_size=16, max_w_size=16, fill_value=(0, 0, 0), always_apply=False, p=0.5),
])

# Apply Augmentation
augmented = augmentations(image=image_rotated_left)['image']

# Convert the augmented image back to NumPy array format (necessary for OpenCV processing)
augmented_image = np.array(augmented)

# Convert to Grayscale
gray = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)

# Median Blur
filter1 = cv2.medianBlur(gray, 5)

# Gaussian Blur
filter2 = cv2.GaussianBlur(filter1, (5, 5), 0)

# Denoising
dst = cv2.fastNlMeansDenoising(filter2, None, 17, 9, 17)

# Binarization
th1 = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Rotate the preprocessed image 90 degrees to the right (clockwise)
th1_rotated_right = cv2.rotate(th1, cv2.ROTATE_90_CLOCKWISE)

# Save the preprocessed and rotated image
cv2.imwrite('PreProcessed_Augmented_Rotated_Right.jpg', th1_rotated_right)

# Display the final preprocessed and rotated image
plt.imshow(th1_rotated_right, cmap='gray')
plt.show()
