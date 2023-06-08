import numpy as np
import cv2


def quantize_colors(image, num_colors):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Convert the pixel values to float
    pixels = np.float32(pixels)

    # Define the criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Perform k-means clustering
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the pixel values back to integers
    centers = np.uint8(centers)

    # Map each pixel to its corresponding center value
    quantized_pixels = centers[labels.flatten()]

    # Reshape the quantized pixels back to the original image shape
    quantized_image = quantized_pixels.reshape(image.shape)

    return quantized_image


# Load the Baboon image
image = cv2.imread('images/Baboon.bmp')

# Resize the image to a smaller size for faster processing
resized_image = cv2.resize(image, (400, 400))

# Perform color quantization to 32 colors
quantized_image_32 = quantize_colors(resized_image, 32)

# Perform color quantization to 16 colors
quantized_image_16 = quantize_colors(resized_image, 16)

# Perform color quantization to 8 colors
quantized_image_8 = quantize_colors(resized_image, 8)

# Calculate MSE and PSNR between the original and quantized images
mse_32 = np.mean((resized_image - quantized_image_32) ** 2)
mse_16 = np.mean((resized_image - quantized_image_16) ** 2)
mse_8 = np.mean((resized_image - quantized_image_8) ** 2)

psnr_32 = cv2.PSNR(resized_image, quantized_image_32)
psnr_16 = cv2.PSNR(resized_image, quantized_image_16)
psnr_8 = cv2.PSNR(resized_image, quantized_image_8)

# Display the original and quantized images
cv2.imshow('Original', resized_image)
cv2.imshow('Quantized (32 colors)', quantized_image_32)
cv2.imshow('Quantized (16 colors)', quantized_image_16)
cv2.imshow('Quantized (8 colors)', quantized_image_8)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the MSE and PSNR values
print(f"MSE (32 colors): {mse_32}")
print(f"MSE (16 colors): {mse_16}")
print(f"MSE (8 colors): {mse_8}")
print()
print(f"PSNR (32 colors): {psnr_32}")
print(f"PSNR (16 colors): {psnr_16}")
print(f"PSNR (8 colors):{psnr_8}")

