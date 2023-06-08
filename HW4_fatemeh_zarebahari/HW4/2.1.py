import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2

def harris_corner_detector(image, sigma=1, threshold=0.01):
    # Compute derivatives
    dx = np.gradient(image, axis=1)
    dy = np.gradient(image, axis=0)

    # Compute products of derivatives
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # Apply Gaussian filter to the products of derivatives
    dx2_smoothed = gaussian_filter(dx2, sigma)
    dy2_smoothed = gaussian_filter(dy2, sigma)
    dxy_smoothed = gaussian_filter(dxy, sigma)

    # Compute Harris response
    det = dx2_smoothed * dy2_smoothed - dxy_smoothed ** 2
    trace = dx2_smoothed + dy2_smoothed
    harris_response = det - 0.04 * (trace ** 2)

    # Threshold the Harris response
    corners = np.zeros_like(image)
    corners[harris_response > threshold * harris_response.max()] = 255

    return corners


def resize_image(image, scale):
    height, width = image.shape[:2]
    resized_height = int(scale * height)
    resized_width = int(scale * width)
    resized_image = cv2.resize(image, (resized_width, resized_height))
    return resized_image


# Load and convert the image to grayscale
image = cv2.imread('images/harris.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the scale factors you want to apply
scales = [0.25, 0.5, 0.75, 1]  # Example scales

for scale in scales:
    resized_gray = resize_image(gray, scale)
    resized_image = resize_image(image, scale)

    # Apply Harris Corner detection
    corners = harris_corner_detector(resized_gray, sigma=1, threshold=0.01)

    # Find centroids of detected corners
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners)

    # Draw circles around the centroids
    for centroid in centroids[1:]:
        cv2.circle(resized_image, (int(centroid[0]), int(centroid[1])), 3, (0, 255, 0), -1)

    # Display the result
    cv2.imshow('Corners (Scale: {})'.format(scale), resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
