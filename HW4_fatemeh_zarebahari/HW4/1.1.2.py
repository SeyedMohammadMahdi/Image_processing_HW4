import cv2
import numpy as np

def rgb_to_lab(image):
    # Convert RGB to XYZ
    xyz_image = rgb_to_xyz(image)

    # Normalize XYZ values
    xyz_image_norm = normalize_xyz(xyz_image)

    # Convert normalized XYZ to LAB
    lab_image = xyz_to_lab(xyz_image_norm)

    # Separate the LAB channels
    l, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]

    # Convert to uint8 for display
    l = l.astype(np.uint8)
    a = a.astype(np.uint8)
    b = b.astype(np.uint8)

    return l, a, b

def rgb_to_xyz(image):
    # Conversion matrix from RGB to XYZ
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])

    # Reshape the image for matrix multiplication
    rgb_image = image.reshape((-1, 3))

    # Apply the conversion matrix
    xyz_image = np.dot(rgb_image, matrix.T)

    # Reshape back to original image dimensions
    xyz_image = xyz_image.reshape(image.shape)

    return xyz_image

def normalize_xyz(xyz_image):
    # Normalize XYZ values
    xyz_norm = xyz_image / np.array([0.950456, 1.0, 1.088754])

    # Apply nonlinear transformation for nonlinearity correction
    epsilon = 0.008856
    kappa = 903.3
    xyz_norm = np.where(xyz_norm > epsilon, xyz_norm**(1/3), (kappa * xyz_norm + 16) / 116)

    return xyz_norm

def xyz_to_lab(xyz_image):
    # Reference white point in XYZ
    ref_white_xyz = np.array([0.950456, 1.0, 1.088754])

    # Compute XYZ / reference white
    xyz_norm = xyz_image / ref_white_xyz

    # Compute f(t) function for nonlinear transformation
    epsilon = 0.008856
    kappa = 903.3
    f_t = np.where(xyz_norm > epsilon, xyz_norm**(1/3), (kappa * xyz_norm + 16) / 116)

    # Compute LAB components
    l = (116 * f_t[:, :, 1]) - 16
    a = 500 * (f_t[:, :, 0] - f_t[:, :, 1])
    b = 200 * (f_t[:, :, 1] - f_t[:, :, 2])

    # Stack LAB channels
    lab_image = np.stack((l, a, b), axis=-1)

    return lab_image

# Load the RGB image
image_path = "images/Lena.bmp"  # Replace with your image path
image = cv2.imread(image_path)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert RGB to LAB
l, a, b = rgb_to_lab(image_rgb)

# Display the L, A, B components
cv2.imshow("L Component", l)
cv2.imshow("A Component", a)
cv2.imshow("B Component", b)
cv2.waitKey(0)
cv2.destroyAllWindows()
