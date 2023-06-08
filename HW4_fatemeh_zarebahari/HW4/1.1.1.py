import numpy as np
import cv2

def rgb_to_hsv(image):
    image_norm = image / 255.0

    red, green, blue = image_norm[:, :, 0], image_norm[:, :, 1], image_norm[:, :, 2]

    v = np.max(image_norm, axis=2)

    delta = np.max(image_norm, axis=2) - np.min(image_norm, axis=2)
    s = np.where(v != 0, delta / v, 0)

    h = np.zeros_like(v)
    mask = delta != 0
    h[mask] = np.where(v[mask] == red[mask], (green[mask] - blue[mask]) / delta[mask],
                      np.where(v[mask] == green[mask], 2 + (blue[mask] - red[mask]) / delta[mask],
                               4 + (red[mask] - green[mask]) / delta[mask]))
    h[mask] = (h[mask] / 6.0 + 1) % 1

    h = (h * 255).astype(np.uint8)
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)

    return h, s, v

rgb_image = cv2.imread("images/Lena.bmp")

h, s, v = rgb_to_hsv(rgb_image)

cv2.imshow("Hue (H)", h)
cv2.imshow("Saturation (S)", s)
cv2.imshow("Value (V)", v)
cv2.waitKey(0)
cv2.destroyAllWindows()
