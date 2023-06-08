import cv2
import numpy as np

img1 = cv2.imread('images/sl.jpg')
img2 = cv2.imread('images/sm.jpg')
img3 = cv2.imread('images/sr.jpg')


def panorama(img1, img2):
    # For SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # For SURF
    # surf = cv2.xfeatures2d.SURF_create()
    # keypoints1, descriptors1 = surf.detectAndCompute(img1, None)
    # keypoints2, descriptors2 = surf.detectAndCompute(img2, None)

    # Initialize the matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors between img1 and img2
    matches1 = bf.match(descriptors1, descriptors2)

    # Sort the matches by distance
    matches1 = sorted(matches1, key=lambda x: x.distance)

    # Keep only the top matches (e.g., the first 20 matches)
    top_matches1 = matches1[:20]

    # Draw the matches between img1 and img2
    match_img1 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, top_matches1, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # to create panaroma:
    points1 = np.float32([keypoints1[match.queryIdx].pt for match in matches1])
    points2 = np.float32([keypoints2[match.trainIdx].pt for match in matches1])

    # Ensure the same number of keypoints for points1, points2, and points3
    min_points = min(len(points1), len(points2))
    points1 = points1[:min_points]
    points2 = points2[:min_points]

    homography_matrix1, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    panorama1 = cv2.warpPerspective(img2, homography_matrix1, (img1.shape[1] + img2.shape[1], img1.shape[0]))

    panorama = np.zeros_like(panorama1)
    panorama[:img1.shape[0], :img1.shape[1]] = img1
    panorama[:img2.shape[0], img1.shape[1]:] = panorama1[:, img1.shape[1]:]

    return match_img1, panorama



match_point_1_2, panorama1 = panorama(img1,img2)
match_point_2_3, panorama2 = panorama(img2,img3)

_,final_panorama = panorama(panorama1[:,:800],panorama2)


cv2.imshow('Matches between img1 and img2', match_point_1_2)
cv2.imshow('Matches between img2 and img3', match_point_2_3)
cv2.imshow('panorama', final_panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
