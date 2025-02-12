import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


image1 = cv2.imread('data/example_1/I1.PNG')
image2 = cv2.imread('data/example_1/I2.PNG')

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#section 1

def points_of_interest(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use SIFT to detect keypoints and compute descriptors
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def draw_keypoints(image1, keypoints1, image2, keypoints2):
    # Draw keypoints on the images
    img_with_keypoints1 = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 0, 255))
    img_with_keypoints2 = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 0, 255))
    
    # Create a subplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(img_with_keypoints1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image 1 with Keypoints')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(img_with_keypoints2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Image 2 with Keypoints')
    axes[1].axis('off')

    plt.show()


# Find keypoints and descriptors
keypoints1, descriptors1 = points_of_interest(image1)
keypoints2, descriptors2 = points_of_interest(image2)

# Draw keypoints on both images
draw_keypoints(image1, keypoints1, image2, keypoints2)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#section 2

def draw_matches(image1, image2, keypoints1, keypoints2, matches):
    # Draw matches with circles on the images
    image1 = image1.copy()
    image2 = image2.copy()
    for match in matches[:]:
        pt1 = tuple(map(int, keypoints1[match.queryIdx].pt))
        pt2 = tuple(map(int, keypoints2[match.trainIdx].pt))
        cv2.circle(image1, pt1, 3, (0, 0, 255), -1)  
        cv2.circle(image2, pt2, 3, (0, 0, 255), -1)  
    
    # Update the subplot with the images showing matches with circles
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image 1 with Matches')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Image 2 with Matches')
    axes[1].axis('off')

    plt.show()

def draw_matches_with_lines(image1, image2, keypoints1, keypoints2, matches, title, num_matches=100):
    # Create a new image to draw matches with lines
    img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:num_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Convert BGR image to RGB for matplotlib
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    
    # Display the image with matches
    plt.figure(figsize=(12, 6))
    plt.imshow(img_matches_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()



# Find matches
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches with circles (without lines)
draw_matches(image1, image2, keypoints1, keypoints2, matches)

# Draw matches with lines connecting keypoints
draw_matches_with_lines(image1, image2, keypoints1, keypoints2, matches, "Matches with Lines")


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#section 3
#a

# Read the intrinsic camera matrix from a the text file K
def read_intrinsic_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.strip().split(','))) for line in lines]
        return np.array(matrix)

# Load the intrinsic matrix
K = read_intrinsic_matrix('data/example_1/K.txt')
#print(f'{K}')

def find_essential_fundamental_matrix(keypoints1, keypoints2, matches, K):
    points1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
    E, maskE = cv2.findEssentialMat(points1, points2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    #F, maskF = cv2.findFundamentalMat(points1, points2, method=cv2.RANSAC)
    F = np.linalg.inv(K).T @ E @np.linalg.inv(K)
    print("Essential Matrix:\n", E)
    print("Fundamental Matrix:\n", F)
    return E, F, maskE

E, F, maskE = find_essential_fundamental_matrix(keypoints1, keypoints2, matches, K)


#b
def draw_inlier_matches(image1, keypoints1, image2, keypoints2, matches, mask):
    inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
    draw_matches_with_lines(image1, image2, keypoints1, keypoints2, inlier_matches, "Matches with Lines Without outliers")
    return inlier_matches

# Draw inlier matches based on the Essential matrix
inlier_matches = draw_inlier_matches(image1.copy(), keypoints1, image2.copy(), keypoints2, matches, maskE.ravel().tolist())


#c

def draw_epipolar_lines(image1, image2, points1, points2, F):
    # Convert images to RGB
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    # Draw the points and epipolar lines on the images
    for pt1, pt2 in zip(points1, points2):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, pt2))

        # Draw the points
        cv2.circle(image1_rgb, pt1, 5, (0, 255, 0), -1)
        cv2.circle(image2_rgb, pt2, 5, (0, 255, 0), -1)

        # Compute the epipolar lines in the second image for the point in the first image
        line2 = cv2.computeCorrespondEpilines(np.array([pt1]), 1, F).reshape(-1, 3)
        for r in line2:
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [image2_rgb.shape[1], -(r[2] + r[0] * image2_rgb.shape[1]) / r[1]])
            cv2.line(image2_rgb, (x0, y0), (x1, y1), (0, 255, 0), 1)

        # Compute the epipolar lines in the first image for the point in the second image
        line1 = cv2.computeCorrespondEpilines(np.array([pt2]), 2, F).reshape(-1, 3)
        for r in line1:
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [image1_rgb.shape[1], -(r[2] + r[0] * image1_rgb.shape[1]) / r[1]])
            cv2.line(image1_rgb, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # Display the images with epipolar lines
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image1_rgb)
    axes[0].set_title('Epipolar Lines in Image 1')
    axes[0].axis('off')

    axes[1].imshow(image2_rgb)
    axes[1].set_title('Epipolar Lines in Image 2')
    axes[1].axis('off')

    plt.show()

def visualize_inlier_matches_with_epipolar_lines(image1, image2, keypoints1, keypoints2, inlier_matches, F, num_matches=50):
    # Randomly select a small number of inlier matches
    selected_matches = random.sample(inlier_matches, num_matches)

    points1 = np.array([keypoints1[m.queryIdx].pt for m in selected_matches])
    points2 = np.array([keypoints2[m.trainIdx].pt for m in selected_matches])

    # Draw the selected points and epipolar lines
    draw_epipolar_lines(image1.copy(), image2.copy(), points1, points2, F)

# Visualize inlier matches with epipolar lines
visualize_inlier_matches_with_epipolar_lines(image1, image2, keypoints1, keypoints2, inlier_matches, F)


