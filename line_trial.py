import numpy as np
import cv2

im = cv2.imread('masked_image_olivedrab.png')
for r in range(im.shape[0]):
    for c in range(im.shape[1]):
        if (im[r][c] == [0,0,0]).all():
            im[r][c] = (255,255,255)

cpnts, kpnts = [], []

params = cv2.SimpleBlobDetector_Params()
   
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.042
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.00001

detector = cv2.SimpleBlobDetector_create(params)
img_clone = im.copy()

while True:
    kp = detector.detect(img_clone)
    if len(kp) == 0:
        break
    kpnts.append(kp)
    for i in kp:
        x, y = round(i.pt[0]), round(i.pt[1])
        cpnts.append((x, y))
        img_clone[y, x] = (0, 0, 0)
        cv2.circle(img_clone, (x, y), round(i.size / 2), (0, 0, 0), -1)
    width, height, _ = img_clone.shape
    for i in range(width):
        for j in range(height):
            if (img_clone[i, j] == (0, 0, 0)).all():
                img_clone[i, j] = (255, 255, 255)

im_with_keypoints = im.copy()
for i in kpnts:
    im_with_keypoints = cv2.drawKeypoints(
        im_with_keypoints,
        i,
        np.array([]),
        (0, 0, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
cv2.imshow('idk', im_with_keypoints)
cv2.waitKey(0)