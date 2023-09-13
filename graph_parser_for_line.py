import cv2
import matplotlib.pyplot as plt
import numpy as np

# %matplotlib inline
# from google.colab.patches import cv2_imshow
import keras_ocr
from scipy.spatial import KDTree
from sklearn.linear_model import RANSACRegressor
import easyocr
import re


class GraphParser():
    def __init__(self) -> None:
        return

    def read_img(self,img_path):
        #"/content/drive/MyDrive/Graph Ingestion Engine/Scatter_plots/875.png"
        img = cv2.imread(img_path)
        # plt.imshow(img)
        # plt.show(block="True")

        return img

    def img_to_binary(self,img):
        # Convert image to grayscale and binasry
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_dilation = cv2.dilate(img_gray, np.ones((3, 3), np.uint8), iterations=1)
        img_erosion = cv2.erode(img_dilation, np.ones((3, 3), np.uint8), iterations=1)
        th, img_gray_th_otsu = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)        
        # cv2.imshow("gray",img_gray_th_otsu)
        return img_gray, img_gray_th_otsu

    """## Axis Detection"""

    def detect_edges(self,img_gray_th_otsu):
        #Edge detection
        edges = cv2.Canny(img_gray_th_otsu, 0, 50)
        # plt.imshow(edges,cmap="Greys")
        # plt.show(block="True")
        return edges

    def line_detection(self,edges):
        #Line detection using Hough Transform
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 50  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
        lines = np.array(lines)
        lines = np.squeeze(lines)
        return lines

    def find_axes(self,img,lines):
        line_image = img.copy() * 0  # creating a blank to draw lines on
        #Find longest horizontal and vertical lines as x and y axes
        xaxis_len = 0
        xaxis_idx = -1
        yaxis_len = 0
        y_axis_idx = -1
        for idx,line in enumerate(lines):
            x1,y1,x2,y2=line
            if x1==x2: #vertical line
                if np.abs(y1-y2)>yaxis_len:
                    yaxis_len = np.abs(y1-y2)
                    yaxis_idx = idx
            if y1==y2: #horizontal line
                if np.abs(x1-x2)>xaxis_len:
                    xaxis_len = np.abs(x1-x2)
                    xaxis_idx = idx

        xaxis = lines[xaxis_idx]
        yaxis = lines[yaxis_idx]

        xx1,xy1,xx2,xy2 = lines[xaxis_idx]
        yx1,yy1,yx2,yy2 = lines[yaxis_idx]
        _ = cv2.line(line_image,(xx1,xy1),(xx2,xy2),(255,255,255),3)
        _ = cv2.line(line_image,(yx1,yy1),(yx2,yy2),(255,255,255),3)

        gray_line_image = cv2.cvtColor(line_image, cv2.COLOR_BGR2GRAY)

        # plt.imshow(line_image, cmap="gray")
        # plt.show(block="True")
        return xaxis, yaxis, gray_line_image


    def remove_axis_lines(self,gray_line_image, img):
        img_without_axes = img.copy()
        indices = np.where(gray_line_image==[255])
        coordinates = np.asarray(list(zip(indices[0], indices[1])))
        for i in coordinates:            
            img_without_axes[i[0] - 5:,:i[1] + 5] = 255            
        return img_without_axes


    def detect_x_axis_ticks(self,img_bin,xaxis):
        """x-axis tick detection"""

        xx1,xy1,xx2,xy2 = xaxis
        x_axis_image_segment = img_bin[xy1-6:xy1+8,xx1:xx2]
        # cv2.imshow("X-axis segment",x_axis_image_segment)
        # cv2.waitKey(0)

        x_axis_image_segment_bin = np.where(x_axis_image_segment==255,1,0)
        x_axis_profile = np.sum(x_axis_image_segment_bin,axis=0)

        x_tick_idx = np.where(x_axis_profile>4)[0]
        x_tick_idx = x_tick_idx + xx1

        x_ticks_len = len(x_tick_idx)
        return np.vstack((x_tick_idx,[xy1]*x_ticks_len)).T

    def detect_y_axis_ticks(self,img_bin,yaxis):
        """Y-Axis tick detection"""

        yx1,yy1,yx2,yy2 = yaxis

        y_axis_image_segment = img_bin[yy2:yy1,yx1-6:yx2+8]
        # cv2.imshow("Y-axis segment",y_axis_image_segment)
        # cv2.waitKey(0)

        y_axis_image_segment_bin = np.where(y_axis_image_segment==255,1,0)
        y_axis_profile = np.sum(y_axis_image_segment_bin,axis=1)

        y_tick_idx = np.where(y_axis_profile>4)[0]
        y_tick_idx += yy2

        y_ticks_len = len(y_tick_idx)
        return np.vstack(([yx1]*y_ticks_len,y_tick_idx)).T

    def visualize_ticks(self,img, x_tick_coords,y_tick_coords):
        """## Visualizing ticks"""

        img2 = img.copy()
        for tick in x_tick_coords:
            img2 = cv2.circle(img2, (tick[0],tick[1]), 2, [0,0,255], 2)
        for tick in y_tick_coords:
            img2 = cv2.circle(img2, (tick[0],tick[1]), 2, [0,0,255], 2)

        # plt.imshow(img2)
        # plt.show(block="True")

    def text_recognition_easyocr(self, img_path):
        reader = easyocr.Reader(['en'])
        result = reader.readtext(img_path)
        np_result = np.array(result)

        predicted_text = np_result[:,1]

        array_of_bounding_boxes = np.array([np.array(xi,dtype=int) for xi in np_result[:,0]])
        predicted_text_center = np.mean(array_of_bounding_boxes,axis=1)

        return predicted_text, predicted_text_center


    def text_recognition_keras(self,img_path):
        """# Text Detection"""

        # keras-ocr will automatically download pretrained
        # weights for the detector and recognizer.
        pipeline = keras_ocr.pipeline.Pipeline()

        images = [keras_ocr.tools.read(img_path)]

        # Each list of predictions in prediction_groups is a list of
        # (word, box) tuples.
        prediction_groups = pipeline.recognize(images)

        # Plot the predictions
        # fig, axs = plt.subplots(nrows=1, figsize=(20, 20))
        # for image, predictions in zip(images[0], prediction_groups[0]):
        # keras_ocr.tools.drawAnnotations(image=images[0], predictions=prediction_groups[0], ax=axs)

        predicted_text = np.array([pred[0] for pred in prediction_groups[0]])
        predicted_text_center = np.array([np.mean(pred[1],axis=0) for pred in prediction_groups[0]])

        return predicted_text, predicted_text_center


    def filter_digits_ocr(self,predicted_text, predicted_text_center):
        digit_idx = [idx for idx,text in enumerate(predicted_text) if text.isdigit()]

        predicted_digits = predicted_text[digit_idx]
        # digit_idx = []
        # predicted_digits = []
        # for idx,text in enumerate(predicted_text):
        #     try:
        #         x = float(text)
        #         digit_idx.append(idx)
        #         predicted_digits.append(text)
        #     except:
        #         continue

        predicted_digit_center = predicted_text_center[digit_idx]
        return predicted_digits, predicted_digit_center


    def match_tick_to_digits(self,predicted_digit_center, x_tick_coords, y_tick_coords):
        """Matching ticks and digits"""

        tree = KDTree(predicted_digit_center)
        x_digit_matches = tree.query(x_tick_coords)
        y_digit_matches = tree.query(y_tick_coords)
        return x_digit_matches, y_digit_matches

    def filter_tick_matches_by_dist(self,x_digit_matches, y_digit_matches, x_tick_coords, y_tick_coords, predicted_digits):
        """Filter out matches that are too far"""

        x_digit_dist,x_digit_idx = x_digit_matches

        y_digit_dist,y_digit_idx = y_digit_matches

        tol = 2

        mean_x_dist = np.mean(x_digit_dist)
        mean_y_dist = np.mean(y_digit_dist)

        bad_x_matches = np.where(x_digit_dist> mean_x_dist+tol)[0]
        bad_y_matches = np.where(y_digit_dist> mean_y_dist+tol)[0]

        x_digit_idx = np.delete(x_digit_idx,bad_x_matches)
        y_digit_idx = np.delete(y_digit_idx,bad_y_matches)

        x_tick_coords = np.delete(x_tick_coords,bad_x_matches,axis=0)
        y_tick_coords = np.delete(y_tick_coords,bad_y_matches,axis=0)

        x_digit = np.array(predicted_digits[x_digit_idx],dtype="float")
        x_digit_coords = np.vstack((x_digit,np.zeros(len(x_digit)))).T
        y_digit = np.array(predicted_digits[y_digit_idx],dtype="float")
        y_digit_coords = np.vstack((np.zeros(len(y_digit)),y_digit)).T

        return x_digit_coords,y_digit_coords,x_tick_coords,y_tick_coords

    """### RANSAC Regression"""


    def plot_ransac_regression(self,reg,X,y):
        #
        # Get the Inlier mask; Create outlier mask
        #
        inlier_mask = reg.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        #
        # Create scatter plot for inlier datset
        #
        plt.figure(figsize=(8, 8))
        plt.scatter(X[inlier_mask], y[inlier_mask],
                    c='steelblue', edgecolor='white',
                    marker='o', label='Inliers')
        
        # Create scatter plot for outlier datset
        
        plt.scatter(X[outlier_mask], y[outlier_mask],
                    c='limegreen', edgecolor='white',
                    marker='s', label='Outliers')
        #
        # Draw the best fit line
        #
        line_X = np.arange(3, 500, 1)
        line_y_ransac = reg.predict(line_X[:, np.newaxis])
        plt.plot(line_X, line_y_ransac, color='black', lw=2)
        plt.xlabel('pixel coords', fontsize=15)
        plt.ylabel('graph coords', fontsize=15)
        plt.legend(loc='upper left', fontsize=12)
        plt.show(block="True")

    def x_axis_regression(self,x_tick_coords,x_digit_coords):
        reg_x = RANSACRegressor(min_samples=2,max_trials=100,
                                loss='absolute_error', random_state=42,
                                residual_threshold=10).fit(x_tick_coords[:,0].reshape(-1,1), x_digit_coords[:,0].reshape(-1,1))

        self.plot_ransac_regression(reg_x,x_tick_coords,x_digit_coords)
        return reg_x

    def y_axis_regression(self,y_tick_coords,y_digit_coords):
        reg_y = RANSACRegressor(min_samples=3,max_trials=100,
                                loss='absolute_error', random_state=42,
                                residual_threshold=10).fit(y_tick_coords[:,1].reshape(-1,1), y_digit_coords[:,1].reshape(-1,1))

        # self.plot_ransac_regression(reg_y,y_tick_coords,y_digit_coords)
        return reg_y
