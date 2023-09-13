import cv2
import numpy as np
import webcolors
from collections import defaultdict
import copy

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
    return closest_name


def extract_colors(image):
	# Convert the image to HSV color space.
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

	# Get the hue, saturation, and value channels of the image.
	hue_channel = hsv_image[:, :, 0]
	saturation_channel = hsv_image[:, :, 1]
	value_channel = hsv_image[:, :, 2]

	# Create a list to store the extracted colors.
	colors = set()

	# Iterate over each pixel in the image.
	for i in range(hsv_image.shape[0]):
		for j in range(hsv_image.shape[1]):
		# Extract the hue, saturation, and value of the current pixel.
			hue = hue_channel[i, j]
			saturation = saturation_channel[i, j]
			value = value_channel[i, j]
			colors.add((hue, saturation, value))

	return hsv_image, colors


def HSV_2_RGB(HSV):
	''' Converts an integer HSV tuple (value range from 0 to 255) to an RGB tuple '''

	# Unpack the HSV tuple for readability
	H, S, V = HSV

	# Check if the color is Grayscale
	if S == 0:
		R = V
		G = V
		B = V
		return (R, G, B)

	# Make hue 0-5
	region = H // 43

	# Find remainder part, make it from 0-255
	remainder = (H - (region * 43)) * 6

	# Calculate temp vars, doing integer multiplication
	P = (V * (255 - S)) >> 8
	Q = (V * (255 - ((S * remainder) >> 8))) >> 8
	T = (V * (255 - ((S * (255 - remainder)) >> 8))) >> 8


	# Assign temp vars based on color cone region
	if region == 0:
		R = V
		G = T
		B = P

	elif region == 1:
		R = Q
		G = V
		B = P

	elif region == 2:
		R = P
		G = V
		B = T

	elif region == 3:
		R = P
		G = Q
		B = V

	elif region == 4:
		R = T
		G = P
		B = V

	else:
		R = V
		G = P
		B = Q


	return (R, G, B)


def remove_black_white(img):
	blue_channel = img[:, :, 0]
	green_channel = img[:, :, 1]
	red_channel = img[:, :, 2]

	colors = set()
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			blue = blue_channel[i, j]
			green = green_channel[i, j]
			red = red_channel[i, j]
			if blue==green==red:
				colors.add((red, green, blue))
	for color in colors:
		img[np.all(img==color, axis=-1)] = (0,0,0)
	# cv2.imshow('black', img)
	# cv2.waitKey(0)
	return img

def get_individual_cp():
	# Load the image.
	image = cv2.imread("input_images/sample_online.png")
	og_image = copy.deepcopy(image)
	black_img = remove_black_white(image)
	# Extract all colors from the image.
	hsvImage, colors = extract_colors(black_img)
	# print(len(colors))
	set_colors = defaultdict(set)
	for i in colors:
		rgb = (HSV_2_RGB(i))
		# print(rgb)
		set_colors[get_colour_name(rgb)].add(i)
	set_colors = dict(sorted(set_colors.items(), key=lambda x:len(x[1]), reverse=True))
	# print(len(set_colors))
	colors_hsv_range = defaultdict(list)
	for k, v in set_colors.items():
		min_h, min_s, min_v = 10000,10000,100000
		max_h, max_s, max_v = 0,0,0
		# print(k, v)
		for i in v:
			hue, sat, val = i[0], i[1], i[2]
			min_h, max_h = min(min_h, hue), max(max_h, hue)
			min_s, max_s = min(min_s, sat), max(max_s, sat)
			min_v, max_v = min(min_v, val), max(max_v, val)
		colors_hsv_range[k] = [min_h, max_h]+[min_s, max_s]+[min_v, max_v]
	# print(colors_hsv_range)
	cnt_white = defaultdict(int)
	for k, v in colors_hsv_range.items():
		result = copy.deepcopy(og_image)
		lower_mask = np.array([v[0], v[2], v[4]], dtype=np.int32)
		upper_mask = np.array([v[1], v[3], v[5]], dtype=np.int32)
		# print(lower_mask, upper_mask)
		mask = cv2.inRange(hsvImage, lower_mask, upper_mask)
		cnt_white[k] = np.sum(mask == 255)
	colors_to_use = dict(sorted(cnt_white.items(), key=lambda x:x[1], reverse=True))
	colors_to_use = list(colors_to_use.keys())[1:3]
	# print(colors_to_use)
	res = []
	for k, v in colors_hsv_range.items():
		result = copy.deepcopy(og_image)
		lower = np.array([v[0], v[2], v[4]], dtype=np.uint8)
		upper = np.array([v[1], v[3], v[5]], dtype=np.uint8)
		mask = cv2.inRange(hsvImage, lower, upper)
		if k in colors_to_use:
			res.append(cv2.bitwise_and(og_image, og_image, mask=mask))
			cv2.imshow('mask_'+k+'.png', mask)
			cv2.imshow('masked_image_'+k+'.png', cv2.bitwise_and(og_image, og_image, mask=mask))
			cv2.waitKey(0)
	# for i in range(len(colors_to_use)):
	# 	masked_img = res[i].copy()
	# 	non_black_pixels = np.any(masked_img != [0, 0, 0], axis=-1)

if __name__ == "__main__":
	get_individual_cp()