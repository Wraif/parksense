import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from ast import literal_eval
import sys

input_path = "lot_values.txt"
img_path = "parking_lot2.jpg"

# Open image and convert to greyscale
img_rgb = cv.imread(img_path)

if img_rgb is None:
    sys.exit("Could not read the image.")

img = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

print(f'Image scale: {img.shape}')

try:
    # Read from input file
    inputFile = open(input_path, 'r')
    lines = inputFile.read()

    spaces = []
    for line in lines.split('\n'):
        spaces += [line.split(';')]

    output = img_rgb.copy()

    # Process each parking space
    for space in spaces:
        # Parse input from lot_values.txt
        if len(space) < 3:
            continue
        
        print(space)
        top_left = literal_eval(space[0])
        space_size = literal_eval(space[1])
        confidence = literal_eval(space[2])

        # Define space region of interest
        c1 = top_left[0]
        r1 = top_left[1]
        space = img_rgb[c1:c1+space_size[0], r1:r1+space_size[1]]

        # Draw a rectangle over the space
        colour = (0, 255, 0)
        alpha = 0.2 # Transparency 0-1
        if confidence == 2:
            colour = (0, 0, 255)
        elif confidence == 1:
            colour = (0, 255, 255)

        overlay = output.copy()

        cv.rectangle(overlay, (r1, c1), (r1+space_size[1], c1+space_size[0]), colour, -1)
        cv.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Display the results
    cv.imshow('result', output)
    cv.waitKey(0)
        
except Exception as e:
    print("Error:", e)
finally:
    inputFile.close()