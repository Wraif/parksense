import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

output_path = "lot_values.txt"
img_path = "parking_lot2.jpg"

img = cv.imread(img_path, 0) # Open image in greyscale mode
if img is None:
    sys.exit("Could not read the image.")
    
print(f'Image scale: {img.shape}')

# HSV thresholding
#img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Define bounds for parking spaces (tuples of top left pixel location)
lot = {
    'SUV': (76,32), #img[76:267, 32:134],
    'Hatchback': (76,143),#img[76:267, 143:251],
    'Pickup': (76,260),#img[76:267, 260:362],
    'Mini': (76,371),#img[76:267, 371:472],
    'Sedan': (76,481),#img[76:267, 481:583],
    'Grey': (346,32),
    'Minivan': (346, 143),
    'Oil': (346,260),#img[346:537, 260:362],
    'Empty1': (346, 371),
    'Empty2': (346,481)#img[346:537, 481:583]
}
# Define dimensions of individual parking spaces
space_size = (192,101)

try:
    f = open(output_path, 'w')
    #TODO: implement timestamp for last update to file

    # Process each parking space separately
    kernel = np.ones((5,5),np.float32)/25
    for name,top_left in lot.items():
        # Define space region of interest
        c1 = top_left[0]
        r1 = top_left[1]
        space = img[c1:c1+space_size[0], r1:r1+space_size[1]]

        # https://stackoverflow.com/questions/62042172/how-to-remove-noise-in-image-opencv-python
        # Apply blur and divide filters to remove background noise
        blur = cv.GaussianBlur(space, (0,0), sigmaX=33, sigmaY=33)
        divide = cv.divide(space, blur, scale=255)

        # Apply threshold
        thresh = cv.threshold(divide, 200, 255, cv.THRESH_BINARY)[1]
        
        # https://stackoverflow.com/questions/67285972/how-to-fill-canny-edge-image-in-opencv-python
        # Get largest contour
        contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        big_contour = max(contours, key=cv.contourArea)

        # Draw white filled contour on black background
        result = np.zeros_like(space)
        cv.drawContours(result, [big_contour], 0, (255,255,255), cv.FILLED)

        # Get ratio of filled space from resulting image
        black_pixel_count = np.sum(result == 0)
        filled_ratio = (black_pixel_count / space.size) * 100
        print(f'Thresholding %: {round(filled_ratio, 2)}')

        # Get confidence level of there being an object in the space
        confidence = 0
        if filled_ratio <= 10 and filled_ratio > 0:
            confidence = 1
        elif filled_ratio > 10:
            confidence = 2

        # Write the results to lot_values.txt
        f.write(f"{top_left};{space_size}")
        f.write(';')
        f.write(str(confidence))
        f.write('\n')

        # Put these results into a plot for displaying
        titles = ['Original Image', 'Divide Filter', 'Threshold', f'Contours ({round(filled_ratio, 2)}%)']
        images = [space, divide, thresh, result]
        for i in range(4):
            plt.subplot(2,2,i+1),plt.imshow(images[i],'grey')
            plt.title(titles[i])
            plt.xticks([]),plt.yticks([])
        manager = plt.get_current_fig_manager()
        manager.full_screen_toggle()
        plt.suptitle(name)
        plt.draw()
        plt.waitforbuttonpress(0)
        plt.close()
        
except Exception as e:
    print("Error writing to file:", e)
finally:
    f.close()