# This will analyse a picture and produce 2 derivatives of the picture that visualise the colourfulness of the image
# The first derivative shows a greyscale image where the lightness is proprtional to the colourfulness
# the second derivative shows the original knocked back to grey and then dimmed to 0.25 with the 
# colourfulness (also knocked back to 0.75 so no overflow) added onto the blue channel
# it results in the original still being easily recognisable with the colourful areas then being more red
# Inspired by and using the colourfulness calculation from Adrian at pyImageSearch 
# in response to the blog article http://www.pyimagesearch.com/2017/06/26/labeling-superpixel-colorfulness-opencv-python/

def roicolourfulness(roi):
    # thanks to Adrian for this function

	# split the roi into its respective RGB components
	(B, G, R) = cv2.split(roi.astype("float"))
 
	# compute rg = R - G
	rg = np.absolute(R - G)
 
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
 
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
 
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
 
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

import cv2
import numpy as np 
import time

# start a timer because it's a long process!!
startTime = time.time()

# switch on or off the helpful print or save statements 
debug = True

# how much of the original to include in the second derivative image
proportionOfOriginal = 0.25

# read in a colourful picture (choose one of the images below)
imagePath = "../images/basilCatchingBall.jpg"
imagePath = "../images/basilCatchingBallShrunk10.png"

# where to save the pictures
saveFolder = "../results/"

# read it in unchanged
imageColour = cv2.imread(imagePath)

# get a copy of it in grey and convert it back to bgr so I have a channel for markup on a grey image
imageGrey = cv2.cvtColor(imageColour, cv2.COLOR_BGR2GRAY)
cv2.equalizeHist(imageGrey, imageGrey) # spreads it out so 0 and 255

# knock it  back 
imageGreyHalf = np.zeros_like(imageGrey)
imageGreyHalf[:] = imageGrey * proportionOfOriginal

if debug:
    print("imageGreyHalf.shape", imageGreyHalf.shape)

# add some colour channels so that we can put a function of colourfulness in one of them later
imageGreyForMarkup = cv2.cvtColor(imageGreyHalf, cv2.COLOR_GRAY2BGR)

# get the basic dimensions
height, width, channels = imageColour.shape

# set the size of the window that we will roll over the image
roiSize = min(height,width) / 10 # proportion of min image dimensions to have as roi size
roiSize = int(roiSize / 2) * 2 + 1 # it must be odd
halfRoiSize = int(roiSize / 2)

# create a temporary home for the colourfulness index before I normalize it to 0-255 and put it back on a colour channel
colourfulnessChannel = np.zeros_like(imageGrey)

# miss out the first and last bit where the roi doesn't fully cover the image
for rowIndex in range(halfRoiSize, height - halfRoiSize):
    for pixelIndex in range(halfRoiSize, width - halfRoiSize):

        # define the region of interest that surrounds this pixel
        roi = imageColour[rowIndex - halfRoiSize: rowIndex + halfRoiSize, pixelIndex - halfRoiSize: pixelIndex + halfRoiSize]

        # use the function to calculate the colourfulness
        colourfulnessindex = roicolourfulness(roi)
        
        # store it in the equivalent pixel position
        colourfulnessChannel[rowIndex, pixelIndex] = colourfulnessindex

# stretch the values
cv2.equalizeHist(colourfulnessChannel, colourfulnessChannel) # spreads it out so 0 and 255

# save this out as it's the first derivative I desfribed above
cv2.imwrite(saveFolder + "colourfulnessChannel.png", colourfulnessChannel)

# knock it  back to a fraction and add it to the original image
colourfulnessChannelFraction = np.zeros_like(colourfulnessChannel)
colourfulnessChannelFraction[:] = colourfulnessChannel * (1 - proportionOfOriginal)

# decide which channel to put it on. Aesthetics are your guide here. Remember it's BGR not RGB 
channelToUseForOverlay = 2

# replace that channel with the one we created that is a function of the image and the colourfulness
imageGreyForMarkup[:, :, channelToUseForOverlay] = imageGreyForMarkup[:, :, channelToUseForOverlay] + colourfulnessChannelFraction

# now let's boost the brightness up so it's not too dull
maxBrightnessOfcolourfulnessChannel = np.max(imageGreyForMarkup[:, :, channelToUseForOverlay]) 
brightnessBoostfactor = 255 / maxBrightnessOfcolourfulnessChannel
imageGreyForMarkup[:, :, 0] = imageGreyForMarkup[:, :, 0] * brightnessBoostfactor
imageGreyForMarkup[:, :, 1] = imageGreyForMarkup[:, :, 1] * brightnessBoostfactor
imageGreyForMarkup[:, :, 2] = imageGreyForMarkup[:, :, 2] * brightnessBoostfactor

# and save the second derivative out
cv2.imwrite(saveFolder + "imagePluscolourfulnessChannel.png", imageGreyForMarkup)

# We've finished so time it and print it out
endTime = time.time()
print("\nFinished. The run took this many seconds:", endTime - startTime, "\n")





