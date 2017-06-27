# This will analyse a picture and produce 2 derivatives of the picture that visualise the colourfullness of the image
# The first derivative shows a greyscale image where the lightness is proprtional to the colourfullness
# the second derivative shows the original knocked back to grey and then dimmed to 0.25 with the 
# colourfullness (also knocked back to 0.75 so no overflow) added onto the blue channel
# it results in the original still being easily recognisable with the colourfull areas then being more blue
# Inspired by and using the colourfullness calculation from Adrian at pyImageSearch 
# in response to the blog article 


def roiColourfulness(roi):
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

debug = True

# how much of the original to include in the second derivative image
proportionOfOriginal = 0.3

# read in a colourful picture
imagePath = "../images/tests/colourfullness/20170617_185021.jpg"
imagePath = "../images/tests/colourfullness/20170617_185021ShrunkMore.png"
imagePath = "../images/tests/colourfullness/20170617_185021Shrunk.png"
imagePath = "../images/tests/colourfullness/20170415_154428.jpg"
# where to save the pictures
savePath = "/media/sf_localDesktop/debug"

# read it in unchanged
imageColour = cv2.imread(imagePath)

# get a copy of it in grey and convert it back to bgr so I have a channel for markup on a grey image
imageGrey = cv2.cvtColor(imageColour, cv2.COLOR_BGR2GRAY)
cv2.equalizeHist(imageGrey, imageGrey) # spreads it out so 0 and 255

# knock it  back to quarter
imageGreyHalf = np.zeros_like(imageGrey)
imageGreyHalf[:] = imageGrey * proportionOfOriginal

print("imageGreyHalf.shape", imageGreyHalf.shape)

# need to change the dtype here
imageGreyHalfAsUint8 = imageGreyHalf.view(dtype=np.uint8)

print("imageGreyHalfAsUint8.shape", imageGreyHalfAsUint8.shape)

imageGreyForMarkup = cv2.cvtColor(imageGreyHalfAsUint8, cv2.COLOR_GRAY2BGR)

# get the basic dimensions
height, width, channels = imageColour.shape

# set the size of the window that we will roll over the image
roiSize = min(height,width) / 20 # pixels for height and width of roi. Must be odd
roiSize = int(roiSize / 2) * 2 + 1 # yes, it must be odd
halfRoiSize = int(roiSize / 2)

# create a temporary home for the colourfulness index before I normalize it to 0-255 and put it back on the red channel
colourfullnessChannel = np.zeros_like(imageGrey)

for rowIndex in range(height - halfRoiSize):
    if rowIndex <= halfRoiSize:
        continue
    for pixelIndex in range(width - halfRoiSize):
        if pixelIndex <= halfRoiSize:
            continue
        roi = imageColour[rowIndex - halfRoiSize: rowIndex + halfRoiSize, pixelIndex - halfRoiSize: pixelIndex + halfRoiSize]

        colourfullnessindex = roiColourfulness(roi)
        
        colourfullnessChannel[rowIndex, pixelIndex] = colourfullnessindex

cv2.equalizeHist(colourfullnessChannel, colourfullnessChannel) # spreads it out so 0 and 255

cv2.imwrite(savePath + "/colourfullnessChannel.png", colourfullnessChannel)

# knock it  back to half
colourfullnessChannelHalf = np.zeros_like(colourfullnessChannel)
colourfullnessChannelHalf[:] = colourfullnessChannel * (1 - proportionOfOriginal)


imageGreyForMarkup[:, :, 0] = imageGreyForMarkup[:, :, 0] + colourfullnessChannelHalf

cv2.imwrite(savePath + "/imagePlusColourfullnessChannel.png", imageGreyForMarkup)





