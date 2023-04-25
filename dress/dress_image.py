from m_rcnn import *
from visualize import random_colors, get_mask_contours, draw_mask
import cv2
import numpy as np
import imutils
from sklearn.cluster import KMeans
from collections import Counter
from PIL import Image #pillow

def extract_foreground(pic, mask):
  # split the image into channels
  b, g, r = cv2.split(np.array(pic).astype('uint8'))
  # add an alpha channel with and fill all with transparent pixels (max 255)
  a = np.ones(mask.shape, dtype='uint8') * 255
  # merge the alpha channel back
  alpha_im = cv2.merge([b, g, r, a], 4)
  # create a transparent background
  bg = np.zeros(alpha_im.shape)
  # setup the new mask
  new_mask = np.stack([mask, mask, mask, mask], axis=2)
  # copy only the foreground color pixels from the original image where mask is set
  foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

  return foreground

def removeBlack(estimator_labels, estimator_cluster):
  
  
  # Check for black
  hasBlack = False
  
  # Get the total number of occurance for each color
  occurance_counter = Counter(estimator_labels)

  
  # Quick lambda function to compare to lists
  compare = lambda x, y: Counter(x) == Counter(y)
   
  # Loop through the most common occuring color
  for x in occurance_counter.most_common(len(estimator_cluster)):
    
    # Quick List comprehension to convert each of RBG Numbers to int
    color = [int(i) for i in estimator_cluster[x[0]].tolist() ]

  
    
    # Check if the color is [0,0,0] that if it is black 
    if compare(color , [0,0,0]) == True:
      # delete the occurance
      del occurance_counter[x[0]]
      # remove the cluster 
      hasBlack = True
      # delete the first layer (index) of array
      estimator_cluster = np.delete(estimator_cluster,x[0],0)
      break

  return (occurance_counter,estimator_cluster,hasBlack)

def getColorInformation(estimator_labels, estimator_cluster,hasThresholding=False):
  
  # Variable to keep count of the occurance of each color predicted
  occurance_counter = None
  
  # Output list variable to return
  colorInformation = []
  
  
  #Check for Black
  hasBlack =False
  
  # If a mask has be applied, remove th black
  if hasThresholding == True:
    
    (occurance,cluster,black) = removeBlack(estimator_labels,estimator_cluster)
    occurance_counter =  occurance
    estimator_cluster = cluster
    hasBlack = black
    
  else:
    occurance_counter = Counter(estimator_labels)
 
  # Get the total sum of all the predicted occurances
  totalOccurance = sum(occurance_counter.values())

  max_percentage = 0
  # Loop through all the predicted colors
  for x in occurance_counter.most_common(len(estimator_cluster)):
    
    index = (int(x[0]))
    
    # Quick fix for index out of bound when there is no threshold
    index =  (index-1) if ((hasThresholding & hasBlack)& (int(index) !=0)) else index
    
    
    # Get the color number into a list
    color = estimator_cluster[index].tolist()
    
    # Get the percentage of each color
    color_percentage= (x[1]/totalOccurance)

    if max_percentage < color_percentage:
    	max_percentage = color_percentage
    	r, g, b = color

    
    #make the dictionay of the information
    colorInfo = {"cluster_index":index , "color": color , "color_percentage" : color_percentage }
    
    # Add the dictionary to the list
    colorInformation.append(colorInfo)

  return colorInformation, r, g, b, max_percentage

def extractDominantColor(image,number_of_colors=5,hasThresholding=False):
  
  # Quick Fix Increase cluster counter to neglect the black(Read Article) 
  if hasThresholding == True:
    number_of_colors +=1
  
  # Taking Copy of the image
  img = image.copy()
  
  # Convert Image into RGB Colours Space
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  
  # Reshape Image
  img = img.reshape((img.shape[0]*img.shape[1]) , 3)
  
  #Initiate KMeans Object
  estimator = KMeans(n_clusters=number_of_colors)
  
  # Fit the image
  estimator.fit(img)

  print(estimator.cluster_centers_)
  
  # Get Colour Information
  colorInformation, r, g, b, max_percentage = getColorInformation(estimator.labels_,estimator.cluster_centers_,hasThresholding)
  # r+=28
  g+=28
  b+=18.5
  print(max_percentage, r, g, b)
  return colorInformation, r, g, b

colours_list = ((255, 0,0, "Red"),
								(255, 140,0,"Orange"),
								(255,255,0,"Yellow"),
								(0,128,0,"Green"),
								(0,255,255,"Cyan"),
								(0,0,255,"Blue"),
								(255,0,255,"Magenta"),
								(128,0,128,"Purple"),
								(255,245,238,"White"),
								(0,0,0,"Black"),
								(100,100,100,"Gray"),
								
								(255,20,147,"Pink"),
								(128,0,0,"Maroon"),
								(165,42,42,"Brown"),
								
								
								
								(127,255,0,"Green"),
								(234,170,0,"Mustard"),
								
								(0,128,128,"Teal"),
								(0,0,128,"Navy blue"),
								(75,0,130,"Indigo"),
								(155,38,182,"Violet"))

def nearest_colour( subjects, query ):
    return min( subjects, key = lambda subject: sum( (s - q) ** 2 for s, q in zip( subject, query ) ) )

original_image = cv2.imread("person.png")

test_model, inference_config = load_inference_model(1, "mask_rcnn_object_0004.h5")
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

# Detect results
r = test_model.detect([image])[0]
colors = random_colors(80)

# Get Coordinates and show it on the image
object_count = len(r["class_ids"])
for i in range(object_count):
    # 1. Mask
    mask = r["masks"][:, :, i]
    contours = get_mask_contours(mask)
    for cnt in contours:
        cv2.polylines(original_image, [cnt], True, colors[i], 2)
        masked_image = draw_mask(original_image, [cnt], colors[i])

cv2.imshow("image", masked_image)

print(mask.shape)
background = np.zeros(mask.shape)
bin_mask = np.where(mask, 255, background).astype(np.uint8)
cv2.imshow("binary_image", bin_mask)
cv2.imwrite('binary image.png', bin_mask)

# img = cv2.imread("/content/img_0883.jpeg")
foreground = extract_foreground(original_image ,bin_mask)
# cv2_imshow(foreground)
# img_rgb = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
cv2.imshow("Foreground_with_transparent_background", foreground)
# file name with different extensions gives different display, in reality image is with black background


# Resize image to a width of 250
img = imutils.resize(foreground,width=250)

# Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors 
dominantColors, r, g, b = extractDominantColor(img,hasThresholding=True)
colour_name_dress = nearest_colour( colours_list, (r, g, b) )[3]
print(colour_name_dress)

cv2.imwrite('Foreground({!s} ({!s}, {!s}, {!s}))_with_transparent_background.png'.format(colour_name, r, g, b), foreground)

# if cv2.waitKey(0) & 0xFF == ord('q'):
#   cv2.destroyAllWindows()
