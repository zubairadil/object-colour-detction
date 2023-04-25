import cv2
import numpy as np
from hair_utils import HairSegmentation
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt



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
    	print(max_percentage, r, g, b)
    
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
  estimator = KMeans(n_clusters=number_of_colors, random_state=0)
  
  # Fit the image
  estimator.fit(img)
  
  # Get Colour Information
  colorInformation, r, g, b, max_percentage = getColorInformation(estimator.labels_,estimator.cluster_centers_,hasThresholding)
  # r+=28
  # g+=28
  # b+=18.5
  print(max_percentage, r, g, b)
  return colorInformation, r, g, b

# def plotColorBar(colorInformation):
#   #Create a 500x100 black image
#   color_bar = np.zeros((100,500,3), dtype="uint8")
  
#   top_x = 0
#   for x in colorInformation:    
#     bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

#     color = tuple(map(int,(x['color'])))
  
#     cv2.rectangle(color_bar , (int(top_x),0) , (int(bottom_x),color_bar.shape[0]) ,color , -1)
#     top_x = bottom_x

#   return color_bar
colours_list = ((255, 0,0, "Red"),
                
              
                (170, 139, 103,"Blonde"),
                (251, 231, 161,"Blonde"),

                (255,245,238,"White"),
                (0,0,0,"Black"),
                (192, 192, 192,"Gray"),
                
                (255,20,147,"Pink"),
                (128,0,0,"Maroon"),
                (165,42,42,"Brown"))

def nearest_colour( subjects, query ):
    return min( subjects, key = lambda subject: sum( (s - q) ** 2 for s, q in zip( subject, query ) ) )


# Initialize webcam
cap = cv2.VideoCapture(6)
webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(webcam_width)
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(webcam_height)

# Inialize hair segmentation model
hair_segmentation = HairSegmentation(webcam_width, webcam_height)

while cap.isOpened():

	# Read frame
	ret, frame = cap.read()

	img_height, img_width, _ = frame.shape


	if not ret:
		continue

	#converting original image into foreground image
	# Flip the image horizontally
	original_image = cv2.flip(frame, 1)

	# Segment hair
	hair_mask = hair_segmentation(original_image)
	background = np.zeros(hair_mask.shape)
	bin_mask = np.where(hair_mask, 255, background).astype(np.uint8)

	# displays hair with transparent background
	foreground = extract_foreground(original_image ,bin_mask)

	
	cv2.imshow("Binary Mask", bin_mask)
	cv2.imshow("Extracted image", foreground)
	cv2.imwrite('Foreground_with_transparent_background.png', foreground)

	# Resize image to a width of 250
	img = imutils.resize(foreground,width=250)

	# Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors 
	dominantColors, r, g, b = extractDominantColor(img,hasThresholding=True)
	colour_name = nearest_colour( colours_list, (r, g, b) )[3] 
	print(colour_name) 


	# #Show in the dominant color as bar
	# print("Color Bar")
	# colour_bar = plotColorBar(dominantColors)
	# plt.axis("off")
	# plt.imshow(colour_bar)
	# plt.show()

	# actual_name, closest_name = get_colour_name(colors[index_color])
	# print('Actual name -> ' + str(actual_name) + ', closest_name -> ' + closest_name)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
