import cv2
import numpy as np


def fill_rotated_rectangle(img, rect_coords):
    # Create a blank mask with the same size as the image
    mask = np.zeros((480,640))

    # Convert rectangle coordinates to integer
    rect_coords = np.array(rect_coords, dtype=np.int32)
    print (rect_coords)
    print(mask.shape)
    # Fill the polygon defined by the rectangle coordinates with white color (255)
    cv2.fillPoly(mask, [rect_coords], 255)
    print("max")
    print(np.max(mask))
    print("max")
   
    print(mask.shape)
    
    # Bitwise AND operation between the mask and the image to get the pixels within the rectangle
    result = cv2.bitwise_or(img, mask)

    # Threshold the result to get the binary image
    result[result > 0] = 1

    return result





base_path = "../archive"
rectangle = "/01/pcd0100cpos.txt"
img = "/01/pcd0100r.png"
imgs = cv2.imread(base_path + img, cv2.IMREAD_UNCHANGED)
new_img_q = np.zeros((480,640))

f = open(base_path + rectangle, "r")
line = f.readline()
i = 1
all_xy = []
color = (0,0,255)
while line:
    data = line.strip().split(" ")
    location = np.array(data, dtype=np.cfloat)
    all_xy.append((int(np.round(location[0])), int(np.round(location[1]))))
    cv2.circle(imgs, (int(np.round(location[0])), int(np.round(location[1]))), 2, color, 1)

    line = f.readline()
    if i % 4 == 0:
        if color == (0,0,255):
            color = (255,0,0)
        else:
            color = (0,0,255)
        new_img_q = fill_rotated_rectangle(new_img_q, all_xy)
    i += 1


new_img_q *= 255
print("=======")
print(np.max(new_img_q))

cv2.imshow('Image', imgs)
cv2.imshow('greyscale', new_img_q)
cv2.waitKey(0)
cv2.destroyAllWindows()