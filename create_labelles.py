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

    # Bitwise AND operation between the mask and the image to get the pixels within the rectangle
    result = cv2.bitwise_or(img, mask)

    # Threshold the result to get the binary image
    result[result > 0] = 1

    return result

def fill_angle(img, rect_coords, angle):
    mask = np.zeros((480,640))

    # Convert rectangle coordinates to integer
    rect_coords = np.array(rect_coords, dtype=np.int32)
    print (rect_coords)
    print(mask.shape)
    # Fill the polygon defined by the rectangle coordinates with white color (255)
    cv2.fillPoly(mask, [rect_coords], angle)
    print("max")
    print(np.max(mask))
    print("max")
   
    print(mask.shape)
    
    # Bitwise AND operation between the mask and the image to get the pixels within the rectangle
    #result = cv2.bitwise_or(img, mask)
    zero_indices = np.where((img == 0) & (mask != 0))
    img[zero_indices] = mask[zero_indices]
    print("imgdata")
    print(np.max(img))
    return img    

def angle_between_vectors(v1, v2):
    print(v2)
    print(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1, v2)
    cross_product = np.cross(v1, v2)
    angle = np.arctan2(cross_product, dot_product)
    return angle


base_path = "../archive"
rectangle = "/01/pcd0100cpos.txt"
img = "/01/pcd0100r.png"
imgs = cv2.imread(base_path + img, cv2.IMREAD_UNCHANGED)
new_img_q = np.zeros((480,640))
new_img_phi = np.zeros((480,640))
new_img_grip_width = np.zeros((480,640))

f = open(base_path + rectangle, "r")
line = f.readline()
i = 1
all_xy = []
color = (0,0,255)
norm_vec = np.array([1,0])
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
        width = all_xy[1][0] - all_xy[0][0]
        defo = abs(width)
        height = all_xy[3][1] - all_xy[0][1]
        width_3 = round (width / 3)
        height_3 = round (height / 3)
        # get centre third of binding box
        direction_vec = np.array([all_xy[1][0] -all_xy[0][0] ,all_xy[1][1] -all_xy[0][1]] )
        phi = angle_between_vectors(norm_vec, direction_vec)

        all_xy[0] = (all_xy[0][0] + width_3, all_xy[0][1] + height_3 )


        all_xy[1] = (all_xy[1][0] - width_3,  all_xy[1][1] + height_3)


        all_xy[2] = (all_xy[2][0] - width_3, all_xy[2][1] - height_3 )


        all_xy[3] = (all_xy[3][0] + width_3, all_xy[3][1] - height_3 )


        new_img_q = fill_rotated_rectangle(new_img_q, all_xy)
        new_img_phi =fill_angle(new_img_phi, all_xy, phi)
        new_img_grip_width =fill_angle(new_img_grip_width, all_xy, defo / 150)
        
        all_xy = []
    i += 1
    

new_img_q *= 255
print("=======")
print(np.max(new_img_q))

cv2.imshow('Image', imgs)
cv2.imshow('greyscale', new_img_q)
cv2.normalize(new_img_phi, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('greyscale2', new_img_phi)
print(np.max(new_img_phi))
cv2.normalize(new_img_grip_width, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('greyscale3', new_img_grip_width)

cv2.waitKey(0)
cv2.destroyAllWindows()