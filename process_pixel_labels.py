import cv2
import os
import numpy as np
import sys

def draw_poly(img, bounding_poly):
    pts = np.array(bounding_poly, np.int32)

    #http://stackoverflow.com/a/15343106/3479446
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_corners = np.array([pts], dtype=np.int32)

    ignore_mask_color = (255,)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color, lineType=cv2.LINE_8)
    return mask

def post_process(img):
    # img = open_close(img)
    img = get_largest_cc(img)
    img = fill_holes(img)

    # img = min_area_rectangle(img)
    img, coords = improve_min_area_rectangle(img)

    return img, coords

def open_close(img):
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 15)
    dilation = cv2.dilate(erosion,kernel,iterations = 15)

    return dilation


def get_largest_cc(img):
    img = img.copy()
    ret, th = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    connectivity = 4
    output= cv2.connectedComponentsWithStats(th, connectivity, cv2.CV_32S)
    cnts = output[2][1:,4]
    largest = cnts.argmax() + 1
    img[output[1] != largest] = 0

    return img

def get_iou(gt_img, pred_img):
    inter = gt_img & pred_img
    union = gt_img | pred_img

    iou = np.count_nonzero(inter) / float(np.count_nonzero(union))

    return iou

def draw_box(img, box):
    box = np.int0(box)
    draw = np.zeros_like(img)
    cv2.drawContours(draw,[box],0,(255),-1)
    return draw

def compute_iou(img, box):
    # box = np.int0(box)
    # draw = np.zeros_like(img)
    # cv2.drawContours(draw,[box],0,(255),-1)
    draw = draw_box(img, box)
    v = get_iou(img, draw)
    return v

def step_box(img, box, step_size=1):
    best_val = -1
    best_box = None
    for index, x in np.ndenumerate(box):
        for d in [-step_size, step_size]:
            alt_box = box.copy()
            alt_box[index] = x + d

            v = compute_iou(img, alt_box)
            if best_val < v:
                best_val = v
                best_box = alt_box
    return best_val, best_box



def improve_min_area_rectangle(img):
    img = img.copy()
    _, contours,_ = cv2.findContours(img, 1, 2)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    best_val = compute_iou(img, box)
    best_box = box

    while True:
        new_val, new_box = step_box(img, best_box, step_size=1)
        # print new_val
        if new_val <= best_val:
            break
        best_val = new_val
        best_box = new_box

    return draw_box(img, best_box), best_box


def min_area_rectangle(img):
    img = img.copy()
    _, contours,_ = cv2.findContours(img, 1, 2)
    cnt = contours[0]

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    draw = np.zeros_like(img)
    cv2.drawContours(draw,[box],0,(255),-1)

    return draw


def fill_holes(img):
    im_th = img.copy()


    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    if img[0,0] != 0:
        print "WARNING: Filling something you shouldn't"
    cv2.floodFill(im_floodfill, mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    return im_out

if __name__ == "__main__":
    pred_folder = sys.argv[1]
    out_folder = sys.argv[2]

    pred_imgs = {}
    for root, folders, files in os.walk(pred_folder):
        for f in files:
            if f.endswith(".png"):
                pred_imgs[f] = os.path.join(root, f)

    for k in pred_imgs:
        pred_img = cv2.imread(pred_imgs[k], 0)
        post_img = post_process(pred_img)

        cv2.imwrite(os.path.join(out_folder, k), post_img)
