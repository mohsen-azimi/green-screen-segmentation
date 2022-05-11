import cv2
import numpy as np
import glob
import os

# Label ids of the dataset (BGR)
mask_colors = {
    "circular": (255, 0, 0),
    "rectangular": (255, 0, 0),
    "girder": (255, 255, 0),
}
''' "(0, 0, 0)": 0,  # outlier
    "(255, 0, 0)": 1,  # circular
    "(255, 255, 0)": 2,  # rectangular
    "(128, 0, 255)": 3,  # girder
    "(255, 128, 0)": 4,  # ...
    "(0, 0, 255)": 5,  # ...
    "(128, 255, 255)": 6,  # ...
    "(0, 255, 0)": 7,  # ...
    "(128, 128, 128)": 8  # ...'''

object_color = mask_colors["rectangular"]

video = cv2.VideoCapture(1)

lab_background = cv2.imread("rgb.png")


def nothing(x):
    pass


cv2.namedWindow('control')
cv2.resizeWindow('control', 500, 500)

w, h = 640, 480
cv2.createTrackbar("LH", "control", 0, 255, nothing)
cv2.createTrackbar('LS', "control", 0, 255, nothing)
cv2.createTrackbar("LV", "control", 0, 255, nothing)
cv2.createTrackbar("UH", "control", 255, 255, nothing)
cv2.createTrackbar("US", "control", 255, 255, nothing)
cv2.createTrackbar("UV", "control", 40, 255, nothing)

cv2.createTrackbar("crop_T", "control", 173, int(h / 2), nothing)
cv2.createTrackbar("crop_B", "control", 30, int(h / 2), nothing)
cv2.createTrackbar("crop_L", "control", 155, int(w / 2), nothing)
cv2.createTrackbar("crop_R", "control", 190, int(w / 2), nothing)

cv2.createTrackbar('save', 'control', 0, 1, nothing)

# Get the latest generated file from the output directory
if len(os.listdir('output/output')) == 0:
    i = 0
else:
    list_of_created = glob.glob('output/image/*')  # * means all if need specific format then *.csv
    latest_file = max(list_of_created, key=os.path.getctime)
    i = int(latest_file.split()[1][:-4])

while True:

    ret, input_frame = video.read()

    input_frame = cv2.resize(input_frame, (w, h))
    lab_background = cv2.resize(lab_background, (w, h))

    l_green = np.array([cv2.getTrackbarPos("LH", "control"),
                        cv2.getTrackbarPos("LS", "control"),
                        cv2.getTrackbarPos("LV", "control")])

    u_green = np.array([cv2.getTrackbarPos("UH", "control"),
                        cv2.getTrackbarPos("US", "control"),
                        cv2.getTrackbarPos("UV", "control")])

    mask = cv2.inRange(input_frame, l_green, u_green)
    mask = cv2.bitwise_not(mask)  # invert mask; i.e., 255-mask
    crop = np.array([cv2.getTrackbarPos("crop_T", "control"),
                     cv2.getTrackbarPos("crop_B", "control"),
                     cv2.getTrackbarPos("crop_L", "control"),
                     cv2.getTrackbarPos("crop_R", "control")])

    mask[:crop[0], :] = 0
    mask[(mask.shape[0] - crop[1]):, :] = 0

    mask[:, :crop[2]] = 0
    mask[:, (mask.shape[1] - crop[3]):] = 0

    mask = cv2.medianBlur(mask, 3)
    kernel = np.ones((3, 3), 'uint8')
    # mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask_colored = np.zeros([h, w, 3], dtype=np.uint8)
    mask_colored[:, :] = [object_color[0], object_color[1], object_color[2]]

    res = cv2.bitwise_and(input_frame, input_frame, mask=cv2.bitwise_not(mask))  # invert back the mask
    image = input_frame - res
    image = np.where(image == 0, lab_background, image)
    mask2 = np.where((input_frame - res) == 0, np.zeros([h, w, 3], dtype=np.uint8), mask_colored)

    cv2.imshow("input_frame", np.vstack((input_frame, image, mask2)) )

    s = cv2.getTrackbarPos('save', 'control')

    if s == 1:
        print(f"Saving the image {i} ...")
        cv2.imwrite(f'output/input/frame {i}.png', input_frame)
        cv2.imwrite(f'output/binary_mask/frame {i}.png', mask)
        cv2.imwrite(f'output/color_mask/frame {i}.png', mask2)
        cv2.imwrite(f'output/output/frame {i}.png', image)
        # cv2.imwrite('output/mask.png', mask)
        i += 1
        s = cv2.setTrackbarPos('save', 'control', 0)

    # If we've waited at least 1 ms And we've pressed the Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
