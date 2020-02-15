import numpy as np
import cv2
from skimage import data
from skimage.feature import match_template
import matplotlib.pyplot as plot

cp = cv2.VideoCapture('video.mp4')
count = 0
frames = []
color_frames = []

#template and window values for object crackers
# template_start_x = 665
# template_start_y = 615
# template_end_x = 890
# template_end_y = 925

# window_start_x = 500
# window_start_y = 300
# window_end_x = 1050
# window_end_y = 1000


#template and window values for object cup
template_start_x = 1265
template_start_y = 735
template_end_x = 1465
template_end_y = 925

window_start_x = 1000
window_start_y = 450
window_end_x = 1700
window_end_y = 960

while (cp.isOpened()):
    count += 1
    print("Processing frame number " + str(count))

    ret, frame = cp.read()
    if not ret:
        break

    frames.append(frame)
    color_frames.append(frame)

    if count == 1:
        colored_image = cv2.rectangle(frames[0], (template_start_x, template_start_y), (template_end_x, template_end_y), (0, 0, 255), 4)
        colored_image = cv2.rectangle(colored_image, (window_start_x, window_start_y), (window_end_x, window_end_y), (80, 0, 120), 4)
        cv2.imwrite("image_with_template.jpg", colored_image)


temp_img = cv2.rectangle(frames[0], (template_start_x, template_start_y), (template_end_x, template_end_y), (0, 0, 255), 2)
 
count = 0

temp_grey = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("greyscale_template.jpg", temp_grey)
temp_cropped = temp_grey[template_start_y:template_end_y, template_start_x:template_end_x] 

max_coord_x = []
max_coord_y = []

for src in frames:
    count += 1
    grey_source = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_cropped = grey_source[window_start_y:window_end_y, window_start_x:window_end_x]
    result = match_template(src_cropped, temp_cropped, pad_input=True)

    cur_max_x = 0
    cur_max_y = 0
    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            if result[i][j] > result[cur_max_x][cur_max_y]:
                cur_max_x = i
                cur_max_y = j
    max_coord_x.append(cur_max_x)
    max_coord_y.append(cur_max_y)

plot.imshow(result, cmap="gray")
plot.xlabel('Pixel Location in X Direction')
plot.ylabel('Pixel Location in Y Direction')
plot.show()

plot.scatter(max_coord_y, max_coord_x)
plot.xlabel('Y Pixel Shift')
plot.ylabel('X Pixel')
plot.show()

final_image = np.zeros(frames[0].shape)
for i in range(0, len(frames)):
    translate_mat = np.zeros((2, 3))
    translate_mat[0][1] = 1.0
    translate_mat[1][0] = 1.0
    translate_mat[0][2] = -(max_coord_x[i] - max_coord_x[0])
    translate_mat[1][2] = -(max_coord_y[i] - max_coord_y[0])
    shifted_image = np.zeros(frames[i].shape)
    size = shifted_image.shape
    rows = size[0]
    cols = size[1]
    shifted_image = cv2.warpAffine(
        src=frames[i], dsize=(rows, cols), M=translate_mat)
    final_image += np.divide(np.transpose(shifted_image,
                                          (1, 0, 2)), len(frames))

cv2.imwrite("result.jpg", final_image)

cp.release()
cv2.destroyAllWindows()
