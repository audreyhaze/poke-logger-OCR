import cv2
import numpy as np

img = np.zeros((500, 500, 3), dtype=np.uint8)

cv2.namedWindow("image")

while True:
    cv2.imshow("image", img)

    if cv2.waitKey(10) != -1:
        break

cv2.destroyAllWindows()