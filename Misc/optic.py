import cv2
import numpy as np

def detect_optic_disc(image_path):
    imge = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, imge = cv2.threshold(imge, 127, 255, cv2.THRESH_BINARY)
    height, width = imge.shape

    max_intensity = 0
    max_intensity_position = (0, 0)

    window_size = 29

    for y in range(0, height - window_size + 1, window_size):
        for x in range(0, width - window_size + 1, window_size):
            window = imge[y:y + window_size, x:x + window_size]

            intensity = np.sum(window)

            if intensity > max_intensity:
                max_intensity = intensity
                max_intensity_position = (x, y)

    x, y = max_intensity_position
    most_intense_region = imge[y:y + window_size, x:x + window_size]

    box_color = (0, 255, 0)
    box_thickness = 4
    image_with_rectangle = cv2.imread(image_path)
    cv2.rectangle(image_with_rectangle, (x, y), (x + window_size, y + window_size), box_color, box_thickness)

    circle_center = (x + window_size // 2, y + window_size // 2)
    circle_radius = int(window_size * 0.85)
    mask = np.zeros_like(imge)
    cv2.circle(mask, circle_center, circle_radius, (255), -1)
    masked_image = cv2.bitwise_and(imge, mask)

    print(f"the location : {x},{y} ")
    print(f"the intensity : {max_intensity}")
    #cv2_imshow(image_with_rectangle)
    #cv2_imshow(masked_image)
    cv2.imshow("image_with_rectangle", image_with_rectangle)
    cv2.imshow("masked_image", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return most_intense_region

image_path = "/home/rayuga/Documents/DataSet/GAN/minor_data/256x256_original/140_left.png"
most_intense_region = detect_optic_disc(image_path)