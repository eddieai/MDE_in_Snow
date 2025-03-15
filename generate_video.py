import cv2
import os

image_folder = 'inference_weighted_linear/'
video_name = 'video_weighted_linear.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 5, (width,height))

for image in sorted(images):
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()