import cv2
import rembg
import numpy as np

input_image_path = r'C:\Users\Sameer\Downloads\AI Engineer - ProductizeTech Assignment -20240201T024125Z-001\AI Engineer - ProductizeTech Assignment\TEST IMAGES\3.jpg'
image = cv2.imread(input_image_path)
cv2.imshow('Input Image', image)
cv2.waitKey(1)

roi = cv2.selectROI('Select Object', image)
cv2.destroyAllWindows()

x, y, w, h = roi
roi_cropped = image[y:y+h, x:x+w]

roi_buffer = cv2.imencode('.png', roi_cropped)[1].tostring()

output_buffer = rembg.remove(roi_buffer)

output_data = np.frombuffer(output_buffer, dtype=np.uint8)
output_image = cv2.imdecode(output_data, cv2.IMREAD_UNCHANGED)

contours, _ = cv2.findContours(cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

cv2.imshow('Object with Outline', output_image)
cv2.waitKey(0)

output_image_path = r'C:\Users\Sameer\Desktop\project\output_image.png'  
cv2.imwrite(output_image_path, output_image)
print(f"Output image saved at: {output_image_path}")

cv2.destroyAllWindows()
