import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import requests
import base64

def understand_image(image) -> str:
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer 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"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's this product and what is the typical pricing for this specific product? Name the company or name of product, and respond in format '{item}, {price}'."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 150
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()['choices'][0]

capture = cv2.VideoCapture(2)
labels = []

while True:
    ret, singleFrame = capture.read()
    boundingBoxes, label, confidences = cv.detect_common_objects(singleFrame, model='yolov4-tiny')
    
    height, width, _ = singleFrame.shape
    centerX = width // 2
    centerY = height // 2
    
    distances_from_center = []
    images = []
    for box in boundingBoxes:
        x, y, w, h = box
        box_center_x = x + w // 2
        box_center_y = y + h // 2
        distance = ((centerX - box_center_x) ** 2 + (centerY - box_center_y) ** 2) ** 0.5
        distances_from_center.append(distance)
        displayImage = singleFrame.copy()
        images.append(displayImage[y:h,x:w])
        
    if distances_from_center:
        closest_index = distances_from_center.index(min(distances_from_center))
        
        draw_bbox(displayImage, [boundingBoxes[closest_index]], [label[closest_index]], [confidences[closest_index]])
        
        if label[closest_index] not in labels:
            labels.append(label[closest_index])
        
        cv2.imshow("Item detection project", displayImage)
    
    if cv2.waitKey(1) & 0xFF == ord(" "):
        break

if labels:
    closestObject = labels[0]
    print("Closest object detected:", closestObject)
    
    if images and closest_index < len(images):
        finalImage = images[closest_index]

        cv2.imshow("Closest Object", finalImage)
        object_identified = understand_image(finalImage)
        print(f"object identified: {object_identified}")
            
        while True:
            if cv2.waitKey(1) & 0xFF == ord(" "):
                break

capture.release()
cv2.destroyAllWindows()