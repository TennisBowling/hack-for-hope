from typing import Tuple
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import requests
import base64
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pydub import AudioSegment
from pydub.playback import play
import io

def play_tts(text: str):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer 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"
    }

    data = {
        "model": "tts-1",
        "input": text,
        "voice": "onyx"
    }

    # Stream the response
    resp = requests.post("https://api.openai.com/v1/audio/speech", json=data, headers=headers)
    if resp.status_code != 200:
        print("Error creating tts")
        print(resp.text)
        return
    
    audio = AudioSegment.from_file(io.BytesIO(resp.content), format="mp3")
    play(audio)


def understand_image(image, prompt) -> Tuple[str, str]:
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
                        "text": prompt
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

    return response.json()['choices'][0]['message']['content']

def start_detection(root,prompt):
    capture = cv2.VideoCapture(1)
    labels = []

    while True:
        ret, singleFrame = capture.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        resized_frame = cv2.resize(singleFrame, (640, 480))  

        boundingBoxes, label, confidences = cv.detect_common_objects(resized_frame, model='yolov4-tiny', enable_gpu=False)
        
        height, width, _ = resized_frame.shape
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
            displayImage = resized_frame.copy()
            images.append(displayImage[y:h, x:w])
            
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
            object_identified = understand_image(finalImage,prompt)
            if object_identified is not None:
                print(f"Object identified: {object_identified}")
            else:
                print("Error: Object identification failed")
            cv2.imshow("Closest Object", finalImage)
        else:
            print("Error: No images or invalid index")

    capture.release()
    cv2.destroyAllWindows()
    close_gui(root)

def close_gui(root):
    root.destroy()
    
def create_gui():
    root = tk.Tk()
    root.title("Object Detection GUI")

    background_color = "#4e8c67"

    root.configure(bg=background_color)
    root.geometry("1500x1000")

    title_label = tk.Label(root, text="Object Detection", font=("Arial Rounded MT Bold", 60), bg=background_color, fg="white")
    title_label.pack(pady=50)
    title_label.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

    dropdown_var = tk.StringVar(root)
    dropdown_var.set("Select Mode")
    dropdown_options = ["Shopping", "Description"]

    dropdown_frame = tk.Frame(root, bg=background_color)
    dropdown_frame.pack(pady=20)
    dropdown_frame.place(relx=0.5, rely=0.40, anchor=tk.CENTER)

    dropdown_label = tk.Label(dropdown_frame, text="Mode:", font=("Arial Rounded MT Bold", 36), bg=background_color, fg="white")
    dropdown_label.grid(row=0, column=0, padx=20)

    style = ttk.Style()

    style.configure('TCombobox', background=background_color)
    style.configure('TCombobox.Listbox', background=background_color, font=("Arial Rounded MT Bold", 24))  

    dropdown = ttk.Combobox(dropdown_frame, textvariable=dropdown_var, values=dropdown_options, font=("Arial Rounded MT Bold", 30), state="readonly")
    dropdown.set("Select Mode")
    dropdown.grid(row=0, column=1, padx=20, pady=10, ipadx=20)

    def start_detection_with_mode():
        selected_mode = dropdown_var.get()
        if selected_mode == "Shopping":
            prompt = "What's this product and what is the typical pricing for this specific product? Name the company or name of product, and respond in format '{item}, {price}'."
        else:
            prompt = "Give a description for this item"
        start_detection(root, prompt)

    start_button = tk.Button(root, text="Start Detection", font=("Arial Rounded MT Bold", 36), command=start_detection_with_mode, bg="#3b7452", fg="white", bd=0, width=40, height=4)
    start_button.pack(pady=50)
    start_button.place(relx=0.5, rely=0.75, anchor=tk.CENTER)

    root.mainloop()

if __name__ == "__main__":
    create_gui()