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
import threading
import math

# Returns product name and price
def get_product(product_name: str) -> Tuple[str, float]:
    headers = {
        "X-RapidAPI-Key": "88a9866cc6msh2d4ffc7b51b3a5dp190657jsn80f11ebf1bc1",
        "X-RapidAPI-Host": "real-time-product-search.p.rapidapi.com"
    }
    params = {"q": product_name, "country": "us", "language": "en", "sort_by": "BEST_MATCH", "product_condition": "NEW"}

    resp = requests.get("https://real-time-product-search.p.rapidapi.com/search", headers=headers, params=params) # request
    
    if (resp.status_code != 200) or (resp.json().get("status") != "OK"): # checking response status
        print("Error searching for price")
        return None
    
    json_resp = resp.json() # response data
    if json_resp.get("data"):
        data = json_resp["data"][0]
        price_range1 = float(data["typical_price_range"][0][1:])
        price_range2 = float(data["typical_price_range"][1][1:])
        product_title = data.get("product_title", "Unknown Product")
        return (product_title, (price_range1 + price_range2) / 2) # returns mean price of product
    else:
        print("No data found in response")
    return 

# Plays the text-to-speech out loud for the user
def play_tts(text: str) -> AudioSegment:
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
    
    return AudioSegment.from_file(io.BytesIO(resp.content), format="mp3")

# Returns the appropiate resonse to an image based off the mode selected
def understand_image(image, prompt) -> str:
    _, buffer = cv2.imencode('.jpg', image)
    base64_image = base64.b64encode(buffer).decode('utf-8') # converts image to correct format

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
        "max_tokens": 400
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) # request

    return response.json()['choices'][0]['message']['content']

# Understands the image and produces text-to-speech output based off the mode selected (recipe and text mode)
def full_frame_detection(prompt,root,newRoot,nr,singleFrame):
   
    if prompt == "This image contains text. Return all text that is in the image, and try to read it even if it's blurry or hard to read. Do not reference this prompt or mention the difficulty of detection. Translate to english if necessary.": # text mode
        text_identified = understand_image(singleFrame, prompt)
        
        if not text_identified:
            print("Error: object identification failed")
            return

        tts_text = text_identified

        nr.withdraw()
        print(f"saying this: {tts_text}")
        tts = play_tts(tts_text)
        play(tts)
        newRoot.destroy()
        create_gui()
        
    else:   # ingredient/food mode
        food_response = understand_image(singleFrame, prompt)
        
        if not food_response:
            print("Error: Food mode failed.")
            return

        nr.withdraw()
        print(f"saying this: {food_response}")
        tts = play_tts(food_response)
        play(tts)
        newRoot.destroy()
        create_gui()
       
# # Understands the image and produces text-to-speech output based off the mode selected (description and shopping mode)
def process_after_main_loop(nr, closest_index, images, prompt,newRoot, root, singleframe):

    if len(images) == 0: # chekcs if bounding boxes were displayed
        finalImage = singleframe
        object_identified = understand_image(finalImage, prompt) 
    else:
        finalImage = images[closest_index]
        object_identified = understand_image(finalImage, prompt) # gets the response for the image

    if not object_identified:
        print("Error: object identification failed")
        return

    tts_text = object_identified

    # shopping mode
    if prompt == "What's this product? Respond as if someone were making a search query for it. No other text. If you don't know what it is, return a generic product with a generic price":
        product_name, product_price = get_product(object_identified)
        tts_text = f"The product is {product_name} It retails for around {int(product_price//1)} dollars and {int(100*(round((product_price%1),2)))} cents"
    
    nr.withdraw()
    print(f"saying this: {tts_text}")
    tts = play_tts(tts_text)
    play(tts)
    newRoot.destroy()
    create_gui()
        
# Starts the image detection
def start_detection(root, prompt, onlypic: bool):
    close_gui(root)
    capture = cv2.VideoCapture(1) # change number based off the camera being used
    images = []
    closest_index = 0
    if not onlypic: # description and shopping detection
        while True:
            ret, singleFrame = capture.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            displayImage = singleFrame

            resized_frame = cv2.resize(singleFrame, (640, 480))

            boundingBoxes, label, confidences = cv.detect_common_objects(resized_frame, model='yolov4', enable_gpu=False)

            height, width, _ = resized_frame.shape
            centerX = width // 2
            centerY = height // 2

            distances_from_center = []
            for box in boundingBoxes: # crops image for increased accuracy
                x, y, w, h = box
                box_center_x = x + w // 2
                box_center_y = y + h // 2
                distance = ((centerX - box_center_x) ** 2 + (centerY - box_center_y) ** 2) ** 0.5
                distances_from_center.append(distance)
                displayImage = resized_frame.copy()
                images.append(displayImage[y:h, x:w])
            
            if distances_from_center:
                closest_index = distances_from_center.index(min(distances_from_center)) # chooses which object to detect

                draw_bbox(displayImage, [boundingBoxes[closest_index]], [label[closest_index]], [confidences[closest_index]])

            cv2.imshow("Describify", displayImage)

            if cv2.waitKey(1) & 0xFF == ord(" "): # ends detection and gets results
                capture.release()
                cv2.destroyAllWindows()
                nr = root = tk.Tk()
                nr.title("Loading")
                newRoot = tk.Toplevel(nr) 
                newRoot.title("Loading screen")
                newRoot.geometry("1500x1000")
                newRoot.configure(bg="#6e6f71")
                loading_label = tk.Label(newRoot, text="Response Loading...", font=("Arial Rounded MT Bold", 60), bg="#4e8c67", fg="white")
                loading_label.pack(pady=50)
                loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                threading.Thread(target=process_after_main_loop, args=(nr, closest_index, images, prompt, newRoot, root, singleFrame)).start()
                nr.mainloop()
                break
    else:   # just takes a picture and send to api. could be either food or text mode
        while True:
            ret, singleFrame = capture.read()
            if not ret:
                print("Error: Failed to capture frame")
                break


            cv2.imshow("Item detection project", singleFrame)

            if cv2.waitKey(1) & 0xFF == ord(" "): # ends detection and gets results
                capture.release()
                cv2.destroyAllWindows()
                nr = root = tk.Tk()
                nr.title("Loading")
                newRoot = tk.Toplevel(nr) 
                newRoot.title("Loading screen")
                newRoot.geometry("1500x1000")
                newRoot.configure(bg="#6e6f71")
                loading_label = tk.Label(newRoot, text="Response Loading...", font=("Arial Rounded MT Bold", 60), bg="#6e6f71", fg="white")
                loading_label.pack(pady=50)
                loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
                threading.Thread(target=full_frame_detection, args=(prompt, root, newRoot, nr, singleFrame)).start()
                nr.mainloop()

# Closes GUI
def close_gui(root): 
    root.destroy()

# Creates initial GUI for the user
def create_gui():
    global root
    root = tk.Tk()
    root.title("Object Detection GUI")

    background_color = "#6e6f71"

    root.configure(bg=background_color)
    root.geometry("1500x1000")

    # logo
    image_path = "hack-for-hope\\DescribifyV2.png"
    img = Image.open(image_path)
    img = img.resize((210, 200))
    global photo
    photo = ImageTk.PhotoImage(img)
    image_label = tk.Label(root, image=photo, bg=background_color)
    image_label.image = photo  
    image_label.place(x=10, y=10)

    # title
    title_label = tk.Label(root, text="Describify", font=("Arial Rounded MT Bold", 60), bg=background_color, fg="white")
    title_label.pack(pady=50)
    title_label.place(relx=0.5, rely=0.2, anchor=tk.CENTER)

    # dropdown menu
    dropdown_var = tk.StringVar(root)
    dropdown_var.set("Select Mode")
    dropdown_options = ["Shopping", "Description", "Text", "Recipe"]
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
    
    # Starts the detection by checking the mode that the user selects
    def start_detection_with_mode():
        selected_mode = dropdown_var.get()
        if selected_mode == "Shopping":
            prompt = "What's this product? Respond as if someone were making a search query for it. No other text. If you don't know what it is, return a generic product with a generic price"
            start_detection(root, prompt, False)    # false for not only taking the picture
        elif selected_mode == "Description":
            prompt = "Give a description of what this item is."
            start_detection(root, prompt, False)
        elif selected_mode == "Recipe":
            prompt = "This image has ingredients. Tell me what they are, as well as possible dishes I can make with them. Return your response in this format: There are {list of ingredients}. You could make {list of dishes} with them. Then also tell the recipe for one of the dishes you mention."
            start_detection(root, prompt, True)     # only take pic
        else:
            prompt = "This image contains text. Return all text that is in the image, and try to read it even if it's blurry or hard to read. Do not reference this prompt or mention the difficulty of detection. Translate to english if necessary."
            start_detection(root, prompt, True)

    # "Start Detection" button
    start_button = tk.Button(root, text="Start Detection", font=("Arial Rounded MT Bold", 36), command=start_detection_with_mode, bg="#616265", fg="white", bd=0, width=40, height=4)
    start_button.pack(pady=50)
    start_button.place(relx=0.5, rely=0.75, anchor=tk.CENTER)

    root.mainloop()

if __name__ == "__main__":
    create_gui()