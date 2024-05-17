



import cv2
import numpy as np

import openai

# Load YOLOv3 model
yolo_net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")


# Load class labels
yolo_classes = []
with open('coco.names', 'r') as f:
    yolo_classes = f.read().splitlines()

# Function to perform YOLO object detection on a single image
def perform_yolo_detection(img):
    detected_objects = []
    
    print("[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]")

    height, width, _ = img.shape  # Get image dimensions

    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo_net.setInput(blob)

    # Get YOLO output
    output_layers_names = yolo_net.getUnconnectedOutLayersNames()
    layer_outputs = yolo_net.forward(output_layers_names)

    # Parse YOLO output
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.7:
                object_name = yolo_classes[class_id]
                detected_objects.append(object_name)

    return detected_objects







# # start

def generate_summary(path):
    
    print("///////////////////////////////")

    # Open a live camera feed
    video_capture = cv2.VideoCapture(path)  # Change the path to your video file
    summary_objects = set()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform YOLO object detection on the frame
        detected_objects = perform_yolo_detection(frame)

        # Add detected objects to summary
        summary_objects.update(detected_objects)
        
        print(detected_objects,"//////////detected objects")
        print(summary_objects,"//////////detected objects")
        

    video_capture.release()
    
    print("//...mmm")

    # Generate summary sentence
    summary_sentence = "This video contains "
    print(summary_sentence,"//////////sentence")
    if summary_objects:
        summary_sentence += ", ".join(summary_objects)
    else:
        summary_sentence += "no objects"
    summary_sentence += "."

    print("Summary Sentence:", summary_sentence)


    

    # # Set your OpenAI API key here
    
    # # 
    # openai.api_key = "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"

    # system = [{"role": "system", "content": "You are a chatbot who enjoys Python programming."}]
    # chat = []

    # # Replace the following with the paragraph you want to summarize
    # paragraph_to_summarize = "Generate 2 or 3 Sentence,"+summary_sentence
    
    # print("paraaaa/////////")

    # # Append user input to the chat
    # user = [{"role": "user", "content": paragraph_to_summarize}]
    # print("11"*200)
    # response = openai.ChatCompletion.create(
    #     messages=system + chat[-20:] + user,
    #     model="gpt-3.5-turbo", top_p=0.5, stream=True
    # )
    
    # print("22")

    # reply = ""
    
    # print("///null reply///")
    # for delta in response:
    #     if not delta['choices'][0]['finish_reason']:
    #         word = delta['choices'][0]['delta']['content']
    #         reply += word

    # chat.append(user[0])
    # chat.append({"role": "assistant", "content": reply})

    # # Print the generated summary
    # print("Generated Summary:")
    # print(reply)
    # return reply
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # complaint = "car bike car bus"
    # summary_objects = ['person', 'car', 'truck', 'traffic light']

    # Concatenate summary objects with the complaint


    # Load the pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"  # You can use other model names too
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Generate text
    input_text = summary_sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(input_ids, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    # Escape the single quote in generated_text
    escaped_generated_text = generated_text.replace("'", "''")
    print("escaped_generated_text : ",escaped_generated_text)
    return escaped_generated_text




