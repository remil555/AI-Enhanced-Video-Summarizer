video_capture = cv2.VideoCapture("overpass.mp4")  # Change the path to your video file
summary_objects = set()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Perform YOLO object detection on the frame
    detected_objects = perform_yolo_detection(frame)

    # Add detected objects to summary
    summary_objects.update(detected_objects)

video_capture.release()

# Generate summary sentence
summary_sentence = "This video contains "
if summary_objects:
    summary_sentence += ", ".join(summary_objects)
else:
    summary_sentence += "no objects"
summary_sentence += "."

# print("Summary Sentence:", summary_sentence)


import openai

# Set your OpenAI API key here
openai.api_key = "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"

system = [{"role": "system", "content": "You are a chatbot who enjoys Python programming."}]
chat = []

# Replace the following with the paragraph you want to summarize
paragraph_to_summarize = "Generate 2 or 3 Sentence,"+summary_sentence

# Append user input to the chat
user = [{"role": "user", "content": paragraph_to_summarize}]
response = openai.ChatCompletion.create(
    messages=system + chat[-20:] + user,
    model="gpt-3.5-turbo", top_p=0.5, stream=True
)

reply = ""
for delta in response:
    if not delta['choices'][0]['finish_reason']:
        word = delta['choices'][0]['delta']['content']
        reply += word

chat.append(user[0])
chat.append({"role": "assistant", "content": reply})

# Print the generated summary
print("Generated Summary:")
print(reply)
