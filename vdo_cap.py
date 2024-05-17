import torch
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertForMaskedLM
import cv2
import numpy as np

# Load pre-trained language model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Load pre-trained image recognition model
# Example: resnet50
resnet_model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
resnet_model.eval()

# Open video file
video_file = "overpass.mp4"
cap = cv2.VideoCapture(video_file)

# Function to preprocess video frames
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    image = image.unsqueeze(0)
    return image

# Process each frame
captions = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for image recognition
    image = preprocess_image(frame)

    # Extract image features using pre-trained model
    with torch.no_grad():
        features = resnet_model(image)

    # Convert image features to text using BERT language model
    caption = ""
    for i in range(len(features)):
        text = tokenizer.decode(features[i].argmax().item())
        caption += text + " "

    captions.append(caption)

cap.release()

# Generate video captions
for caption in captions:
    input_text = "[CLS] " + caption + " [SEP]"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)
    output = model(input_ids)
    predicted_token_index = torch.argmax(output[0][0, mask_token_index]).item()
    predicted_token = tokenizer.decode([predicted_token_index])
    generated_caption = input_text.replace('[MASK]', predicted_token)
    print("Generated Caption:", generated_caption)
