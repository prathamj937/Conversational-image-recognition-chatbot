from flask import Flask, request, jsonify, session, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import torch

app = Flask(__name__)
app.secret_key = "your-very-secure-secret-key"  # Replace with something strong & secure

# Git caption pipeline
git_pipe = pipeline("image-to-text", model="microsoft/git-large-textcaps")

# Chatbot setup (DialoGPT)
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Model paths
MODEL_DIR = "models"
model_files = {
    "flower": os.path.join(MODEL_DIR, "flower.onnx"),
    "bird": os.path.join(MODEL_DIR, "bird.onnx"),
    "dog": os.path.join(MODEL_DIR, "dog.onnx"),
    "landmark": os.path.join(MODEL_DIR, "landmark.onnx")
}

# Check models exist
for name, path in model_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")

# Load ONNX sessions
flower_session = ort.InferenceSession(model_files["flower"])
bird_session = ort.InferenceSession(model_files["bird"])
dog_session = ort.InferenceSession(model_files["dog"])
landmark_session = ort.InferenceSession(model_files["landmark"])

# Labels
dog_list = ["Bulldog", "Chihuahua", "Dobermann", "German Shepherd", "Golden Retriever", "Husky", "Labrador Retriever", "Pomeranian", "Pug", "Rottweiler", "Street dog"]
flower_list = ["Jasmine", "Lavender", "Lily", "Lotus", "Orchid", "Rose", "Sunflower", "Tulip", "Daisy", "Dandelion"]
bird_list = ["Crow", "Eagle", "Flamingo", "Hummingbird", "Parrot", "Peacock", "Pigeon", "Sparrow", "Swan"]
landmark_list = ["The Agra Fort", "Ajanta Caves", "Alai Darwaza", "Amarnath Temple", "The Amber Fort", "Basilica of Bom Jesus", "Brihadisvara Temple", "Charminar", "Chhatrapati Shivaji Terminus", "Dal Lake", "The Elephanta Caves", "Ellora Caves", "Fatehpur Sikri", "Gateway of India", "Golden Temple", "Hawa Mahal", "Humayun's Tomb", "India Gate", "Jagannath Temple", "Jama Masjid", "Jantar Mantar", "Kedarnath Temple", "Konark Sun Temple", "Meenakshi Temple", "Nalanda Mahavihara", "Qutb Minar", "The Red Fort", "Taj Mahal", "Victoria Memorial"]

# Preprocessing for ONNX
# Image preprocessing for ONNX models
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # shape: (1, 224, 224, 3)
    return img_array

# Predict using ONNX
def predict(session, img_array, label_list):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    return label_list[np.argmax(outputs[0])]

# Classifier wrappers
def identify_dog(img): return predict(dog_session, preprocess_image(img), dog_list)
def identify_flower(img): return predict(flower_session, preprocess_image(img), flower_list)
def identify_bird(img): return predict(bird_session, preprocess_image(img), bird_list)
def identify_landmark(img): return predict(landmark_session, preprocess_image(img), landmark_list)

# Generate caption + classification
def generate_final_caption(img):
    caption_data = git_pipe(img)
    caption = caption_data[0]["generated_text"]
    lower = caption.lower()
    details = ""

    if "building" in lower:
        details = "The landmark is: " + identify_landmark(img)
    elif "flower" in lower:
        details = "The flower is: " + identify_flower(img)
    elif "dog" in lower or "puppy" in lower:
        details = "The dog is: " + identify_dog(img)
    elif "bird" in lower:
        details = "The bird is: " + identify_bird(img)

    final_caption = caption + "\n" + details
    return final_caption

# Chatbot response
def get_bot_response(query, chat_history_ids=None):
    new_input_ids = tokenizer.encode(query + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = new_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_input_ids], dim=-1)

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Upload and caption image
@app.route("/caption", methods=["POST"])
def caption_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400
    
    img = Image.open(file)
    caption = generate_final_caption(img)
    session['image_context'] = caption
    session['chat_history'] = ""  # Reset chat history
    return jsonify({"caption": caption})

@app.route("/chat", methods=["POST"])
def chatbot():
    data = request.get_json()
    query = data.get("query")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Load previous chat history ids from session (if any)
    chat_history_ids = session.get("chat_history_ids")
    if chat_history_ids:
        chat_history_ids = torch.tensor(chat_history_ids)

    # Get bot response
    response, updated_history_ids = get_bot_response(query, chat_history_ids)

    # Store updated chat history back in session (as list for JSON compatibility)
    session["chat_history_ids"] = updated_history_ids.tolist()

    return jsonify({"response": response})


if __name__ == "__main__":
    app.run()
