from flask import Flask, request, jsonify, session, render_template
from transformers import pipeline
import onnxruntime as ort
from PIL import Image
import numpy as np
import os
import torch
from groq import Groq  

app = Flask(__name__)   
app.secret_key = "AIzaSyC1oNGdNdodbWuY6eKxVrzOf7ai-UM3vfw"  

git_pipe = pipeline("image-to-text", model="microsoft/git-large-textcaps")

MODEL_DIR = "models"
model_files = {
    "flower": os.path.join(MODEL_DIR, "flower.onnx"),
    "bird": os.path.join(MODEL_DIR, "bird.onnx"),
    "dog": os.path.join(MODEL_DIR, "dog.onnx"),
    "landmark": os.path.join(MODEL_DIR, "landmark.onnx")
}

for name, path in model_files.items():
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")

flower_session = ort.InferenceSession(model_files["flower"])
bird_session = ort.InferenceSession(model_files["bird"])
dog_session = ort.InferenceSession(model_files["dog"])
landmark_session = ort.InferenceSession(model_files["landmark"])

dog_list = ["Bulldog", "Chihuahua", "Dobermann", "German Shepherd", "Golden Retriever", 
             "Husky", "Labrador Retriever", "Pomeranian", "Pug", "Rottweiler", "Street dog"]

flower_list = ["Jasmine", "Lavender", "Lily", "Lotus", "Orchid", "Rose", "Sunflower", 
                "Tulip", "Daisy", "Dandelion"]

bird_list = ["Crow", "Eagle", "Flamingo", "Hummingbird", "Parrot", "Peacock", 
              "Pigeon", "Sparrow", "Swan"]

landmark_list = ["The Agra Fort", "Ajanta Caves", "Alai Darwaza", "Amarnath Temple", 
                  "The Amber Fort", "Basilica of Bom Jesus", "Brihadisvara Temple", 
                  "Charminar", "Chhatrapati Shivaji Terminus", "Dal Lake", "The Elephanta Caves", 
                  "Ellora Caves", "Fatehpur Sikri", "Gateway of India", "Golden Temple", 
                  "Hawa Mahal", "Humayun's Tomb", "India Gate", "Jagannath Temple", "Jama Masjid", 
                  "Jantar Mantar", "Kedarnath Temple", "Konark Sun Temple", "Meenakshi Temple", 
                  "Nalanda Mahavihara", "Qutb Minar", "The Red Fort", "Taj Mahal", "Victoria Memorial"]

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def predict(session, img_array, label_list):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_array})
    return label_list[np.argmax(outputs[0])]

def identify_dog(img): return predict(dog_session, preprocess_image(img), dog_list)
def identify_flower(img): return predict(flower_session, preprocess_image(img), flower_list)
def identify_bird(img): return predict(bird_session, preprocess_image(img), bird_list)
def identify_landmark(img): return predict(landmark_session, preprocess_image(img), landmark_list)

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

GROQ_API_KEY = "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" 
groq_client = Groq(api_key=GROQ_API_KEY)

def get_groq_response(query, context=""):
    """
    Sends query + optional image context to Groq LLM and returns response.
    """
    messages = []
    if context:
        messages.append({"role": "system", "content": f"Image context: {context}"})
    messages.append({"role": "user", "content": query})

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Groq API Error:", e)
        return "Sorry, I faced an issue while generating a response."


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/caption", methods=["POST"])
def caption_image():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400
    
    img = Image.open(file)
    caption = generate_final_caption(img)
    session['image_context'] = caption
    return jsonify({"caption": caption})

@app.route("/chat", methods=["POST"])
def chatbot():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    image_context = session.get("image_context", "")

    try:
        response = get_groq_response(query, context=image_context)
        return jsonify({"response": response})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Sorry, I encountered an error. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True)
