# 🖼️🧠 Image Recognition Chatbot using Flask + ONNX + Ollama

This project combines image captioning, object classification (dog, bird, flower, landmark), and a conversational AI chatbot powered by Ollama (local LLM like Mistral/Gemma/LLaMA).

🔥 Features

🧠 AI chatbot powered by Ollama (local LLM like Mistral/Gemma)

🖼️ Image captioning using microsoft/git-large-textcaps

🐶 Image classification using ONNX models for:

Dogs 🐕
Birds 🐦
Flowers 🌸
Landmarks 🏰

🧾 Session-aware conversation with context from uploaded image
🔧 Built with Flask, Transformers, ONNX Runtime, Torch, and Ollama

📁 Folder Structure
.
├── app.py / run.py               # Flask app
├── models/                       # ONNX model files (flower.onnx, dog.onnx, etc.)
├── templates/
│   └── index.html                # Frontend HTML
├── static/                       # Optional CSS/JS
├── venv/                         # Virtual environment
└── README.md

🚀 Setup Instructions

1. 🔃 Clone the Repo

git clone https://github.com/prathamj937/image-recognition-chatbot.git
cd image-recognition-chatbot

2. 🐍 Create Virtual Environment

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

3. 📦 Install Dependencies

pip install -r requirements.txt

If requirements.txt is not created, manually install:

pip install flask transformers torch onnxruntime pillow numpy ollama

4. 📥 Download or Place ONNX Models

Put the following models in the models/ folder:

flower.onnx
dog.onnx
bird.onnx
landmark.onnx

(You can export these from your training script or request them from the developer.)

5. 🧠 Install & Run Ollama

Install Ollama and pull a lightweight model:

ollama pull mistral     # or gemma:2b for lower memory usage

Keep the Ollama server running in a separate terminal:

ollama serve

6. 🚦 Run the Flask App

python run.py

Visit http://localhost:5000 in your browser.

🧠 Chatbot Model Selection (Ollama)

In get_ollama_response() function (inside run.py), you can switch model:

response = ollama.chat(model="mistral", messages=messages)

Use:
mistral
gemma:2b
llama3 (only if enough RAM)

💬 Example Flow

Upload image of a dog 🐶
App captions: "A cute dog sitting in the grass."
Dog breed classification result: "Golden Retriever"
Ask chatbot: "Tell me more about this breed."
Bot answers using local LLM via Ollama 🧠

⚙️ Troubleshooting





Ollama model not found → Run ollama pull mistral



"requires more system memory" → Use lighter model like mistral or gemma:2b
