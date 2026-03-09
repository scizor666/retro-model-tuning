import os
import urllib.request

def download_embedding_model():
    # Recommended lightweight model for mobile/edge
    model_url = "https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite"
    model_path = "models/text_embedder.tflite"
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Downloading MediaPipe text embedder to {model_path}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete.")
    else:
        print(f"Model already exists at {model_path}")

if __name__ == "__main__":
    download_embedding_model()
