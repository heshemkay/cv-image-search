import os
import zipfile
import shutil
import gc
import streamlit as st
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# Configuration for temporary extraction
TEMP_DIR = "extracted_images"

st.set_page_config(page_title="Text-to-Image Search", page_icon="🖼️", layout="centered")
st.title("🖼️ Mini Text-to-Image Search System")
st.write("Upload a `.zip` file of images, then search through them using natural language descriptions.")

# 1. Load the CLIP Model (Using the absolute lightest configuration)
@st.cache_resource
def load_model():
    # Explicitly telling torch to use CPU and single-thread to save RAM
    torch.set_num_threads(1)
    return SentenceTransformer('clip-ViT-B-32')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. File Upload Handling
uploaded_zip = st.file_uploader("Step 1: Upload a Zipped Folder of Images (.zip)", type=["zip"])

if uploaded_zip is not None:
    # Clear out older sessions to free up space
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    try:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(TEMP_DIR)
    except Exception as e:
        st.error(f"Failed to extract zip file: {e}")
        st.stop()

    # 3. Memory-Safe Sequential Processing Loop
    @st.cache_data(show_spinner="Analyzing and indexing images sequentially... 🧠")
    def index_extracted_images(directory):
        supported_extensions = (".png", ".jpg", ".jpeg", ".webp")
        image_paths = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    image_paths.append(os.path.join(root, file))
                    
        if not image_paths:
            return [], None
            
        valid_paths = []
        all_embeddings = []
        
        # FIX: Process one image at a time to prevent RAM spikes
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    # Instantly downsample image size to save heavy RAM space before encoding
                    img.thumbnail((400, 400))
                    img_rgb = img.convert("RGB")
                    
                    with torch.no_grad():
                        embedding = model.encode(img_rgb, convert_to_tensor=True, show_progress_bar=False)
                    
                    all_embeddings.append(embedding)
                    valid_paths.append(path)
            except Exception:
                continue  # Skip unreadable or broken images
            finally:
                # Force Python to clear unused memory blocks out of RAM immediately
                gc.collect()
                
        if not all_embeddings:
            return [], None

        # Stack separate individual vectors back into a clean matrix
        image_embeddings = torch.stack(all_embeddings)
        return valid_paths, image_embeddings

    image_paths, image_embeddings = index_extracted_images(TEMP_DIR)

    # 4. Search UI logic with Safety Checks
    if not image_paths or image_embeddings is None or len(image_embeddings) == 0:
        st.warning("⚠️ No valid images found. Ensure your zip contains standard unencrypted formats like .jpg or .png.")
    else:
        st.success(f"✅ Successfully indexed {len(image_paths)} images safely without crashing!")
        st.divider()
        
        st.subheader("Step 2: Search your Library")
        query = st.text_input("Enter a text description to find your image:", placeholder="e.g., a sunny beach, close up photo of a cat...")
        
        if query:
            with torch.no_grad():
                text_embedding = model.encode(query, convert_to_tensor=True)
                
            cos_scores = util.cos_sim(text_embedding, image_embeddings)
            
            if cos_scores.numel() > 0:
                best_match_idx = int(torch.argmax(cos_scores).item())
                best_score = cos_scores[best_match_idx].item()
                matched_image_path = image_paths[best_match_idx]
                
                st.markdown("### 🏆 Best Match Found")
                st.image(matched_image_path, use_container_width=True)
                st.caption(f"**Filename:** {os.path.basename(matched_image_path)} | **Confidence Score:** {best_score:.4f}")
            else:
                st.error("Could not calculate scores for this search.")
else:
    st.info("💡 Please upload a zip file filled with pictures above to activate the system search features.")
