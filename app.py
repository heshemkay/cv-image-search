import os
import zipfile
import shutil
import streamlit as st
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

# Configuration for temporary extraction
TEMP_DIR = "extracted_images"

st.set_page_config(page_title="Text-to-Image Search", page_icon="🖼️", layout="centered")
st.title("🖼️ Mini Text-to-Image Search System")
st.write("Upload a `.zip` file of images, then search through them using natural language descriptions.")

# 1. Load the CLIP Model (Cached)
@st.cache_resource
def load_model():
    # Utilizing OpenAI's CLIP model via sentence-transformers wrapper
    return SentenceTransformer('clip-ViT-B-32')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 2. File Upload Handling
uploaded_zip = st.file_uploader("Step 1: Upload a Zipped Folder of Images (.zip)", type=["zip"])

if uploaded_zip is not None:
    # Clear out older sessions to avoid mixing files
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Extract the zip file safely
    try:
        with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
            zip_ref.extractall(TEMP_DIR)
    except Exception as e:
        st.error(f"Failed to extract zip file: {e}")
        st.stop()

    # 3. Scan Extracted Files and Index Images
    @st.cache_data(show_spinner="Analyzing and indexing your images... 🧠")
    def index_extracted_images(directory):
        supported_extensions = (".png", ".jpg", ".jpeg", ".webp")
        image_paths = []
        
        # Walk through the directory (handles nested folders inside the zip)
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    image_paths.append(os.path.join(root, file))
                    
        if not image_paths:
            return [], None
            
        loaded_images = []
        valid_paths = []
        
        # Load and convert to RGB format required by CLIP
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                loaded_images.append(img)
                valid_paths.append(path)
            except Exception:
                continue # Skip unreadable or corrupted files
                
        if not loaded_images:
            return [], None

        # Generate vectors (embeddings)
        with torch.no_grad():
            image_embeddings = model.encode(loaded_images, convert_to_tensor=True, show_progress_bar=False)
            
        return valid_paths, image_embeddings

    image_paths, image_embeddings = index_extracted_images(TEMP_DIR)

    # 4. Search UI logic
    if not image_paths:
        st.warning("⚠️ No valid images (.jpg, .jpeg, .png, .webp) found inside the uploaded zip file.")
    else:
        st.success(f"✅ Successfully indexed {len(image_paths)} images from your zip file!")
        st.divider()
        
        # User query input
        st.subheader("Step 2: Search your Library")
        query = st.text_input("Enter a text description to find your image:", placeholder="e.g., a sunny beach, close up photo of a cat, red car...")
        
        if query:
            # Vectorize the text query
            with torch.no_grad():
                text_embedding = model.encode(query, convert_to_tensor=True)
                
            # Compute similarity metrics between the text concept and images
            cos_scores = util.cos_sim(text_embedding, image_embeddings)
            
            # Extract top matching record
            best_match_idx = int(torch.argmax(cos_scores).item())
            best_score = cos_scores[best_match_idx].item()
            matched_image_path = image_paths[best_match_idx]
            
            # Render match results to screen
            st.markdown("### 🏆 Best Match Found")
            st.image(matched_image_path, use_container_width=True)
            st.caption(f"**Filename:** {os.path.basename(matched_image_path)} | **Confidence Score:** {best_score:.4f}")
else:
    st.info("💡 Please upload a zip file filled with pictures above to activate the system search features.")
