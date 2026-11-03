import os
import zipfile
import shutil
import gc
import tempfile
import streamlit as st
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Text-to-Image Search", page_icon="🖼️", layout="centered")
st.title("🖼️ Mini Text-to-Image Search System")
st.write("Upload a `.zip` file of images, then search through them using natural language descriptions.")

# 1. Load the CLIP Model (Cached globally)
@st.cache_resource
def load_model():
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
    # Use a unique runtime session key so Streamlit tracks file state correctly
    session_key = f"data_{uploaded_zip.name}_{uploaded_zip.size}"
    
    # Initialize session state variables for tracking current zip index arrays
    if "current_zip" not in st.session_state or st.session_state["current_zip"] != session_key:
        st.session_state["current_zip"] = session_key
        st.session_state["valid_paths"] = []
        st.session_state["image_embeddings"] = None
        st.session_state["image_bytes"] = {} # Store original images in memory to safely display later

        # Create a unique random directory that won't collide with old sessions
        with tempfile.TemporaryDirectory() as target_temp_dir:
            try:
                with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                    zip_ref.extractall(target_temp_dir)
            except Exception as e:
                st.error(f"Failed to extract zip file: {e}")
                st.stop()

            # 3. Dynamic Image Indexing Loop
            with st.spinner("Analyzing and indexing images sequentially... 🧠"):
                supported_extensions = (".png", ".jpg", ".jpeg", ".webp")
                raw_paths = []
                
                for root, _, files in os.walk(target_temp_dir):
                    for file in files:
                        if file.lower().endswith(supported_extensions):
                            raw_paths.append(os.path.join(root, file))
                            
                if raw_paths:
                    all_embeddings = []
                    valid_names = []
                    
                    for path in raw_paths:
                        try:
                            with Image.open(path) as img:
                                img.thumbnail((400, 400))
                                img_rgb = img.convert("RGB")
                                
                                # Read raw image file content to keep in memory since temp folder auto-deletes
                                filename = os.path.basename(path)
                                
                                with torch.no_grad():
                                    embedding = model.encode(img_rgb, convert_to_tensor=True, show_progress_bar=False)
                                
                                all_embeddings.append(embedding)
                                valid_names.append(filename)
                                
                                # Re-open raw bytes to store inside Streamlit session memory safely
                                with open(path, "rb") as f:
                                    st.session_state["image_bytes"][filename] = f.read()
                        except Exception:
                            continue  # Skip unreadable images
                        finally:
                            gc.collect()

                    if all_embeddings:
                        st.session_state["image_embeddings"] = torch.stack(all_embeddings)
                        st.session_state["valid_paths"] = valid_names

    # Pull active data state out of session memory 
    valid_paths = st.session_state["valid_paths"]
    image_embeddings = st.session_state["image_embeddings"]
    image_bytes_dict = st.session_state["image_bytes"]

    # 4. Search UI logic 
    if not valid_paths or image_embeddings is None:
        st.warning("⚠️ No valid images found inside the uploaded zip file. Ensure it contains formats like .jpg or .png.")
    else:
        st.success(f"✅ Successfully indexed {len(valid_paths)} images safely!")
        st.divider()
        
        st.subheader("Step 2: Search your Library")
        query = st.text_input(
            "Enter a text description to find your image:", 
            placeholder="e.g., a sunny beach, close up photo of a cat...",
            key=f"input_{session_key}"
        )
        
        if query:
            with torch.no_grad():
                text_embedding = model.encode(query, convert_to_tensor=True)
                
            cos_scores = util.cos_sim(text_embedding, image_embeddings)
            
            if cos_scores.numel() > 0:
                best_match_idx = int(torch.argmax(cos_scores).item())
                best_score = cos_scores[best_match_idx].item()
                matched_filename = valid_paths[best_match_idx]
                
                st.markdown("### 🏆 Best Match Found")
                # Pull image data from secure memory dictionary to present to screen
                st.image(image_bytes_dict[matched_filename], use_container_width=True)
                st.caption(f"**Filename:** {matched_filename} | **Confidence Score:** {best_score:.4f}")
            else:
                st.error("Could not calculate scores for this search.")
else:
    st.info("💡 Please upload a zip file filled with pictures above to activate the system search features.")
