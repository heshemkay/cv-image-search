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
    # Build a highly specific unique state tracking token based on file metadata
    session_key = f"data_{uploaded_zip.name}_{uploaded_zip.size}"
    
    # Initialize state variables if a completely new file is detected
    if "current_zip" not in st.session_state or st.session_state["current_zip"] != session_key:
        st.session_state["current_zip"] = session_key
        st.session_state["valid_paths"] = []
        st.session_state["image_embeddings"] = None
        st.session_state["image_bytes"] = {} 

        # Generate isolated temporary path execution blocks
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
                                filename = os.path.basename(path)
                                
                                with torch.no_grad():
                                    embedding = model.encode(img_rgb, convert_to_tensor=True, show_progress_bar=False)
                                
                                all_embeddings.append(embedding)
                                valid_names.append(filename)
                                
                                with open(path, "rb") as f:
                                    st.session_state["image_bytes"][filename] = f.read()
                        except Exception:
                            continue  
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
        # Enforcing a key tied directly to the dataset volume forces the input field to wipe clean on file switches
        query = st.text_input(
            "Enter a text description to find your image:", 
            placeholder="e.g., a sunny beach, close up photo of a cat...",
            key=f"query_input_{session_key}_{len(valid_paths)}"
        )
        
        if query:
            with torch.no_grad():
                text_embedding = model.encode(query, convert_to_tensor=True)
                
            cos_scores = util.cos_sim(text_embedding, image_embeddings)
            
            # FIX: Ensure computed score matrix matches the exact boundary limits of the active dataset array
            if cos_scores.numel() > 0 and cos_scores.shape[1] == len(valid_paths):
                best_match_idx = int(torch.argmax(cos_scores).item())
                
                # Double-checking index boundary strictly before executing array lookups
                if 0 <= best_match_idx < len(valid_paths):
                    best_score = cos_scores[0, best_match_idx].item()
                    matched_filename = valid_paths[best_match_idx]
                    
                    st.markdown("### 🏆 Best Match Found")
                    st.image(image_bytes_dict[matched_filename], use_container_width=True)
                    st.caption(f"**Filename:** {matched_filename} | **Confidence Score:** {best_score:.4f}")
                else:
                    st.error("Index error: Computed vector mapping out of safe bounds.")
            else:
                # Silently catch mismatched sizes during quick transition states
                st.info("Refreshing search index for your new data...")
else:
    st.info("💡 Please upload a zip file filled with pictures above to activate the system search features.")
