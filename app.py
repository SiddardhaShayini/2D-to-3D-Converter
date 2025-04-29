import streamlit as st
import os
import numpy as np
import tempfile
from PIL import Image
import trimesh
import matplotlib.pyplot as plt
from matplotlib import colors
from io import BytesIO

from image_to_3d import process_image_to_3d
from text_to_3d import process_text_to_3d
from visualization import visualize_3d_model

# Set page config
st.set_page_config(
    page_title="2D to 3D Converter",
    page_icon="🧊",
    layout="wide"
)

# App title and description
st.title("2D to 3D Converter")
st.markdown("""
This application converts photos or text prompts into downloadable 3D models using open-source AI/ML libraries.
* Upload a photo (single object like a chair, car, or toy)
* Or enter a text prompt (like "A small toy car")
""")

# Create two tabs for the two input methods
tab1, tab2 = st.tabs(["Image Input", "Text Input"])

with tab1:
    st.header("Upload an Image")
    st.markdown("Upload a clear image of a single object. The application will try to convert it to a 3D model.")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Generate 3D Model from Image"):
            with st.spinner("Processing image and generating 3D model..."):
                try:
                    # Process the image and generate a 3D model
                    mesh = process_image_to_3d(image)
                    
                    # Create temporary files for the 3D model
                    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_obj:
                        mesh.export(tmp_obj.name)
                        obj_path = tmp_obj.name
                    
                    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_stl:
                        mesh.export(tmp_stl.name)
                        stl_path = tmp_stl.name
                    
                    # Display the 3D model
                    st.subheader("Generated 3D Model")
                    fig = visualize_3d_model(mesh)
                    st.pyplot(fig)
                    
                    # Provide download buttons
                    col1, col2 = st.columns(2)
                    
                    with open(obj_path, "rb") as file:
                        obj_bytes = file.read()
                    with open(stl_path, "rb") as file:
                        stl_bytes = file.read()
                    
                    with col1:
                        st.download_button(
                            label="Download as OBJ",
                            data=obj_bytes,
                            file_name="model.obj",
                            mime="application/octet-stream"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download as STL",
                            data=stl_bytes,
                            file_name="model.stl",
                            mime="application/octet-stream"
                        )
                    
                    # Clean up temporary files
                    os.unlink(obj_path)
                    os.unlink(stl_path)
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

with tab2:
    st.header("Enter a Text Prompt")
    st.markdown("Enter a description of the object you want to generate. Be specific and descriptive.")
    
    text_prompt = st.text_input("Text prompt", placeholder="Example: A small toy car")
    
    if text_prompt:
        # Process button
        if st.button("Generate 3D Model from Text"):
            with st.spinner("Processing text prompt and generating 3D model..."):
                try:
                    # Process the text and generate a 3D model
                    mesh = process_text_to_3d(text_prompt)
                    
                    # Create temporary files for the 3D model
                    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as tmp_obj:
                        mesh.export(tmp_obj.name)
                        obj_path = tmp_obj.name
                    
                    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_stl:
                        mesh.export(tmp_stl.name)
                        stl_path = tmp_stl.name
                    
                    # Display the 3D model
                    st.subheader("Generated 3D Model")
                    fig = visualize_3d_model(mesh)
                    st.pyplot(fig)
                    
                    # Provide download buttons
                    col1, col2 = st.columns(2)
                    
                    with open(obj_path, "rb") as file:
                        obj_bytes = file.read()
                    with open(stl_path, "rb") as file:
                        stl_bytes = file.read()
                    
                    with col1:
                        st.download_button(
                            label="Download as OBJ",
                            data=obj_bytes,
                            file_name="model.obj",
                            mime="application/octet-stream"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download as STL",
                            data=stl_bytes,
                            file_name="model.stl",
                            mime="application/octet-stream"
                        )
                    
                    # Clean up temporary files
                    os.unlink(obj_path)
                    os.unlink(stl_path)
                
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

# Add information about the application
st.markdown("---")
st.markdown("""
### How it works
- **Image to 3D**: The application processes the uploaded image to extract the main object, 
  estimates depth information, and creates a 3D mesh representation.
- **Text to 3D**: The application uses Point-E, an open-source text-to-3D model, to generate a 3D representation 
  based on your text description.

### Limitations
- Image to 3D works best with clear, single-object images with good contrast from the background.
- Text to 3D generation quality depends on the clarity and specificity of the description.
- Generated models are simplified representations and may require further refinement for complex applications.
""")
