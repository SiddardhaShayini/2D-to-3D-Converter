# 2D to 3D Converter

This application converts photos or text prompts into downloadable 3D models using open-source AI/ML libraries.

## Features

- Accept input from users in two forms: image upload (.jpg/.png) or text prompt
- Process image inputs with appropriate preprocessing
- Generate 3D models from both image and text inputs
- Provide downloadable .obj or .stl files as output
- Display a 3D visualization of the generated model
- Clean and intuitive user interface with Streamlit

## How It Works

### Image to 3D Conversion
1. The application processes the uploaded image to extract the main object
2. It creates a heightmap based on pixel intensity and distance transform
3. The heightmap is converted into a 3D mesh
4. The mesh is cleaned and prepared for export

### Text to 3D Conversion
1. The application uses open-source models (attempting to use Shap-E or falling back to simple generation)
2. The text prompt is processed to generate a 3D representation
3. The resulting model is converted to standard 3D formats

## Libraries Used

- **Streamlit**: Web application interface
- **OpenCV & PIL**: Image preprocessing
- **Trimesh**: 3D model handling and export
- **Matplotlib**: 3D visualization
- **NumPy**: Numerical operations
- **DiffusionPipeline**: Text-to-3D generation (when available)

## Steps to Run

1. Ensure you have Python 3.7+ installed
2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the dependencies (requirements.txt is handled separately)
4. Run the application:
   ```
   streamlit run app.py
   