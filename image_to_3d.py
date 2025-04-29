import numpy as np
import cv2
from PIL import Image
import trimesh
from scipy import ndimage

def process_image_to_3d(input_image):
    """
    Process an image to create a 3D model.
    
    Args:
        input_image (PIL.Image): The input image to process
        
    Returns:
        trimesh.Trimesh: A 3D mesh representation of the input image
    """
    # Convert PIL image to numpy array for OpenCV
    img = np.array(input_image)
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive thresholding to extract the object
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to identify the object
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask with the largest contour (assumed to be the main object)
    mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Apply the mask to the grayscale image
    masked_img = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Generate a simple height map based on pixel intensity
    # Normalize values to [0,1] range
    height_map = masked_img.astype(float) / 255.0
    
    # Apply a distance transform to create a more natural height map
    # This will make the center of the object higher than the edges
    if np.max(mask) > 0:  # Only if we have a valid mask
        distance = ndimage.distance_transform_edt(mask)
        # Normalize the distance transform
        distance = distance / np.max(distance) if np.max(distance) > 0 else distance
        
        # Combine the intensity and distance for a better height map
        height_map = height_map * 0.7 + distance * 0.3
    
    # Create a mesh from the height map
    # First, create a grid of x, y coordinates
    h, w = height_map.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Create vertices with x, y, z coordinates
    vertices = np.vstack([x.flatten(), y.flatten(), height_map.flatten() * 20]).T
    
    # Create faces for the mesh
    faces = []
    for i in range(h-1):
        for j in range(w-1):
            # Calculate indices for the four vertices of the grid cell
            idx = i * w + j
            # Create two triangular faces for each grid cell
            faces.append([idx, idx + 1, idx + w])
            faces.append([idx + 1, idx + w + 1, idx + w])
    
    faces = np.array(faces)
    
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Clean up the mesh - remove duplicate vertices and unreferenced vertices
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    
    # Flip the mesh to have the correct orientation
    mesh.invert()
    
    return mesh
