import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

def visualize_3d_model(mesh):
    """
    Create a matplotlib figure to visualize a 3D mesh.
    
    Args:
        mesh (trimesh.Trimesh): The mesh to visualize
        
    Returns:
        matplotlib.figure.Figure: The figure containing the visualization
    """
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get mesh vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Compute face centroids and normalize vertices for better visualization
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    
    # Scale the mesh to fit well in the plot
    scale_factor = 2.0 / np.max(max_vals - min_vals)
    center = (min_vals + max_vals) / 2
    vertices = (vertices - center) * scale_factor
    
    # Create a Poly3DCollection to display the mesh
    mesh_collection = Poly3DCollection([vertices[face] for face in faces], 
                                      alpha=0.7, linewidths=0.1, edgecolors='k')
    
    # Set the face colors to a shade of blue with some randomness for better visualization
    face_colors = np.ones((len(faces), 4)) * np.array([0.0, 0.5, 1.0, 1.0])
    # Add a small random variation to the color to help visualize the different faces
    face_colors[:, :3] += np.random.uniform(-0.1, 0.1, size=(len(faces), 3))
    # Ensure colors stay in valid range
    face_colors = np.clip(face_colors, 0, 1)
    
    # Set the face colors
    mesh_collection.set_facecolor(face_colors)
    
    # Add the mesh to the plot
    ax.add_collection3d(mesh_collection)
    
    # Set plot limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Model Visualization')
    
    # Improve the viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add a grid for better depth perception
    ax.grid(True)
    
    return fig
