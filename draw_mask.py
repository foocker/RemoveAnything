import gradio as gr
import numpy as np
from PIL import Image
import os

def save_drawn_mask(editor_output):
    """
    Processes the input from Gradio's ImageEditor component,
    extracts the drawn mask, saves it, and returns it for display.

    Args:
        editor_output (dict): A dictionary from gr.ImageEditor, typically containing:
            'background': np.array of the original uploaded image (or None).
            'layers': list of np.array, where each is a drawn layer (e.g., RGBA).
            'composite': np.array of the final image with layers applied.

    Returns:
        PIL.Image or None: The saved mask as a PIL Image, or None if no mask was drawn/extracted.
    """
    print("--- save_drawn_mask called ---")
    if editor_output is None or not isinstance(editor_output, dict):
        print(f"DEBUG: Invalid input. editor_output is None or not a dict. Type: {type(editor_output)}")
        return None
    
    print(f"DEBUG: editor_output keys: {list(editor_output.keys())}")
    if 'background' in editor_output and editor_output['background'] is not None:
        print(f"DEBUG: editor_output['background'] shape: {editor_output['background'].shape}, dtype: {editor_output['background'].dtype}")
    else:
        print("DEBUG: editor_output['background'] is None or not present.")

    if 'layers' in editor_output and editor_output['layers'] is not None:
        print(f"DEBUG: editor_output['layers'] type: {type(editor_output['layers'])}, len: {len(editor_output['layers']) if isinstance(editor_output['layers'], list) else 'N/A'}")
    else:
        print("DEBUG: editor_output['layers'] is None or not present.")

    if 'composite' in editor_output and editor_output['composite'] is not None:
        print(f"DEBUG: editor_output['composite'] shape: {editor_output['composite'].shape}, dtype: {editor_output['composite'].dtype}")
    else:
        print("DEBUG: editor_output['composite'] is None or not present.")

    layers = editor_output.get('layers')

    if not layers or not isinstance(layers, list) or len(layers) == 0:
        print("DEBUG: No mask drawn (no layers found in editor output or layers is not a non-empty list).")
        return None

    print(f"DEBUG: Found {len(layers)} layer(s).")
    first_layer_rgba = layers[0]

    if not isinstance(first_layer_rgba, np.ndarray):
        print(f"DEBUG: First layer is not a numpy array. Type: {type(first_layer_rgba)}")
        return None
    
    print(f"DEBUG: First layer shape: {first_layer_rgba.shape}, dtype: {first_layer_rgba.dtype}")

    mask_np_raw = None # Before type conversion or checks
    if first_layer_rgba.ndim == 3 and first_layer_rgba.shape[2] == 4: # RGBA
        print("DEBUG: Processing first layer as RGBA.")
        mask_np_raw = first_layer_rgba[:, :, 3] # Extract alpha channel
        print(f"DEBUG: Extracted alpha channel. Shape: {mask_np_raw.shape}, dtype: {mask_np_raw.dtype}, Min: {mask_np_raw.min()}, Max: {mask_np_raw.max()}")
        
        # Ensure mask is uint8, 0-255 for PIL and consistent checks
        if mask_np_raw.max() <= 1.0 and (mask_np_raw.dtype == np.float32 or mask_np_raw.dtype == np.float64):
            print("DEBUG: Alpha channel is float, normalizing to 0-255 uint8.")
            mask_np = (mask_np_raw * 255).astype(np.uint8)
        else:
            print("DEBUG: Alpha channel is not float 0-1 or already uint8. Casting to uint8.")
            mask_np = mask_np_raw.astype(np.uint8)
        print(f"DEBUG: Converted mask_np. Shape: {mask_np.shape}, dtype: {mask_np.dtype}, Min: {mask_np.min()}, Max: {mask_np.max()}")

        if np.all(mask_np == 0):
            print("DEBUG: Mask is empty (all transparent in alpha channel after conversion).")
            return None
            
    elif first_layer_rgba.ndim == 2: # Grayscale layer
        print("DEBUG: Processing first layer as Grayscale.")
        mask_np_raw = first_layer_rgba
        print(f"DEBUG: Grayscale layer. Shape: {mask_np_raw.shape}, dtype: {mask_np_raw.dtype}, Min: {mask_np_raw.min()}, Max: {mask_np_raw.max()}")
        mask_np = mask_np_raw.astype(np.uint8)
        print(f"DEBUG: Converted mask_np. Shape: {mask_np.shape}, dtype: {mask_np.dtype}, Min: {mask_np.min()}, Max: {mask_np.max()}")

        if np.all(mask_np == 0):
             print("DEBUG: Mask is empty (all black grayscale layer after conversion).")
             return None
    else:
        print(f"DEBUG: Unexpected layer format: shape {first_layer_rgba.shape}, dtype {first_layer_rgba.dtype}. Cannot extract mask.")
        return None
    
    if mask_np is None:
        # This case should ideally be caught by earlier checks if logic is sound
        print("DEBUG: Failed to extract mask_np (mask_np is None unexpectedly).")
        return None

    print("DEBUG: Mask extracted successfully. Proceeding to save.")
    # Convert numpy array mask to PIL Image
    try:
        mask_pil = Image.fromarray(mask_np)
        print("DEBUG: Converted mask_np to PIL Image successfully.")
    except Exception as e:
        print(f"DEBUG: Error converting mask numpy array to PIL Image: {e}")
        return None
    
    mask_save_path = "drawn_mask.png"
    try:
        mask_pil.save(mask_save_path)
        print(f"DEBUG: Mask saved to {mask_save_path} at {os.path.abspath(mask_save_path)}")
        return mask_pil # Return the PIL image for display
    except Exception as e:
        print(f"DEBUG: Error saving mask: {e}")
        return None

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Draw Mask\nUpload an image, then use the sketch tool to create a mask. The display area will resize to fit the image.")

    with gr.Row():
        mask_editor = gr.ImageEditor(
            label="Draw Mask Here",
            type="numpy",
            interactive=True,
        )

        output_mask = gr.Image(
            label="Generated Mask Preview",
            type="numpy",
            interactive=False,
        )

    btn = gr.Button("Process and Save Mask")

    btn.click(
        fn=save_drawn_mask,
        inputs=[mask_editor],
        outputs=[output_mask]
    )

if __name__ == "__main__":
    print("Attempting to launch Gradio app...")
    demo.launch(share=False)
    print("Gradio app launch call completed. If you see this immediately, launch() did not block.")