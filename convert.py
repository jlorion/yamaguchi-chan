import os
import tensorflow as tf
import tf2onnx
import onnx

def convert_keras_to_onnx_with_compatibility(keras_model_path, onnx_model_path=None):
    """
    Convert a Keras model to ONNX format with version compatibility handling
    
    Args:
        keras_model_path (str): Path to the Keras model file
        onnx_model_path (str, optional): Output path for the ONNX model
    
    Returns:
        str: Path to the saved ONNX model
    """
    # Make sure the output path is set
    if onnx_model_path is None:
        base_name = os.path.splitext(keras_model_path)[0]
        onnx_model_path = f"{base_name}.onnx"
    
    print(f"Loading Keras model from {keras_model_path}...")
    
    # Try different approaches to load the model
    try:
        # Option 1: Direct load (might fail with version mismatch)
        model = tf.keras.models.load_model(keras_model_path)
    except (ImportError, ModuleNotFoundError, TypeError) as e:
        print(f"Standard loading failed: {str(e)}")
        print("Trying alternative loading method...")
        
        # Option 2: Load with custom objects set to "safe mode"
        try:
            model = tf.keras.models.load_model(
                keras_model_path,
                compile=False,  # Skip loading the optimizer state
                custom_objects={'Functional': tf.keras.Model}  # Try to map the missing class
            )
        except Exception as e2:
            print(f"Alternative loading also failed: {str(e2)}")
            print("Trying to save and reload the model...")
            
            # Option 3: Use tf.saved_model format as an intermediary
            try:
                # If you have access to the original model in memory (in Google Colab)
                # This assumes you have a reference to the model as 'original_model'
                # Change this to match how you have access to your model
                print("Please ensure you have the model in memory as 'model' variable")
                print("Converting to SavedModel format first...")
                
                # Save to a temporary SavedModel directory
                temp_saved_model_dir = os.path.join(os.path.dirname(keras_model_path), "temp_saved_model")
                tf.keras.models.save_model(model, temp_saved_model_dir, save_format='tf')
                
                # Load from SavedModel directory
                model = tf.keras.models.load_model(temp_saved_model_dir)
                
                print(f"Successfully loaded model through SavedModel format")
            except Exception as e3:
                print(f"All loading methods failed. Error: {str(e3)}")
                print("\nRECOMMENDATION:")
                print("1. If you're in Colab or have the model in memory, save it directly to SavedModel format:")
                print("   model.save('my_model', save_format='tf')")
                print("2. Then load it from that format:")
                print("   loaded_model = tf.keras.models.load_model('my_model')")
                print("3. Finally, convert the loaded model to ONNX")
                return None
    
    # Get model input specs
    input_signature = []
    for input_layer in model.inputs:
        input_shape = list(input_layer.shape)
        # Replace None with a dynamic dimension
        input_shape = [1 if dim is None else dim for dim in input_shape]
        input_signature.append(tf.TensorSpec(input_shape, input_layer.dtype))
    
    # Convert the model to ONNX
    print("Converting model to ONNX format...")
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
    
    # Save the ONNX model
    print(f"Saving ONNX model to {onnx_model_path}...")
    onnx.save_model(model_proto, onnx_model_path)
    
    print(f"Conversion complete! ONNX model saved to: {onnx_model_path}")
    return onnx_model_path

# Example usage
if __name__ == "__main__":
    # import argparse
    
    # parser = argparse.ArgumentParser(description='Convert Keras model to ONNX format')
    # parser.add_argument('--input', required=True, help='Path to input .keras or .h5 model file')
    # parser.add_argument('--output', help='Path to output .onnx model file (optional)')
    
    # args = parser.parse_args()
    
    convert_keras_to_onnx_with_compatibility('robin_efficientnetv2s.keras', 'better_mod.onnx')