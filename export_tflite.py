import tensorflow as tf
import numpy as np
import os

def export_tflite():
    # Load test data for evaluation
    X_test = np.load('X_test.npy').astype(np.float32)
    y_test = np.load('y_test.npy').astype(np.int64)

    # Load the trained model
    model = tf.keras.models.load_model('ecg_classifier_model.h5')

    # 1. Evaluate on test set
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {acc:.4f}")

    # 2. Convert to standard TFLite (Float32)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model_float = converter.convert()
    
    with open('model_float.tflite', 'wb') as f:
        f.write(tflite_model_float)
    print("Standard TFLite model saved to model_float.tflite")

    # 3. Convert to Integer Quantized TFLite
    # Representative dataset generator
    def representative_data_gen():
        for i in range(100):
            # Use 100 samples from the test set
            sample = X_test[i:i+1]
            yield [sample]

    converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
    converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
    converter_quant.representative_dataset = representative_data_gen
    # Ensure all operations are integer-only for best performance on STM32
    converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_quant.inference_input_type = tf.float32 # Or tf.int8 if you want to pass raw ADC data
    converter_quant.inference_output_type = tf.float32
    
    tflite_model_quant = converter_quant.convert()
    
    with open('model_quant.tflite', 'wb') as f:
        f.write(tflite_model_quant)
    print("Quantized TFLite model saved to model_quant.tflite")

    # Compare sizes
    print(f"Float model size: {len(tflite_model_float) / 1024:.2f} KB")
    print(f"Quantized model size: {len(tflite_model_quant) / 1024:.2f} KB")

if __name__ == "__main__":
    export_tflite()
