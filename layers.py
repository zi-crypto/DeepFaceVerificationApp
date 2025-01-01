# Custom L1 Distance layer Module
# Needed to load our custom model

# Importing the necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Layer

# Custom L1 Distance layer class
class L1Dist(Layer):

    # Init method - Inheritance
    def __init__(self, **kwargs):
        super().__init__()
    
    # This is were the majic happens
    # Similarity Calculation
    def call(self, input_embedding, validation_embedding):
        # Ensure inputs are tensors (They have to be tenosrs not lists)
        input_embedding = tf.convert_to_tensor(input_embedding)       
        validation_embedding = tf.convert_to_tensor(validation_embedding)

        return tf.math.abs(input_embedding - validation_embedding)