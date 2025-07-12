# # # prompt: write a streamlite code for this model so i can upload a test image and check for the prediction



# # import streamlit as st
# # import tensorflow as tf
# # from PIL import Image
# # import numpy as np

# # # Assuming your model is saved in the 'models/v1' directory or a similar structure
# # # Load your trained model
# # @st.cache_resource
# # def load_model():
# #     # Replace 'models/v1' with the correct path to your saved model
# #     # If you saved it using model.export(), the directory will be something like 'models/v1/saved_model.pb'
# #     # If you saved it using model.save(), it might be a directory structure
# #     model_path = '/Users/chuks/code/learnPy/potato_project/saved_model/v1/saved_model.pb'  # Update this path to your saved model
# #     model = tf.saved_model.load(model_path)
# #     return model

# # model = load_model()

# # # Assuming you have your class names stored or can derive them
# # # Replace this with your actual class names from your training dataset
# # class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']  # Example class names

# # st.title("Image Prediction App")

# # uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # if uploaded_file is not None:
# #     # Display the uploaded image
# #     image = Image.open(uploaded_file).convert('RGB')
# #     st.image(image, caption="Uploaded Image.", use_column_width=True)

# #     # Preprocess the image for the model
# #     img_array = tf.keras.preprocessing.image.img_to_array(image)
# #     img_array = tf.image.resize(img_array, (256, 256)) # Resize to your model's input size
# #     img_array = tf.expand_dims(img_array, 0) # Create a batch

# #     # Make a prediction
# #     # The exact way to call prediction depends on how you loaded the model.
# #     # If using tf.saved_model.load, you might need to access the concrete function.
# #     # Assuming your loaded model has a 'signatures' attribute with a 'serving_default' function
# #     if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
# #         infer = model.signatures['serving_default']
# #         predictions = infer(tf.constant(img_array))
# #         # Assuming the output key in the serving_default signature is 'predictions'
# #         # You might need to inspect your saved model to find the correct output key
# #         output_key = list(predictions.keys())[0] # Get the first output key if unknown
# #         predictions_array = predictions[output_key].numpy()
# #     elif isinstance(model, tf.keras.Model):
# #         # If you loaded a Keras model directly
# #         predictions_array = model.predict(img_array)
# #     else:
# #         st.error("Could not predict. Model type not recognized.")
# #         predictions_array = None

# #     if predictions_array is not None:
# #         predicted_class_index = np.argmax(predictions_array[0])
# #         predicted_class = class_names[predicted_class_index]
# #         confidence = np.max(predictions_array[0]) * 100

# #         st.write(f"Prediction: **{predicted_class}**")
# #         st.write(f"Confidence: **{confidence:.2f}%**")

# # # To run this Streamlit app in Colab, you'll need to use ngrok or a similar tool
# # # to expose the local port where Streamlit runs to the internet.
# # # You would typically run this script from your Colab notebook's terminal like:
# # # !streamlit run your_streamlit_app_file.py
# # # Then use ngrok to expose the port (usually 8501).
# # # !pip install pyngrok
# # # from pyngrok import ngrok
# # # public_url = ngrok.connect(port='8501')
# # # public_url

# import streamlit as st
# import tensorflow as tf
# from PIL import Image
# import numpy as np

# CLASS_NAMES = ['Potato___Early_blight',
#                'Potato___Late_blight',
#                'Potato___healthy']

# @st.cache_resource
# def load_model():
#     model_dir = "/Users/chuks/code/learnPy/potato_project/saved_model/v1"
#     return tf.saved_model.load(model_dir)

# model = load_model()

# st.title("Potato‑leaf disease classifier")

# uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     img = Image.open(uploaded_file).convert("RGB")
#     st.image(img, caption="Input image", use_column_width=True)

#     # --- preprocessing ------------------------------------------------------
#     x = tf.keras.preprocessing.image.img_to_array(img)
#     x = tf.image.resize(x, (256, 256)) / 255.0          # normalise if needed
#     x = tf.expand_dims(x, 0)                            # batch axis

#     # --- inference ----------------------------------------------------------
#     infer = model.signatures["serving_default"]
#     preds = infer(tf.constant(x))
#     output_key = next(iter(preds))
#     probs = preds[output_key].numpy()[0]

#     idx = int(np.argmax(probs))
#     st.markdown(f"### Prediction : **{CLASS_NAMES[idx]}**")
#     st.markdown(f"Confidence : **{probs[idx]*100:.2f}%**")





import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@st.cache_resource
def load_model():
    model_dir = "/Users/chuks/code/learnPy/potato_project/saved_model/v1"  # path to exported model
    return tf.saved_model.load(model_dir)

model = load_model()

st.title("Potato Leaf Disease Classifier")

uploaded_file = st.file_uploader("Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image",width=300, use_container_width=True)

    # Preprocessing (no normalization because model expects raw pixels)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.image.resize(img_array, (256, 256))
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    infer = model.signatures["serving_default"]
    predictions = infer(tf.constant(img_array))

    st.write(predictions)  # Debug: print full prediction output

    # Get output tensor
    output_key = list(predictions.keys())[0]
    probs = predictions[output_key].numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100

    st.markdown(f"### Prediction: **{pred_class}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")
