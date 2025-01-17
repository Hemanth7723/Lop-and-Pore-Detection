import streamlit as st # for creating webapp
import cv2 # for computer vision tasks
from PIL import Image, ImageEnhance 
# PIL - Python Image Library, imported Image for image editing 
# imported imageEnhance for morphological processing operations
# like contrast,brightness,blurriness etc
import  numpy as np # to deal with arrays
# image contains pixel values 0-255 stores in arrays
import os # for os related operations
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import tensorflow as tf

def main():
    
    st.title('LOP or Porosity Detection')
    # st.markdown('For us, Safety is not a priority, it is a precondition')
    
    tasks = ['Detection','Visualization']
    choice = st.sidebar.selectbox('Select Task',tasks)
    
    if choice == 'Visualization':
        st.subheader('Image Visualization')
        image_file = st.file_uploader('Upload', type=['jpg', 'png', 'jpeg'])
        st.text("")
        
        if image_file is not None:
            our_image = Image.open(image_file)
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Display original image in the first column
            col1.markdown('Original Image')
            col1.image(our_image, width=325)
            
            
            # Display modified image in the second column
            col2.markdown('Modified Image')
            
            enhance_type = st.sidebar.radio("Enhance type", ['Original', 'Gray-scale', 'Contrast', 'Brightness', 'Blurring', 'Sharpness'])
            
            if enhance_type == 'Gray-scale':
                img = np.array(our_image.convert('RGB'))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                col2.image(gray, width=325)
            elif enhance_type == 'Contrast':
                rate = st.sidebar.slider("Contrast", 0.5, 6.0)
                enhancer = ImageEnhance.Contrast(our_image)
                enhanced_img = enhancer.enhance(rate)
                col2.image(enhanced_img, width=325)
            elif enhance_type == 'Brightness':
                rate = st.sidebar.slider("Brightness", 0.0, 8.0)
                enhancer = ImageEnhance.Brightness(our_image)
                enhanced_img = enhancer.enhance(rate)
                col2.image(enhanced_img, width=325)
            elif enhance_type == 'Blurring':
                rate = st.sidebar.slider("Blurring", 0.0, 7.0)
                blurred_img = cv2.GaussianBlur(np.array(our_image), (15, 15), rate)
                col2.image(blurred_img, width=325)
            elif enhance_type == 'Sharpness':
                rate = st.sidebar.slider("Sharpness", 0.0, 14.0)
                enhancer = ImageEnhance.Sharpness(our_image)
                enhanced_img = enhancer.enhance(rate)
                col2.image(enhanced_img, width=325)
            elif enhance_type == "Original":
                col2.image(our_image, width=325)
            else:
                col2.image(our_image, width=325)
            
            # Add space between lines
            st.text("")  # or st.markdown("") for Markdown
    
    elif choice == 'Detection':
        tasks=["LOP or Porosity"]
        disease_type = st.sidebar.radio("Enhance type",tasks)
        if disease_type == 'LOP or Porosity':
            # Load the saved model
            loaded_model = tf.keras.models.load_model("model.h5")

            def predict_image_class(img_path):
                img = image.load_img(img_path, target_size=(640, 640))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                predictions = loaded_model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]

                return predicted_class

            # Streamlit App
            st.subheader('LOP or Porosity detection')

            uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
                st.write("")
                st.write("Classifying...")

                # Get predictions
                predicted_class = predict_image_class(uploaded_file)
                
                # Display the predicted class
                num_classes = loaded_model.layers[-1].output_shape[1]
                classes = [str(i) for i in range(num_classes)]
                
                class_names = ["lop", "porosity"]

                st.write(f"Predicted State : {class_names[predicted_class]}")
            


        
  

if __name__ == '__main__':
    main()
