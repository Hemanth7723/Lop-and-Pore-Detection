import streamlit as st
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

def crop_image(image, left, top, right, bottom):
    return image.crop((left, top, right, bottom))

def add_salt_and_pepper(image_array, amount):
    row, col, _ = image_array.shape
    s_vs_p = 0.5
    out = np.copy(image_array)
    num_salt = np.ceil(amount * image_array.size * s_vs_p)
    num_pepper = np.ceil(amount * image_array.size * (1.0 - s_vs_p))

    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_array.shape]
    out[coords[0], coords[1], :] = 1

    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_array.shape]
    out[coords[0], coords[1], :] = 0
    return out

def add_gaussian_noise(image_array, mean=0, var=0.1):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image_array.shape)
    noisy_image = image_array + gauss
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def apply_filter(image, filter_type):
    return image.filter(filter_type)

def apply_canny(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def apply_sobel(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    return np.uint8(sobel)


def main():
    st.title('LOP or Porosity Detection')

    tasks = ['Detection', 'Visualization']
    choice = st.sidebar.selectbox('Select Task', tasks)

    if choice == 'Visualization':
        st.subheader('Image Visualization')
        image_file = st.file_uploader('Upload', type=['jpg', 'png', 'jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            col1, col2 = st.columns(2)
            col1.markdown('Original Image')
            col1.image(our_image, width=325)

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
            else:
                col2.image(our_image, width=325)

            process_choice = st.sidebar.selectbox("Choose a process", ["None", "Crop", "Flip Horizontal", "Flip Vertical", "Add Noise", "Apply Filter", "Edge Detection"])

            if process_choice == "Crop":
                left = st.sidebar.slider("Left", 0, our_image.width, 0)
                top = st.sidebar.slider("Top", 0, our_image.height, 0)
                right = st.sidebar.slider("Right", 0, our_image.width, our_image.width)
                bottom = st.sidebar.slider("Bottom", 0, our_image.height, our_image.height)
                cropped_image = crop_image(our_image, left, top, right, bottom)
                col2.image(cropped_image, width=325)
            elif process_choice == "Flip Horizontal":
                flipped_image = our_image.transpose(Image.FLIP_LEFT_RIGHT)
                col2.image(flipped_image, width=325)
            elif process_choice == "Flip Vertical":
                flipped_image = our_image.transpose(Image.FLIP_TOP_BOTTOM)
                col2.image(flipped_image, width=325)
            elif process_choice == "Add Noise":
                noise_type = st.sidebar.radio("Noise type", ["Salt and Pepper", "Gaussian"])
                if noise_type == "Salt and Pepper":
                    noise_amount = st.sidebar.slider("Amount", 0.01, 0.1, 0.05)
                    noisy_image = add_salt_and_pepper(np.array(our_image.convert('RGB')), noise_amount)
                    col2.image(noisy_image, width=325)
                elif noise_type == "Gaussian":
                    noisy_image = add_gaussian_noise(np.array(our_image.convert('RGB')))
                    col2.image(noisy_image, width=325)
            elif process_choice == "Apply Filter":
                filter_type = st.sidebar.radio("Filter type", ["Average", "Median", "Min", "Max", "Mode", "Gaussian"])
                if filter_type == "Average":
                    filtered_image = apply_filter(our_image, ImageFilter.BoxBlur(3))
                elif filter_type == "Median":
                    filtered_image = apply_filter(our_image, ImageFilter.MedianFilter(size=3))
                elif filter_type == "Min":
                    filtered_image = apply_filter(our_image, ImageFilter.MinFilter(size=3))
                elif filter_type == "Max":
                    filtered_image = apply_filter(our_image, ImageFilter.MaxFilter(size=3))
                elif filter_type == "Mode":
                    filtered_image = apply_filter(our_image, ImageFilter.ModeFilter(size=3))
                elif filter_type == "Gaussian":
                    filtered_image = apply_filter(our_image, ImageFilter.GaussianBlur(radius=3))
                col2.image(filtered_image, width=325)
            elif process_choice == "Edge Detection":
                edge_type = st.sidebar.radio("Edge detection type", ["Canny", "Sobel"])
                if edge_type == "Canny":
                    edge_image = apply_canny(our_image)
                elif edge_type == "Sobel":
                    edge_image = apply_sobel(our_image)
                col2.image(edge_image, width=325)

    elif choice == 'Detection':
        st.subheader('LOP or Porosity detection')
        loaded_model = load_model("model.h5")

        def predict_image_class(img_path):
            img = image.load_img(img_path, target_size=(640, 640))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            predictions = loaded_model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            return predicted_class

        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            st.write("")
            st.write("Classifying...")

            predicted_class = predict_image_class(uploaded_file)
            class_names = ["lop", "porosity"]
            st.write(f"Predicted State: {class_names[predicted_class]}")

if __name__ == '__main__':
    main()
