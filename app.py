import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

# Define class labels
class_labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Load the model
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    yolo_model = YOLO('best.pt')  # Ensure 'best.pt' is in the same directory
    model = yolo_model.model  # Extract the underlying PyTorch model
    model.eval()
    return model

model = load_model()

# Define preprocessing transforms
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to 512x512 as per your model
        transforms.ToTensor(),
        # Add normalization if your model requires it
        # transforms.Normalize(mean=[...], std=[...])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Streamlit UI
st.title("Brain MRI Tumor Classification")
st.write("Upload an MRI image to predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    # Preprocess and predict
    with st.spinner('Analyzing...'):
        try:
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                # Get the predicted class index
                _, predicted_class = torch.max(output, 1)
                prediction = class_labels[predicted_class.item()]
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
        else:
            # Display result
            st.success(f"The model predicts: **{prediction}**")
