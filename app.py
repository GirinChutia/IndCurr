import streamlit as st 
from PIL import Image
from inference import Inference

st.title("Indian Currency Recognition")

infer = Inference('best_weights\IC_ResNet34_9880.pth')

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    infer.run_image(uploaded_file,show=False)
    result = infer.return_result()
    
    st.write("Result :")
    st.write(result)