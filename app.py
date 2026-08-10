import os
import streamlit as st
import pandas as pd
from capture_and_analyze import capture_image, analyze_leaf
from alerts import save_result

st.set_page_config(page_title="AgroTrack Demo", layout="wide")
st.title("AgroTrack Leaf Monitoring Demo")

st.write(
    "Capture a tomato leaf image using laptop camera OR upload an image, "
    "analyze it with DenseNet121, generate an alert, and store the result."
)

CAPTURE_FOLDER = "captured"
os.makedirs(CAPTURE_FOLDER, exist_ok=True)


def show_result(image_path, result):
    st.success("Image analyzed successfully.")

    st.image(image_path, caption="Leaf Image", use_container_width=True)

    st.subheader("Analysis Result")
    st.write(f"**Leaf Status:** {result.get('status', 'N/A')}")
    st.write(f"**Predicted Class:** {result.get('predicted_class', 'N/A')}")
    st.write(f"**Confidence:** {result.get('confidence', 'N/A')}")
    st.write(f"**Risk Level:** {result.get('risk', 'N/A')}")
    st.write(f"**Alert:** {result.get('alert', 'N/A')}")
    st.write(f"**Model Used:** {result.get('model_name', 'N/A')}")

    if result.get("risk") == "Low":
        st.success("No Alert Required")
    elif result.get("risk") == "Medium":
        st.warning("Warning Alert Generated")
    else:
        st.error("Immediate Alert Generated")


st.subheader("Choose Input Method")

option = st.radio(
    "Select image source:",
    ["Capture from Camera", "Upload Image"]
)

if option == "Capture from Camera":
    if st.button("Capture Leaf Image"):
        try:
            image_path = capture_image()

            if image_path:
                result = analyze_leaf(image_path)
                save_result(image_path, result)
                show_result(image_path, result)
            else:
                st.info("No image captured.")

        except Exception as e:
            st.error(f"Error: {e}")

elif option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            upload_path = os.path.join(CAPTURE_FOLDER, uploaded_file.name)

            with open(upload_path, "wb") as f:
                f.write(uploaded_file.read())

            st.image(upload_path, caption="Uploaded Leaf Image", use_container_width=True)

            if st.button("Analyze Uploaded Image"):
                result = analyze_leaf(upload_path)
                save_result(upload_path, result)
                show_result(upload_path, result)

        except Exception as e:
            st.error(f"Error: {e}")

st.subheader("Dashboard Log")

if os.path.exists("logs.csv") and os.path.getsize("logs.csv") > 0:
    try:
        df = pd.read_csv("logs.csv")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not read logs.csv: {e}")
else:
    st.info("No records yet.")