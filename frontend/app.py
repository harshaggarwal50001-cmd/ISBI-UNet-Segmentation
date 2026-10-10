import streamlit as st
import requests
import base64
from io import BytesIO

API_URL = "http://127.0.0.1:8000"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Predict", "History"])
if page == "Welcome":
    st.title("ISBI2012 Image Segmentation")
    st.write("Welcome to the ISBI2012 UNet Image Segmentation App!")
    st.write("This app uses a deep learning UNet model to predict segmentation masks.")
    st.write("")
    st.write("How to use:")
    st.write("1. Go to Predict page from the sidebar")
    st.write("2. Upload a grayscale image (.png, .jpg, .tif)")
    st.write("3. Click Predict to get the segmentation mask")
    st.write("4. Check History to see past predictions or delete unwanted entries")

elif page == "Predict":
    st.title("Predict Segmentation Mask")
    st.write("Upload a grayscale image and the model will predict its segmentation mask.")

    uploaded_file = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg", "tif", "tiff"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        if st.button("Predict"):
            with st.spinner("Predicting"):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                try:
                    response = requests.post(f"{API_URL}/predict", files=files)
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"Prediction #{result['id']} done!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("Original Image")
                            orig_bytes = base64.b64decode(result["original_image"])
                            st.image(orig_bytes, width=250)
                        with col2:
                            st.write("Predicted Mask")
                            pred_bytes = base64.b64decode(result["predicted_image"])
                            st.image(pred_bytes, width=250)
                    else:
                        st.error("Prediction failed!")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")

elif page == "History":
    st.title("Prediction History")

    try:
        response = requests.get(f"{API_URL}/history")
        if response.status_code == 200:
            history = response.json()
            if len(history) == 0:
                st.write("No predictions yet. Go to Predict page to make one!")
            else:
                st.write(f"Total predictions: {len(history)}")
                for item in history:
                    pred_id = item['id']
                    with st.expander(f"Prediction #{pred_id} — {item['created_at']}"):
                        
                        detail_resp = requests.get(f"{API_URL}/history/{pred_id}")
                        if detail_resp.status_code == 200:
                            detail = detail_resp.json()
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Original")
                                orig_bytes = base64.b64decode(detail["original_image"])
                                st.image(orig_bytes, width=250)
                            with col2:
                                st.write("Predicted")
                                pred_bytes = base64.b64decode(detail["predicted_image"])
                                st.image(pred_bytes, width=250)

                            st.write("")
                            if st.button(f"Delete Prediction #{pred_id}", key=f"del_{pred_id}"):
                                del_resp = requests.delete(f"{API_URL}/history/{pred_id}")
                                if del_resp.status_code == 200:
                                    st.success(f"Prediction #{pred_id} deleted!")
                                    st.rerun()
                                else:
                                    st.error("Failed to delete prediction.")
                        else:
                            st.error("Could not load prediction details")
        else:
            st.error("Could not load history")
    except Exception as e:
        st.error(f"Error connecting to API: {e}")