import streamlit as st
from pathlib import Path
from src.entry import entry_point
import tempfile

st.set_page_config(page_title="OMR Checker", layout="wide")
st.title("OMR Checker Web App")

st.write("### Upload the following files together in one go:")
st.write("- `template.json`")
st.write("- `OMRMarker.jpg`")
st.write("- OMR sheet images (JPG/PNG)")

uploaded_files = st.file_uploader(
    "Upload files",
    type=["json", "jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)

        # Save all uploaded files in the temp folder
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

        # Verify required files exist
        if not any(f.name.endswith(".json") for f in temp_dir.iterdir()):
            st.error("❌ You must upload a template.json file!")
        elif not any(f.name.lower() == "omrmarker.jpg" for f in temp_dir.iterdir()):
            st.error("❌ You must upload OMRMarker.jpg!")
        else:
            st.success("✅ All required files uploaded!")

            args = {
                "input_paths": [str(temp_dir)],
                "output_dir": "outputs",
                "debug": False,
                "autoAlign": False,
                "setLayout": False,
            }

            st.info("Processing the OMR sheet... This may take a few seconds.")

            try:
                # Pass only the folder; entry_point reads template.json & OMRMarker.jpg automatically
                entry_point(str(temp_dir), args)
                st.success("✅ OMR sheet processed successfully! Check the 'outputs' folder.")
            except Exception as e:
                st.error(f"❌ Error processing OMR sheet: {e}")
