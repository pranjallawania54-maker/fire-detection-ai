import streamlit as st
from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import serial
import time

# ======================
# PAGE
# ======================
st.set_page_config(
    page_title="Fire Detection AI",
    layout="wide"
)

# ======================
# CSS
# ======================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.result {
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 10px;
    color: white;
}

.fire {
    background-color: #b02a37;
}

.smoke {
    background-color: #997404;
}

.safe {
    background-color: #146c43;
}

.blocked {
    background-color: #0d6efd;
}

</style>
""", unsafe_allow_html=True)

# ======================
# TITLE
# ======================
st.title("🔥 Fire Detection AI")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_ai_model():

    return load_model(
        "keras_Model.h5",
        compile=False
    )

model = load_ai_model()

# ======================
# LOAD LABELS
# ======================
@st.cache_data
def load_labels():

    return open(
        "labels.txt",
        "r"
    ).readlines()

class_names = load_labels()

# ======================
# SIDEBAR
# ======================
mode = st.sidebar.selectbox(
    "Mode",
    [
        "Upload Image",
        "Live Camera + Sensor"
    ]
)

# ======================
# PREDICTION FUNCTION
# ======================
def predict_fire(img):

    image = cv2.resize(
        img,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ).reshape(1, 224, 224, 3)

    image = (image / 127.5) - 1

    prediction = model.predict(
        image,
        verbose=0
    )

    index = np.argmax(prediction)

    class_name = class_names[
        index
    ][2:].strip().lower()

    confidence = float(
        prediction[0][index]
    )

    return class_name, confidence

# ======================
# IMAGE MODE
# ======================
if mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_np = np.array(image)

        class_name, confidence = predict_fire(
            image_np
        )

        confidence_percent = int(
            confidence * 100
        )

        st.image(
            image_np,
            width=500
        )

        if (
            "fire" in class_name
            and confidence > 0.90
        ):

            st.markdown(
                f"""
                <div class="result fire">
                FIRE {confidence_percent}%
                </div>
                """,
                unsafe_allow_html=True
            )

        elif (
            "smoke" in class_name
            and confidence > 0.90
        ):

            st.markdown(
                f"""
                <div class="result smoke">
                SMOKE {confidence_percent}%
                </div>
                """,
                unsafe_allow_html=True
            )

        elif (
            "blocked" in class_name
            and confidence > 0.90
        ):

            st.markdown(
                f"""
                <div class="result blocked">
                CAMERA BLOCKED {confidence_percent}%
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result safe">
                SAFE {confidence_percent}%
                </div>
                """,
                unsafe_allow_html=True
            )

# ======================
# LIVE CAMERA + SENSOR
# ======================
elif mode == "Live Camera + Sensor":

    st.subheader("📷 Camera + 🔥 Sensor")

    # ======================
    # CAMERA SELECTION
    # ======================
    camera_type = st.radio(
        "Select Camera",
        ["Laptop Camera", "Phone Camera"]
    )

    # Laptop Camera
    if camera_type == "Laptop Camera":

        camera_source = st.number_input(
            "Laptop Camera Index",
            min_value=0,
            max_value=5,
            value=0,
            step=1
        )

    # Phone Camera
    else:

        st.info(
            "Use IP Webcam App on phone"
        )

        camera_source = st.text_input(
            "Phone Camera URL",
            "http://192.168.1.5:8080/video"
        )

    # ======================
    # COM PORT
    # ======================
    port = st.text_input(
        "Arduino COM Port",
        "COM5"
    )

    # ======================
    # START BUTTON
    # ======================
    run = st.checkbox(
        "Start System"
    )

    FRAME_WINDOW = st.image([])

    sensor_box = st.empty()

    if run:

        # ======================
        # OPEN CAMERA
        # ======================
        camera = cv2.VideoCapture(
            camera_source
        )

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        # ======================
        # SENSOR
        # ======================
        arduino = None

        try:

            arduino = serial.Serial(
                port,
                9600,
                timeout=1
            )

            time.sleep(2)

            st.success(
                "✅ Sensor Connected"
            )

        except Exception as e:

            st.warning(
                f"⚠️ Sensor Not Connected: {e}"
            )

        # ======================
        # CAMERA CHECK
        # ======================
        if not camera.isOpened():

            st.error(
                "❌ Camera Not Connected"
            )

        else:

            while run:

                success, frame = camera.read()

                if not success:

                    st.error(
                        "❌ Camera Frame Error"
                    )

                    break

                # Flip Frame
                frame = cv2.flip(
                    frame,
                    1
                )

                # ======================
                # AI DETECTION
                # ======================
                try:

                    class_name, confidence = predict_fire(
                        frame
                    )

                    class_name = class_name.lower()

                    confidence_percent = int(
                        confidence * 100
                    )

                    # FIRE
                    if (
                        "fire" in class_name
                        and confidence > 0.90
                    ):

                        text = (
                            f"🔥 FIRE {confidence_percent}%"
                        )

                        color = (
                            0,
                            0,
                            255
                        )

                    # SMOKE
                    elif (
                        "smoke" in class_name
                        and confidence > 0.90
                    ):

                        text = (
                            f"💨 SMOKE {confidence_percent}%"
                        )

                        color = (
                            0,
                            255,
                            255
                        )

                    # BLOCKED
                    elif (
                        "blocked" in class_name
                        and confidence > 0.90
                    ):

                        text = (
                            f"🚫 BLOCKED {confidence_percent}%"
                        )

                        color = (
                            255,
                            0,
                            0
                        )

                    # SAFE
                    else:

                        text = (
                            f"✅ SAFE {confidence_percent}%"
                        )

                        color = (
                            0,
                            255,
                            0
                        )

                except Exception as e:

                    text = "AI ERROR"

                    color = (
                        120,
                        120,
                        120
                    )

                    print(e)

                # ======================
                # SENSOR DATA
                # ======================
                if arduino:

                    try:

                        if arduino.in_waiting > 0:

                            data = (
                                arduino.readline()
                                .decode(errors="ignore")
                                .strip()
                                .upper()
                            )

                            if data == "FIRE":

                                sensor_box.markdown(
                                    """
                                    <div class="result fire">
                                    🔥 SENSOR: FIRE DETECTED
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            elif data == "SMOKE":

                                sensor_box.markdown(
                                    """
                                    <div class="result smoke">
                                    💨 SENSOR: SMOKE DETECTED
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            else:

                                sensor_box.markdown(
                                    """
                                    <div class="result safe">
                                    ✅ SENSOR: SAFE
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    except Exception as sensor_error:

                        sensor_box.error(
                            f"Sensor Error: {sensor_error}"
                        )

                # ======================
                # DRAW TEXT
                # ======================
                cv2.putText(
                    frame,
                    text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2
                )

                # Convert RGB
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Show Frame
                FRAME_WINDOW.image(
                    frame,
                    channels="RGB"
                )

            # ======================
            # RELEASE
            # ======================
            camera.release()

            if arduino:
                arduino.close()