import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Sayfa Ayarları
st.set_page_config(page_title="OmniSight AI", layout="wide", page_icon="👁️")

# CSS: Fütüristik / Askeri HUD Teması
st.markdown("""
    <style>
    .stApp { background-color: #050505; }
    h1 { color: #00ffcc !important; font-family: 'Orbitron', sans-serif; text-shadow: 0 0 10px #00ffcc; }
    .stFileUploader { border: 2px dashed #00ffcc; border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

# Başlık
st.title("👁️ OmniSight: Real-Time Object Detection")
st.markdown("**System:** :green[ONLINE] | **Model:** YOLOv8 Neural Network | **Capability:** 80+ Object Classes")

# Sidebar
st.sidebar.header("⚙️ Detection Config")
confidence = st.sidebar.slider("AI Confidence Threshold", 0.25, 1.0, 0.40)
st.sidebar.info("Model detects common objects: Person, Car, Cell Phone, Laptop, Dog, etc.")

# Modeli Yükle 
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt') 

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")

# Görsel Yükleme
uploaded_file = st.file_uploader("📂 Upload Environmental Scan...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Resmi Hazırla
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Yapay Zeka Taraması 
    with st.spinner('🤖 Analyzing visual data... Identifying objects...'):
        results = model.predict(img_array, conf=confidence)
        
        # Sonuç Görselini Al
        res_plotted = results[0].plot() # Kutuları çizilmiş halini verir
        
        # Renkleri Düzelt (BGR -> RGB)
        res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        
        # Ekrana Bas
        st.image(res_image, caption='OmniSight AI Output', use_container_width=True)
        
        # İstatistikler (Ne buldu?)
        st.subheader("📊 Scan Inventory")
        
        # Bulunan nesneleri say
        boxes = results[0].boxes
        detected_objects = {}
        
        if boxes:
            for box in boxes:
                cls = int(box.cls[0]) # Sınıf ID'si
                name = model.names[cls] # Sınıf Adı 
                detected_objects[name] = detected_objects.get(name, 0) + 1
            
            
            cols = st.columns(len(detected_objects))
            for idx, (obj_name, count) in enumerate(detected_objects.items()):
                
                col_idx = idx % 4 
                if idx < 4: 
                     c = cols[idx]
                else:
                    
                     break 
                
                with cols[idx] if idx < len(cols) else cols[0]:
                    st.metric(label=obj_name.upper(), value=count)
            
            st.success(f"✅ Detection Complete: {sum(detected_objects.values())} objects identified.")
            
            # Detaylı Liste 
            with st.expander("📂 View Raw Detection Log"):
                 st.write(detected_objects)
                 
        else:
            st.warning("⚠️ No recognized objects detected in this sector.")