import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import altair as alt
from io import BytesIO
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Cattle Breed Identification",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .breed-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES = [
    "Ayrshire", "Brown Swiss", "Dangi", "Hallikar", "Gir",
    "Jaffarabadi buffalo", "Holstein friesian crossbreed",
    "Murrah", "Alambadi"
]

BREED_INFO = {
    "Ayrshire": {
        "description": "Known for high milk yield and strong adaptability. Commonly used in dairy production.",
        "origin": "Scotland",
        "type": "Dairy Cattle",
        "avg_milk": "7,000-9,000 kg/year",
        "characteristics": ["Hardy", "Adaptable", "Good Udder Health"],
        "optimal_climate": "Temperate, cool climates",
        "feed_requirements": "High-quality forage, grain supplements",
        "market_value": "High (dairy production)"
    },
    "Brown Swiss": {
        "description": "Renowned for strong build and excellent milk quality with high protein content.",
        "origin": "Switzerland",
        "type": "Dual Purpose",
        "avg_milk": "8,000-10,000 kg/year",
        "characteristics": ["Docile", "Long-lived", "High Protein Milk"],
        "optimal_climate": "Mountainous, temperate regions",
        "feed_requirements": "Grass-based diet, moderate grain",
        "market_value": "Very High (quality milk & meat)"
    },
    "Dangi": {
        "description": "Native breed suitable for draught purposes and moderate milk production.",
        "origin": "Maharashtra, India",
        "type": "Draught & Dairy",
        "avg_milk": "600-800 kg/lactation",
        "characteristics": ["Disease Resistant", "Heat Tolerant", "Strong"],
        "optimal_climate": "Hot, semi-arid regions",
        "feed_requirements": "Low maintenance, crop residues",
        "market_value": "Medium (draught work)"
    },
    "Hallikar": {
        "description": "Traditional draught breed, known for endurance and disease resistance.",
        "origin": "Karnataka, India",
        "type": "Draught",
        "avg_milk": "400-600 kg/lactation",
        "characteristics": ["Powerful", "Endurance", "Disease Resistant"],
        "optimal_climate": "Hot, dry regions",
        "feed_requirements": "Low maintenance, local fodder",
        "market_value": "Medium (agricultural work)"
    },
    "Gir": {
        "description": "High milk-producing Indian breed with adaptability to hot climates.",
        "origin": "Gujarat, India",
        "type": "Dairy Cattle",
        "avg_milk": "1,500-2,500 kg/lactation",
        "characteristics": ["Heat Tolerant", "Tick Resistant", "Docile"],
        "optimal_climate": "Hot, humid tropical",
        "feed_requirements": "Moderate, local fodder",
        "market_value": "High (indigenous dairy)"
    },
    "Jaffarabadi buffalo": {
        "description": "Excellent milk yield, often used in dairy farms for buffalo milk products.",
        "origin": "Gujarat, India",
        "type": "Dairy Buffalo",
        "avg_milk": "2,000-3,000 kg/lactation",
        "characteristics": ["Heavy Build", "High Fat Content", "Black Coat"],
        "optimal_climate": "Hot, humid coastal areas",
        "feed_requirements": "High-quality fodder, concentrates",
        "market_value": "Very High (premium milk)"
    },
    "Holstein friesian crossbreed": {
        "description": "High-yielding dairy cows, popular in commercial dairy farms.",
        "origin": "Netherlands/Germany",
        "type": "Dairy Cattle",
        "avg_milk": "6,000-8,000 kg/lactation",
        "characteristics": ["High Yield", "Black & White", "Fast Growth"],
        "optimal_climate": "Temperate, controlled environments",
        "feed_requirements": "High-quality feed, balanced diet",
        "market_value": "Very High (commercial dairy)"
    },
    "Murrah": {
        "description": "Premium buffalo breed for milk production and butterfat content.",
        "origin": "Haryana, India",
        "type": "Dairy Buffalo",
        "avg_milk": "2,000-2,500 kg/lactation",
        "characteristics": ["Curled Horns", "High Fat", "Black Color"],
        "optimal_climate": "Hot, humid subtropical",
        "feed_requirements": "Good quality green/dry fodder",
        "market_value": "Very High (quality milk)"
    },
    "Alambadi": {
        "description": "Local breed known for draught and moderate milk production.",
        "origin": "Tamil Nadu, India",
        "type": "Draught & Dairy",
        "avg_milk": "500-700 kg/lactation",
        "characteristics": ["Compact", "Hardy", "Red Color"],
        "optimal_climate": "Hot, dry tropical",
        "feed_requirements": "Low, drought-resistant fodder",
        "market_value": "Medium (local farming)"
    }
}

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_resource
def load_trained_model():
    try:
        model = load_model("high_res_best_model.h5", compile=False)
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image(img: Image.Image, img_size):
    img = img.resize(img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet_v2.preprocess_input(img_array)
    return img_array

def apply_image_enhancements(img, brightness, contrast, sharpness):
    """Apply image enhancements for better prediction"""
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)
    return img

def get_confidence_level(conf):
    if conf >= 90:
        return "Very High", "🟢"
    elif conf >= 75:
        return "High", "🟡"
    elif conf >= 60:
        return "Moderate", "🟠"
    else:
        return "Low", "🔴"

def batch_predict(images, model, img_size):
    """Process multiple images at once"""
    results = []
    for img in images:
        processed_img = preprocess_image(img, img_size)
        predictions = model.predict(processed_img, verbose=0)[0]
        top_idx = np.argmax(predictions)
        results.append({
            'breed': CLASS_NAMES[top_idx],
            'confidence': predictions[top_idx] * 100,
            'all_predictions': predictions
        })
    return results

def compare_predictions(pred1, pred2):
    """Compare two predictions and show differences"""
    comparison = []
    for i, breed in enumerate(CLASS_NAMES):
        comparison.append({
            'Breed': breed,
            'Image 1 (%)': pred1[i] * 100,
            'Image 2 (%)': pred2[i] * 100,
            'Difference (%)': abs(pred1[i] - pred2[i]) * 100
        })
    return pd.DataFrame(comparison)

def export_full_report(breed, confidence, predictions, img_name):
    """Generate a comprehensive JSON report"""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_name': img_name,
        'primary_prediction': {
            'breed': breed,
            'confidence': float(confidence),
            'confidence_level': get_confidence_level(confidence)[0]
        },
        'top_3_predictions': [
            {
                'rank': i + 1,
                'breed': CLASS_NAMES[idx],
                'confidence': float(predictions[idx] * 100)
            }
            for i, idx in enumerate(predictions.argsort()[-3:][::-1])
        ],
        'all_predictions': {
            breed: float(conf * 100) 
            for breed, conf in zip(CLASS_NAMES, predictions)
        },
        'breed_information': BREED_INFO[breed]
    }
    return json.dumps(report, indent=2)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2970/2970884.png", width=100)
    st.title("🎛️ Controls")
    
    mode = st.radio("Select Mode:", 
                    ["Single Image", "Batch Processing", "Image Comparison", "Prediction History"],
                    help="Choose analysis mode")
    
    if mode == "Single Image":
        st.markdown("### Image Enhancement")
        enhance_image = st.checkbox("Enable Enhancement", value=False)
        
        if enhance_image:
            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
            sharpness = st.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
        else:
            brightness = contrast = sharpness = 1.0
        
        st.markdown("### Confidence Threshold")
        conf_threshold = st.slider("Minimum Confidence (%)", 0, 100, 50, 5,
                                   help="Only show predictions above this threshold")
    
    st.markdown("---")
    st.title("About")
    st.markdown("""
    **Features:**
    - Single/Batch image analysis
    - Image comparison tool
    - Prediction history tracking
    - Image enhancement controls
    - Detailed breed information
    - Export capabilities (CSV/JSON)
    
    **Supported Breeds:** 9 cattle & buffalo breeds
    """)

# Main content
st.markdown('<h1 class="main-header">🐄 Cattle Breed Identification System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered breed recognition with multi-image analysis</p>', unsafe_allow_html=True)

# Load model
model, error = load_trained_model()

if error:
    st.error(f"❌ Error loading model: {error}")
    st.info("Please ensure 'high_res_best_model.h5' is in the same directory as this script.")
    st.stop()

IMG_SIZE = model.input_shape[1:3]

# Mode: Single Image
if mode == "Single Image":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader(
            "Upload an image of cattle or buffalo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, JPEG, PNG, WEBP"
        )
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        
        # Apply enhancements if enabled
        if enhance_image:
            img_enhanced = apply_image_enhancements(img, brightness, contrast, sharpness)
        else:
            img_enhanced = img
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            if enhance_image:
                tab1, tab2 = st.tabs(["Enhanced", "Original"])
                with tab1:
                    st.image(img_enhanced, use_container_width=True)
                with tab2:
                    st.image(img, use_container_width=True)
            else:
                st.image(img, use_container_width=True)
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"**File:** {uploaded_file.name}")
            st.markdown(f"**Size:** {img.size[0]} x {img.size[1]} pixels")
            st.markdown(f"**Format:** {img.format}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner("🤖 AI is analyzing the image..."):
                processed_img = preprocess_image(img_enhanced, IMG_SIZE)
                predictions = model.predict(processed_img, verbose=0)[0]
            
            top1_idx = np.argmax(predictions)
            top1_class = CLASS_NAMES[top1_idx]
            top1_conf = predictions[top1_idx] * 100
            conf_level, conf_icon = get_confidence_level(top1_conf)
            
            # Save to history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'image': uploaded_file.name,
                'breed': top1_class,
                'confidence': top1_conf
            })
            
            st.markdown(f"""
            <div class="breed-card">
                <h2 style="margin:0; font-size: 2rem;">🏆 {top1_class}</h2>
                <p style="font-size: 1.5rem; margin: 0.5rem 0;">
                    {conf_icon} {top1_conf:.2f}% Confidence
                </p>
                <p style="margin: 0; opacity: 0.9;">Confidence Level: {conf_level}</p>
            </div>
            """, unsafe_allow_html=True)
            
            info = BREED_INFO[top1_class]
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Origin", info["origin"])
            with metric_cols[1]:
                st.metric("Type", info["type"])
            with metric_cols[2]:
                st.metric("Avg. Milk", info["avg_milk"])
        
        st.markdown("---")
        st.subheader("📋 Detailed Information")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📖 Overview", "📊 Filtered Predictions", "📈 Full Analysis", "📥 Export"])
        
        with tab1:
            info = BREED_INFO[top1_class]
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Description:**")
                st.info(info["description"])
                st.markdown("**Optimal Climate:**")
                st.write(info["optimal_climate"])
                st.markdown("**Feed Requirements:**")
                st.write(info["feed_requirements"])
                
            with col2:
                st.markdown("**Key Characteristics:**")
                for char in info["characteristics"]:
                    st.markdown(f"✓ {char}")
                st.markdown("**Market Value:**")
                st.write(info["market_value"])
        
        with tab2:
            st.markdown(f"**Predictions above {conf_threshold}% confidence:**")
            filtered_preds = [(CLASS_NAMES[i], predictions[i] * 100) 
                            for i in range(len(predictions)) 
                            if predictions[i] * 100 >= conf_threshold]
            filtered_preds.sort(key=lambda x: x[1], reverse=True)
            
            if filtered_preds:
                for i, (breed, conf) in enumerate(filtered_preds):
                    _, icon = get_confidence_level(conf)
                    with st.expander(f"#{i+1} {icon} {breed} - {conf:.2f}%", expanded=(i==0)):
                        info = BREED_INFO[breed]
                        st.markdown(info["description"])
                        cols = st.columns(3)
                        cols[0].metric("Origin", info["origin"])
                        cols[1].metric("Type", info["type"])
                        cols[2].metric("Milk Yield", info["avg_milk"])
            else:
                st.warning(f"No predictions above {conf_threshold}% threshold.")
        
        with tab3:
            prob_df = pd.DataFrame({
                "Breed": CLASS_NAMES,
                "Probability": predictions * 100
            }).sort_values("Probability", ascending=False)
            
            chart = (
                alt.Chart(prob_df)
                .mark_bar()
                .encode(
                    x=alt.X("Probability:Q", title="Confidence (%)"),
                    y=alt.Y("Breed:N", sort="-x", title="Breed"),
                    color=alt.condition(
                        alt.datum.Breed == top1_class,
                        alt.value("#667eea"),
                        alt.value("#cccccc")
                    ),
                    tooltip=["Breed", alt.Tooltip("Probability:Q", format=".2f")]
                )
                .properties(height=400)
            )
            st.altair_chart(chart, use_container_width=True)
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                csv = prob_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv,
                    file_name=f"breed_predictions_{uploaded_file.name}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_report = export_full_report(top1_class, top1_conf, predictions, uploaded_file.name)
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_report,
                    file_name=f"breed_report_{uploaded_file.name}.json",
                    mime="application/json"
                )

# Mode: Batch Processing
elif mode == "Batch Processing":
    st.subheader("📦 Batch Image Processing")
    st.info("Upload multiple images to process them all at once")
    
    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Process All Images"):
            images = [Image.open(f).convert("RGB") for f in uploaded_files]
            
            with st.spinner(f"Processing {len(images)} images..."):
                results = batch_predict(images, model, IMG_SIZE)
            
            st.success(f"✅ Processed {len(results)} images successfully!")
            
            # Display results in grid
            cols_per_row = 3
            for i in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(results):
                        with col:
                            st.image(images[i + j], use_container_width=True)
                            result = results[i + j]
                            _, icon = get_confidence_level(result['confidence'])
                            st.markdown(f"**{icon} {result['breed']}**")
                            st.markdown(f"*{result['confidence']:.2f}% confidence*")
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Batch Summary")
            
            breed_counts = {}
            for result in results:
                breed_counts[result['breed']] = breed_counts.get(result['breed'], 0) + 1
            
            summary_df = pd.DataFrame([
                {'Breed': breed, 'Count': count, 'Percentage': (count/len(results))*100}
                for breed, count in breed_counts.items()
            ]).sort_values('Count', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                chart = alt.Chart(summary_df).mark_bar().encode(
                    x=alt.X('Count:Q', title='Number of Images'),
                    y=alt.Y('Breed:N', sort='-x', title='Breed'),
                    color=alt.value('#667eea'),
                    tooltip=['Breed', 'Count', alt.Tooltip('Percentage:Q', format='.1f')]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)
            
            with col2:
                st.dataframe(summary_df, use_container_width=True)
            
            # Export batch results
            batch_export = []
            for i, result in enumerate(results):
                batch_export.append({
                    'Image': uploaded_files[i].name,
                    'Predicted Breed': result['breed'],
                    'Confidence (%)': result['confidence']
                })
            
            batch_df = pd.DataFrame(batch_export)
            csv = batch_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Batch Results",
                data=csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )

# Mode: Image Comparison
elif mode == "Image Comparison":
    st.subheader("🔄 Compare Two Images")
    st.info("Upload two images to compare their predictions side-by-side")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader("Upload First Image", type=["jpg", "jpeg", "png", "webp"], key="img1")
    
    with col2:
        file2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png", "webp"], key="img2")
    
    if file1 and file2:
        img1 = Image.open(file1).convert("RGB")
        img2 = Image.open(file2).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img1, caption="Image 1", use_container_width=True)
        with col2:
            st.image(img2, caption="Image 2", use_container_width=True)
        
        if st.button("🔍 Compare Images"):
            with st.spinner("Analyzing both images..."):
                pred1 = model.predict(preprocess_image(img1, IMG_SIZE), verbose=0)[0]
                pred2 = model.predict(preprocess_image(img2, IMG_SIZE), verbose=0)[0]
            
            idx1 = np.argmax(pred1)
            idx2 = np.argmax(pred2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Image 1 Prediction", CLASS_NAMES[idx1], f"{pred1[idx1]*100:.2f}%")
            with col2:
                st.metric("Image 2 Prediction", CLASS_NAMES[idx2], f"{pred2[idx2]*100:.2f}%")
            
            if CLASS_NAMES[idx1] == CLASS_NAMES[idx2]:
                st.success("✅ Both images predicted as the same breed!")
            else:
                st.warning("⚠️ Different breed predictions detected")
            
            st.markdown("---")
            st.subheader("📊 Detailed Comparison")
            
            comparison_df = compare_predictions(pred1, pred2)
            comparison_df = comparison_df.sort_values('Difference (%)', ascending=False)
            
            st.dataframe(comparison_df.style.format({
                'Image 1 (%)': '{:.2f}',
                'Image 2 (%)': '{:.2f}',
                'Difference (%)': '{:.2f}'
            }).background_gradient(subset=['Difference (%)'], cmap='Reds'), use_container_width=True)
            
            # Side-by-side comparison chart
            comparison_long = pd.melt(
                comparison_df[['Breed', 'Image 1 (%)', 'Image 2 (%)']],
                id_vars=['Breed'],
                var_name='Image',
                value_name='Confidence (%)'
            )
            
            chart = alt.Chart(comparison_long).mark_bar().encode(
                x=alt.X('Breed:N', title='Breed'),
                y=alt.Y('Confidence (%):Q', title='Confidence (%)'),
                color=alt.Color('Image:N', scale=alt.Scale(scheme='category10')),
                xOffset='Image:N',
                tooltip=['Breed', 'Image', alt.Tooltip('Confidence (%):Q', format='.2f')]
            ).properties(height=400)
            
            st.altair_chart(chart, use_container_width=True)

# Mode: Prediction History
elif mode == "Prediction History":
    st.subheader("📜 Prediction History")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            st.metric("Unique Breeds", history_df['breed'].nunique())
        with col3:
            st.metric("Avg Confidence", f"{history_df['confidence'].mean():.2f}%")
        
        st.dataframe(history_df.style.format({'confidence': '{:.2f}%'}), use_container_width=True)
        
        # Breed distribution in history
        breed_dist = history_df['breed'].value_counts().reset_index()
        breed_dist.columns = ['Breed', 'Count']
        
        chart = alt.Chart(breed_dist).mark_arc().encode(
            theta=alt.Theta('Count:Q'),
            color=alt.Color('Breed:N', legend=alt.Legend(title="Breeds")),
            tooltip=['Breed', 'Count']
        ).properties(height=400, title="Breed Distribution in History")
        
        st.altair_chart(chart, use_container_width=True)
        
        # Export and clear options
        col1, col2 = st.columns(2)
        with col1:
            csv = history_df.to_csv(index=False)
            st.download_button(
                "📥 Export History",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
    else:
        st.info("No predictions yet. Start analyzing images to build your history!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Built with Streamlit & TensorFlow | Advanced Multi-Mode Analysis System</p>
</div>
""", unsafe_allow_html=True)