import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sys
import traceback

# Custom CSS for colorful frontend with background
st.set_page_config(page_title="ECG Signal Classification", layout="wide")

custom_css = """
<style>
    /* Background image */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Title styling */
    h1 {
        color: #ffffff;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    
    h2 {
        color: #e0e0e0;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }
    
    /* Main container styling */
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Input labels styling */
    label {
        color: #667eea !important;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 1.1em;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Number input styling */
    .stNumberInput input {
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    /* Subheader styling */
    h3 {
        color: #667eea;
        border-bottom: 3px solid #764ba2;
        padding-bottom: 10px;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda !important;
        border-left: 5px solid #28a745 !important;
        border-radius: 8px !important;
    }
    
    /* Info/diagnostics box styling */
    .stWrite {
        background-color: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    /* Text color */
    p, span, div {
        color: #333333;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Attempt to load model artifacts; record diagnostics instead of failing
diagnostics = {}

try:
    with open('random_forest_reduced_complexity_model.pkl', 'rb') as file:
        model = pickle.load(file)
    diagnostics['model_loaded'] = True
except Exception as e:
    model = None
    diagnostics['model_loaded'] = False
    diagnostics['model_load_error'] = str(e)

try:
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    diagnostics['scaler_loaded'] = True
except Exception as e:
    scaler = None
    diagnostics['scaler_loaded'] = False
    diagnostics['scaler_load_error'] = str(e)

try:
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    diagnostics['label_encoder_loaded'] = True
    diagnostics['label_encoder_classes'] = list(getattr(label_encoder, "classes_", []))
except Exception as e:
    label_encoder = None
    diagnostics['label_encoder_loaded'] = False
    diagnostics['label_encoder_load_error'] = str(e)
    diagnostics['label_encoder_classes'] = []

# Streamlit UI
st.title('🫀 ECG Signal Classification')
st.write('📊 Enter the ECG feature values below to predict the ECG signal type.')

feature_names = [
    'hbpermin', 'Pseg', 'PQseg', 'QTseg', 'STseg', 'Tseg', 'RRmean',
    'SDRR', 'SDSD', 'RMSSD', 'NN50', 'pNN50'
]

sample_data = {
    'hbpermin': 90.2686567164179,
    'Pseg': 0.0596560846560846,
    'PQseg': 0.05,
    'QTseg': 0.10271164021164,
    'STseg': 0.10271164021164,
    'Tseg': 0.0955687830687831,
    'RRmean': 241.073170731707,
    'SDRR': 4.50685247315165,
    'SDSD': 5.82832523114488,
    'RMSSD': 241.115294757205,
    'NN50': 0,
    'pNN50': 0.0
}

# Create columns for better layout
col1, col2, col3, col4 = st.columns(4)

input_data = {}
columns = [col1, col2, col3, col4]

for idx, feature in enumerate(feature_names):
    with columns[idx % 4]:
        input_data[feature] = st.number_input(
            f'{feature}',
            value=sample_data[feature],
            format="%.10f"
        )

# Center the button
col_button = st.columns([1, 1, 1])[1]
with col_button:
    button_clicked = st.button('🔍 Run Diagnostics', use_container_width=True)

# When user requests prediction, collect diagnostics and do not raise errors
if button_clicked:
    diag = dict(diagnostics)  # start with load diagnostics
    diag['python_executable'] = sys.executable
    diag['python_version'] = sys.version.split()[0]
    diag['input_data'] = input_data

    # Build DataFrame
    try:
        input_df = pd.DataFrame([input_data])
        diag['input_df_columns'] = list(input_df.columns)
    except Exception as e:
        input_df = None
        diag['input_df_error'] = str(e)
        diag['input_df_traceback'] = traceback.format_exc()

    # Scaler info / expected columns
    try:
        diag['scaler_feature_names_in'] = list(getattr(scaler, "feature_names_in_", [])) if scaler is not None else None
    except Exception as e:
        diag['scaler_feature_names_in'] = None
        diag['scaler_info_error'] = str(e)

    # Attempt to reorder to scaler features if available
    scaled_input = None
    if input_df is not None and scaler is not None:
        try:
            if hasattr(scaler, "feature_names_in_") and len(getattr(scaler, "feature_names_in_", [])) > 0:
                expected = list(scaler.feature_names_in_)
                missing = [c for c in expected if c not in input_df.columns]
                diag['scaler_expected_columns'] = expected
                diag['scaler_missing_columns'] = missing
                # only reorder columns that exist, do not fail
                available_expected = [c for c in expected if c in input_df.columns]
                if len(available_expected) > 0:
                    input_for_scale = input_df[available_expected]
                else:
                    input_for_scale = input_df
            else:
                input_for_scale = input_df

            # Try transform, fallback to values, capture errors
            try:
                scaled_input = scaler.transform(input_for_scale)
                diag['scaling_success'] = True
                diag['scaled_shape'] = getattr(scaled_input, 'shape', None)
            except Exception as e_scale:
                diag['scaling_error'] = str(e_scale)
                try:
                    scaled_input = scaler.transform(input_for_scale.values)
                    diag['scaling_success_fallback'] = True
                    diag['scaled_shape'] = getattr(scaled_input, 'shape', None)
                except Exception as e_scale2:
                    diag['scaling_fallback_error'] = str(e_scale2)
                    scaled_input = None
        except Exception as e:
            diag['scaler_preparation_error'] = str(e)

    else:
        if scaler is None:
            diag['scaling_skipped'] = "scaler_not_loaded"
        else:
            diag['scaling_skipped'] = "input_df_missing"

    # Model prediction (capture errors, do not raise)
    prediction_encoded = None
    if model is not None and scaled_input is not None:
        try:
            prediction_encoded = model.predict(scaled_input)
            diag['model_predict_success'] = True
            diag['raw_prediction'] = prediction_encoded.tolist() if hasattr(prediction_encoded, 'tolist') else str(prediction_encoded)
        except Exception as e:
            diag['model_predict_error'] = str(e)
            diag['model_predict_traceback'] = traceback.format_exc()
    else:
        if model is None:
            diag['model_predict_skipped'] = "model_not_loaded"
        if scaled_input is None:
            diag['model_predict_skipped_scaled'] = "no_scaled_input"

    # Label decode
    decoded_label = None
    if label_encoder is not None and prediction_encoded is not None:
        try:
            decoded = label_encoder.inverse_transform(prediction_encoded)
            decoded_label = decoded.tolist() if hasattr(decoded, 'tolist') else str(decoded)
            diag['decoded_prediction'] = decoded_label
        except Exception as e:
            diag['label_inverse_error'] = str(e)
            # include label encoder classes for debugging
            try:
                diag['label_encoder_known_classes'] = list(label_encoder.classes_)
            except Exception:
                diag['label_encoder_known_classes'] = None
    else:
        if label_encoder is None:
            diag['label_decode_skipped'] = "label_encoder_not_loaded"
        if prediction_encoded is None:
            diag['label_decode_skipped_pred'] = "no_prediction"

    # Display only diagnostics (no traceback unless included in values above)
    st.subheader("📋 Diagnostics Report")
    for k, v in diag.items():
        st.write(f"**{k}:** {v}")