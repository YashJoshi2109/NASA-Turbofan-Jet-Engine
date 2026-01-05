import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, classification_report)
import xgboost as xgb
import warnings
import os
import kagglehub
from io import StringIO
import pickle

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NASA Predictive Maintenance - RUL Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'valid_data' not in st.session_state:
    st.session_state.valid_data = None
if 'y_valid' not in st.session_state:
    st.session_state.y_valid = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'pickled_models_loaded' not in st.session_state:
    st.session_state.pickled_models_loaded = False
if 'pickled_models' not in st.session_state:
    st.session_state.pickled_models = {}
if 'feature_info' not in st.session_state:
    st.session_state.feature_info = None

# Helper functions
@st.cache_data
def load_data():
    """Load NASA C-MAPSS dataset"""
    try:
        # Try to download dataset
        dataset_path = kagglehub.dataset_download("behrad3d/nasa-cmaps")
        data_path = f"{dataset_path}/CMaps"
        
        # Define column names
        index_names = ['unit_number', 'time_cycles']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = [f's_{i+1}' for i in range(0, 21)]
        col_names = index_names + setting_names + sensor_names
        
        # Load data
        train = pd.read_csv(f'{data_path}/train_FD001.txt', sep=r'\s+', header=None, 
                           index_col=False, names=col_names)
        valid = pd.read_csv(f'{data_path}/test_FD001.txt', sep=r'\s+', header=None, 
                           index_col=False, names=col_names)
        y_valid = pd.read_csv(f'{data_path}/RUL_FD001.txt', sep=r'\s+', header=None, 
                             index_col=False, names=['RUL'])
        
        return train, valid, y_valid
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure kagglehub is installed and you have internet connection")
        return None, None, None

def add_RUL_column(df):
    """Add RUL column to dataframe"""
    df = df.copy()
    rul = []
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit]
        max_cycle = unit_data['time_cycles'].max()
        unit_rul = max_cycle - unit_data['time_cycles']
        rul.extend(unit_rul.values)
    df['RUL'] = rul
    return df

def clip_rul(df, clip_value=195):
    """Clip RUL values at specified threshold"""
    df = df.copy()
    df['RUL'] = df['RUL'].clip(upper=clip_value)
    return df

def prepare_features(df, drop_labels=None, include_historical=False):
    """Prepare features for modeling"""
    df = df.copy()
    
    # Drop specified labels
    if drop_labels:
        df = df.drop(columns=drop_labels, errors='ignore')
    
    # Drop unit_number and time_cycles
    if 'unit_number' in df.columns:
        df = df.drop(columns=['unit_number'])
    if 'time_cycles' in df.columns:
        df = df.drop(columns=['time_cycles'])
    
    # Separate features and target
    if 'RUL' in df.columns:
        X = df.drop(columns=['RUL'])
        y = df['RUL']
        return X, y
    else:
        return df, None

def evaluate_regression(y_true, y_pred):
    """Calculate regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'R² Score': r2, 'MAE': mae}

def evaluate_classification(y_true, y_pred):
    """Calculate classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}

@st.cache_resource
def load_pickled_models():
    """Load pre-trained models from pickle files"""
    models = {}
    scaler = None
    feature_info = None
    classification_models = {}
    load_errors = []
    
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        return None, None, None, classification_models, load_errors
    
    try:
        # Load scaler
        if os.path.exists(f'{models_dir}/scaler.pkl'):
            with open(f'{models_dir}/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        
        # Load feature info
        if os.path.exists(f'{models_dir}/feature_info.pkl'):
            with open(f'{models_dir}/feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)
        
        # Load regression models
        regression_model_files = {
            'linear_regression': f'{models_dir}/linear_regression_model.pkl',
            'svr': f'{models_dir}/svr_model.pkl',
            'random_forest': f'{models_dir}/random_forest_model.pkl',
            'xgboost': f'{models_dir}/xgboost_model.pkl'
        }
        
        for model_name, file_path in regression_model_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        models[model_name] = pickle.load(f)
                except Exception as e:
                    load_errors.append(f"{model_name}: {e}")
        
        # Load classification models (if they exist)
        classification_model_files = {
            'svc': f'{models_dir}/svc_classifier_model.pkl',
            'random_forest_classifier': f'{models_dir}/random_forest_classifier_model.pkl',
            'naive_bayes': f'{models_dir}/naive_bayes_model.pkl',
            'knn': f'{models_dir}/knn_classifier_model.pkl'
        }
        
        for model_name, file_path in classification_model_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'rb') as f:
                        classification_models[model_name] = pickle.load(f)
                except Exception as e:
                    load_errors.append(f"{model_name}: {e}")
        
        return models, scaler, feature_info, classification_models, load_errors
    
    except Exception as e:
        load_errors.append(str(e))
        return None, None, None, classification_models, load_errors

# Sidebar navigation
st.sidebar.title("✈️ Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "📊 Data Overview", "🔍 Exploratory Data Analysis", 
     "⚙️ Feature Engineering", "🤖 Regression Models", "📈 Classification Models",
     "📊 Model Comparison", "🔮 Predictions", "📈 Performance Analysis"]
)

# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header">NASA Predictive Maintenance System</div>', unsafe_allow_html=True)
    st.markdown("### Remaining Useful Life (RUL) Prediction for Turbofan Engines")

    repo_url = "https://github.com/YashJoshi2109/NASA-Turbofan-Engine-Degradation-Simulation"
    kaggle_url = "https://www.kaggle.com/datasets/behrad3d/nasa-cmaps"

    st.markdown(
        f"""
        <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:center; margin-bottom:10px;">
            <a href="{repo_url}" target="_blank" style="text-decoration:none;">
                <button style="background-color:#1f77b4; color:white; border:none; padding:8px 12px; border-radius:6px;">GitHub Repository</button>
            </a>
            <a href="{kaggle_url}" target="_blank" style="text-decoration:none;">
                <button style="background-color:#ff7f0e; color:white; border:none; padding:8px 12px; border-radius:6px;">Kaggle Dataset</button>
            </a>
            <span style="font-weight:600; color:#444;">Made by Yash Joshi</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Training Samples", "20,631")
    with col2:
        st.metric("Validation Samples", "13,096")
    with col3:
        st.metric("Features", "24 (21 Sensors + 3 Settings)")
    
    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This application provides a comprehensive predictive maintenance system for turbofan engines using NASA's C-MAPSS dataset.
    
    **Key Features:**
    - 📊 **Data Exploration**: Comprehensive EDA with correlation analysis and visualizations
    - ⚙️ **Feature Engineering**: RUL calculation, moving averages, feature selection
    - 🤖 **Regression Models**: Linear Regression, SVR, Random Forest, XGBoost
    - 📈 **Classification Models**: SVC, Random Forest, Naive Bayes, KNN
    - 📊 **Model Comparison**: Side-by-side performance metrics
    - 🔮 **Predictions**: Real-time RUL predictions for new engine data
    
    **Dataset Information:**
    - **Source**: NASA C-MAPSS FD001 dataset
    - **Objective**: Predict Remaining Useful Life (RUL) of aircraft engines
    - **Application**: Enable proactive maintenance scheduling
    
    **Getting Started:**
    1. Navigate to "Data Overview" to load and explore the dataset
    2. Use "Exploratory Data Analysis" to understand data patterns
    3. Apply "Feature Engineering" to prepare data for modeling
    4. Train models in "Regression Models" or "Classification Models"
    5. Compare performance in "Model Comparison"
    6. Make predictions in "Predictions" section
    """)
    
    if st.button("🚀 Load Dataset", use_container_width=True):
        with st.spinner("Loading NASA C-MAPSS dataset..."):
            train, valid, y_valid = load_data()
            if train is not None:
                st.session_state.train_data = train
                st.session_state.valid_data = valid
                st.session_state.y_valid = y_valid
                st.session_state.data_loaded = True
                st.success("✅ Dataset loaded successfully!")
                st.balloons()

# Data Overview Page
elif page == "📊 Data Overview":
    st.title("📊 Data Overview")
    
    if not st.session_state.data_loaded:
        if st.button("Load Dataset"):
            with st.spinner("Loading data..."):
                train, valid, y_valid = load_data()
                if train is not None:
                    st.session_state.train_data = train
                    st.session_state.valid_data = valid
                    st.session_state.y_valid = y_valid
                    st.session_state.data_loaded = True
                    st.rerun()
    
    if st.session_state.data_loaded:
        train = st.session_state.train_data
        valid = st.session_state.valid_data
        y_valid = st.session_state.y_valid
        
        # Dataset Statistics
        st.subheader("📈 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Samples", f"{len(train):,}")
        with col2:
            st.metric("Validation Samples", f"{len(valid):,}")
        with col3:
            st.metric("Features", len(train.columns))
        with col4:
            st.metric("Engines (Train)", train['unit_number'].nunique())
        
        # Data Preview
        st.subheader("📋 Data Preview")
        tab1, tab2, tab3 = st.tabs(["Training Data", "Validation Data", "RUL Labels"])
        
        with tab1:
            st.dataframe(train.head(100), use_container_width=True)
            st.info(f"Shape: {train.shape}")
        
        with tab2:
            st.dataframe(valid.head(100), use_container_width=True)
            st.info(f"Shape: {valid.shape}")
        
        with tab3:
            st.dataframe(y_valid.head(100), use_container_width=True)
            st.info(f"Shape: {y_valid.shape}")
        
        # Basic Statistics
        st.subheader("📊 Descriptive Statistics")
        st.dataframe(train.describe(), use_container_width=True)
        
        # Missing Values
        st.subheader("🔍 Data Quality Check")
        col1, col2 = st.columns(2)
        
        with col1:
            missing_train = train.isnull().sum()
            if missing_train.sum() > 0:
                st.write("**Missing Values (Training):**")
                st.dataframe(missing_train[missing_train > 0])
            else:
                st.success("✅ No missing values in training data")
        
        with col2:
            missing_valid = valid.isnull().sum()
            if missing_valid.sum() > 0:
                st.write("**Missing Values (Validation):**")
                st.dataframe(missing_valid[missing_valid > 0])
            else:
                st.success("✅ No missing values in validation data")
        
        # Engine Lifetime Distribution
        st.subheader("⏱️ Engine Lifetime Distribution")
        train_with_rul = add_RUL_column(train)
        max_cycles = train_with_rul.groupby('unit_number')['time_cycles'].max()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(max_cycles, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Engine Lifetime (Cycles)')
        ax.set_ylabel('Number of Engines')
        ax.set_title('Distribution of Engine Lifetimes')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# Exploratory Data Analysis Page
elif page == "🔍 Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Data Overview page.")
    else:
        train = st.session_state.train_data.copy()
        train_with_rul = add_RUL_column(train)
        
        st.subheader("📊 Correlation Analysis")
        
        # Calculate correlation matrix
        numeric_cols = train_with_rul.select_dtypes(include=[np.number]).columns
        corr_matrix = train_with_rul[numeric_cols].corr()
        
        # Correlation with RUL
        st.write("**Correlation with RUL:**")
        rul_corr = corr_matrix['RUL'].sort_values(ascending=False)
        st.dataframe(rul_corr, use_container_width=True)
        
        # Full Correlation Heatmap
        st.write("**Full Correlation Matrix Heatmap:**")
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        st.pyplot(fig)
        
        # Feature Evolution Plots
        st.subheader("📈 Feature Evolution Over Time")
        
        selected_engine = st.selectbox("Select Engine Unit", train['unit_number'].unique()[:20])
        selected_features = st.multiselect(
            "Select Features to Plot",
            options=[col for col in train.columns if col not in ['unit_number', 'time_cycles']],
            default=['s_2', 's_3', 's_4', 's_7', 's_11', 's_12']
        )
        
        if selected_features:
            engine_data = train_with_rul[train_with_rul['unit_number'] == selected_engine]
            
            fig, axes = plt.subplots(len(selected_features), 1, figsize=(12, 3*len(selected_features)))
            if len(selected_features) == 1:
                axes = [axes]
            
            for idx, feature in enumerate(selected_features):
                axes[idx].plot(engine_data['time_cycles'], engine_data[feature], 
                              label=feature, linewidth=2)
                axes[idx].set_xlabel('Time Cycles')
                axes[idx].set_ylabel(feature)
                axes[idx].set_title(f'{feature} Evolution for Engine {selected_engine}')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend()
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Sensor vs RUL
        st.subheader("🔗 Sensor Readings vs RUL")
        sensor_col = st.selectbox("Select Sensor", 
                                  [col for col in train.columns if col.startswith('s_')])
        
        if sensor_col:
            fig, ax = plt.subplots(figsize=(10, 6))
            sample_engines = train['unit_number'].unique()[:5]
            for engine in sample_engines:
                engine_data = train_with_rul[train_with_rul['unit_number'] == engine]
                ax.scatter(engine_data['RUL'], engine_data[sensor_col], 
                          alpha=0.6, label=f'Engine {engine}', s=20)
            ax.set_xlabel('RUL (Remaining Useful Life)')
            ax.set_ylabel(sensor_col)
            ax.set_title(f'{sensor_col} vs RUL')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Feature Distributions
        st.subheader("📊 Feature Distributions")
        dist_features = st.multiselect(
            "Select Features for Distribution",
            options=[col for col in train.columns if col not in ['unit_number', 'time_cycles']],
            default=['s_2', 's_3', 's_4']
        )
        
        if dist_features:
            n_cols = min(3, len(dist_features))
            n_rows = (len(dist_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            
            # Handle different subplot configurations - ensure axes is always a list/array
            if n_rows == 1 and n_cols == 1:
                # Single subplot - axes is a single Axes object
                axes = np.array([axes])
            elif not isinstance(axes, np.ndarray):
                # If axes is not an array, make it one
                axes = np.array([axes])
            else:
                # Flatten to 1D array for consistent indexing
                axes = axes.flatten()
            
            for idx, feature in enumerate(dist_features):
                ax = axes[idx]
                ax.hist(train[feature], bins=50, edgecolor='black', alpha=0.7)
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {feature}')
                ax.grid(True, alpha=0.3)
            
            # Hide extra subplots
            if len(dist_features) < len(axes):
                for idx in range(len(dist_features), len(axes)):
                    axes[idx].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)

# Feature Engineering Page
elif page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Data Overview page.")
    else:
        train = st.session_state.train_data.copy()
        
        st.subheader("🔧 Feature Engineering Options")
        
        # RUL Calculation
        st.write("**1. RUL Calculation**")
        st.info("RUL (Remaining Useful Life) is calculated as: max_cycles - current_cycle for each engine")
        
        if st.button("Calculate RUL"):
            train_with_rul = add_RUL_column(train)
            st.session_state.train_data = train_with_rul
            st.success("✅ RUL column added!")
            st.dataframe(train_with_rul[['unit_number', 'time_cycles', 'RUL']].head(20))
        
        # RUL Clipping
        st.write("**2. RUL Clipping**")
        clip_value = st.slider("RUL Clip Value", min_value=100, max_value=300, value=195)
        
        if st.button("Apply RUL Clipping"):
            if 'RUL' in train.columns:
                train_clipped = clip_rul(train, clip_value)
                st.session_state.train_data = train_clipped
                st.success(f"✅ RUL clipped at {clip_value} cycles!")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(train['RUL'], bins=50, alpha=0.5, label='Original', edgecolor='black')
                ax.hist(train_clipped['RUL'], bins=50, alpha=0.5, label='Clipped', edgecolor='black')
                ax.set_xlabel('RUL')
                ax.set_ylabel('Frequency')
                ax.set_title('RUL Distribution: Original vs Clipped')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            else:
                st.error("Please calculate RUL first!")
        
        # Feature Selection
        st.write("**3. Feature Selection**")
        st.info("Remove constant or non-informative features")
        
        # Identify constant features
        constant_features = []
        for col in train.select_dtypes(include=[np.number]).columns:
            if train[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            st.write(f"**Constant Features Found:** {constant_features}")
            if st.button("Remove Constant Features"):
                train = train.drop(columns=constant_features)
                st.session_state.train_data = train
                st.success(f"✅ Removed {len(constant_features)} constant features!")
        else:
            st.success("✅ No constant features found!")
        
        # Feature Statistics
        st.subheader("📊 Feature Statistics")
        if st.checkbox("Show Feature Statistics"):
            st.dataframe(train.describe(), use_container_width=True)

# Regression Models Page
elif page == "🤖 Regression Models":
    st.title("🤖 Regression Models")
    
    # Check for pickled models
    use_pickled = st.checkbox("Use Pre-trained Pickled Models (from notebook)", value=False)
    
    if use_pickled:
        if not st.session_state.pickled_models_loaded:
            with st.spinner("Loading pickled models..."):
                models, scaler, feature_info, class_models, load_errors = load_pickled_models()
                if models or class_models:
                    st.session_state.pickled_models = models
                    st.session_state.pickled_classification_models = class_models
                    if scaler:
                        st.session_state.scaler = scaler
                    if feature_info:
                        st.session_state.feature_info = feature_info
                    st.session_state.pickled_models_loaded = True
                    total_models = len(models) + len(class_models)
                    st.success(f"✅ Loaded {total_models} pre-trained model(s)! ({len(models)} regression, {len(class_models)} classification)")
                    if load_errors:
                        st.warning("Some pickled models were skipped:\n" + "\n".join(load_errors))
                else:
                    st.error("❌ Could not load pickled models. Please ensure models are saved in 'models/' directory.")
                    st.info("Run the code from 'NOTEBOOK_CELL_SAVE_MODELS.py' in your notebook to create pickle files.")
                    use_pickled = False
        
        if st.session_state.pickled_models_loaded:
            st.subheader("📦 Loaded Pre-trained Models")
            
            available_models = list(st.session_state.pickled_models.keys())
            model_display_names = {
                'linear_regression': 'Linear Regression',
                'svr': 'SVR',
                'random_forest': 'Random Forest',
                'xgboost': 'XGBoost'
            }
            
            st.write("**Available Models:**")
            for model_key in available_models:
                st.write(f"✓ {model_display_names.get(model_key, model_key)}")
            
            if st.session_state.feature_info:
                st.write("**Feature Information:**")
                st.json(st.session_state.feature_info)
            
            st.info("💡 These models are ready to use for predictions! Go to the 'Predictions' page.")
    
    if not st.session_state.data_loaded and not use_pickled:
        st.warning("⚠️ Please load the dataset first from the Data Overview page, or use pre-trained models.")
    elif not use_pickled:
        train = st.session_state.train_data.copy()
        
        # Check if RUL exists
        if 'RUL' not in train.columns:
            st.warning("⚠️ RUL column not found. Calculating RUL...")
            train = add_RUL_column(train)
            st.session_state.train_data = train
        
        st.subheader("⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.4, 0.3)
            random_state = st.number_input("Random State", value=42)
        with col2:
            use_scaling = st.checkbox("Use Feature Scaling", value=True)
            clip_rul_val = st.checkbox("Clip RUL at 195", value=True)
        
        # Prepare data
        if clip_rul_val:
            train = clip_rul(train, 195)
        
        # Drop labels
        drop_labels = ['unit_number', 'time_cycles']
        X, y = prepare_features(train, drop_labels=drop_labels)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scaling
        if use_scaling:
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            st.session_state.scaler = scaler
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        st.subheader("🚀 Train Models")
        
        models_to_train = st.multiselect(
            "Select Models to Train",
            ["Linear Regression", "SVR", "Random Forest", "XGBoost"],
            default=["Linear Regression", "SVR", "Random Forest", "XGBoost"]
        )
        
        if st.button("Train Selected Models"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            models = {}
            results = {}
            
            for idx, model_name in enumerate(models_to_train):
                status_text.text(f"Training {model_name}...")
                progress_bar.progress((idx + 1) / len(models_to_train))
                
                try:
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                        model.fit(X_train_scaled, y_train)
                        y_pred_train = model.predict(X_train_scaled)
                        y_pred_test = model.predict(X_test_scaled)
                    
                    elif model_name == "SVR":
                        model = SVR(kernel='rbf')
                        model.fit(X_train_scaled, y_train)
                        y_pred_train = model.predict(X_train_scaled)
                        y_pred_test = model.predict(X_test_scaled)
                    
                    elif model_name == "Random Forest":
                        model = RandomForestRegressor(max_features="sqrt", random_state=42, n_estimators=100)
                        model.fit(X_train_scaled, y_train)
                        y_pred_train = model.predict(X_train_scaled)
                        y_pred_test = model.predict(X_test_scaled)
                    
                    elif model_name == "XGBoost":
                        model = xgb.XGBRegressor(n_estimators=110, learning_rate=0.02, 
                                                gamma=0, subsample=0.8, colsample_bytree=0.5, 
                                                max_depth=3, random_state=42)
                        model.fit(X_train_scaled, y_train)
                        y_pred_train = model.predict(X_train_scaled)
                        y_pred_test = model.predict(X_test_scaled)
                    
                    # Evaluate
                    train_metrics = evaluate_regression(y_train, y_pred_train)
                    test_metrics = evaluate_regression(y_test, y_pred_test)
                    
                    models[model_name] = model
                    results[model_name] = {
                        'train': train_metrics,
                        'test': test_metrics,
                        'y_pred_test': y_pred_test,
                        'y_test': y_test
                    }
                    
                except Exception as e:
                    st.error(f"Error training {model_name}: {str(e)}")
            
            st.session_state.models_trained = models
            st.session_state.regression_results = results
            status_text.text("✅ Training complete!")
            progress_bar.progress(1.0)
            st.success("All models trained successfully!")
        
        # Display Results
        if 'regression_results' in st.session_state:
            st.subheader("📊 Model Performance")
            
            results = st.session_state.regression_results
            
            # Metrics Table
            metrics_data = []
            for model_name, result in results.items():
                metrics_data.append({
                    'Model': model_name,
                    'Train RMSE': f"{result['train']['RMSE']:.2f}",
                    'Train R²': f"{result['train']['R² Score']:.3f}",
                    'Test RMSE': f"{result['test']['RMSE']:.2f}",
                    'Test R²': f"{result['test']['R² Score']:.3f}",
                    'Test MAE': f"{result['test']['MAE']:.2f}"
                })
            
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
            
            # Visualizations
            st.subheader("📈 Predictions Visualization")
            
            selected_model = st.selectbox("Select Model for Visualization", list(results.keys()))
            
            if selected_model:
                result = results[selected_model]
                y_test = result['y_test']
                y_pred = result['y_pred_test']
                
                # Predicted vs Actual
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Scatter plot
                axes[0].scatter(y_test, y_pred, alpha=0.5)
                axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                           'r--', lw=2, label='Perfect Prediction')
                axes[0].set_xlabel('Actual RUL')
                axes[0].set_ylabel('Predicted RUL')
                axes[0].set_title(f'{selected_model}: Predicted vs Actual RUL')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Bar chart comparison (sample)
                sample_size = min(50, len(y_test))
                indices = np.arange(sample_size)
                axes[1].bar(indices - 0.2, y_test.iloc[:sample_size], 0.4, 
                           label='Actual', alpha=0.7)
                axes[1].bar(indices + 0.2, y_pred[:sample_size], 0.4, 
                           label='Predicted', alpha=0.7)
                axes[1].set_xlabel('Sample Index')
                axes[1].set_ylabel('RUL')
                axes[1].set_title(f'{selected_model}: Sample Predictions')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Residuals plot
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_pred, residuals, alpha=0.5)
                ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
                ax.set_xlabel('Predicted RUL')
                ax.set_ylabel('Residuals')
                ax.set_title(f'{selected_model}: Residuals Plot')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

# Classification Models Page
elif page == "📈 Classification Models":
    st.title("📈 Classification Models")
    
    # Check for pickled classification models
    use_pickled_class = st.checkbox("Use Pre-trained Pickled Classification Models (from notebook)", value=False)
    
    if use_pickled_class:
        if not st.session_state.pickled_models_loaded:
            with st.spinner("Loading pickled models..."):
                models, scaler, feature_info, class_models, load_errors = load_pickled_models()
                if class_models:
                    st.session_state.pickled_classification_models = class_models
                    if scaler:
                        st.session_state.scaler = scaler
                    if feature_info:
                        st.session_state.feature_info = feature_info
                    st.session_state.pickled_models_loaded = True
                    st.success(f"✅ Loaded {len(class_models)} pre-trained classification model(s)!")
                    if load_errors:
                        st.warning("Some pickled models were skipped:\n" + "\n".join(load_errors))
                else:
                    st.error("❌ Could not load pickled classification models.")
                    st.info("Make sure you've saved classification models using the notebook script.")
                    use_pickled_class = False
        
        if st.session_state.pickled_models_loaded and 'pickled_classification_models' in st.session_state:
            st.subheader("📦 Loaded Pre-trained Classification Models")
            available_models = list(st.session_state.pickled_classification_models.keys())
            model_display_names = {
                'svc': 'SVC',
                'random_forest_classifier': 'Random Forest',
                'naive_bayes': 'Naive Bayes',
                'knn': 'KNN'
            }
            
            st.write("**Available Models:**")
            for model_key in available_models:
                st.write(f"✓ {model_display_names.get(model_key, model_key)}")
            
            st.info("💡 These models are ready to use for predictions!")
    
    if not st.session_state.data_loaded and not use_pickled_class:
        st.warning("⚠️ Please load the dataset first from the Data Overview page, or use pre-trained models.")
    elif not use_pickled_class:
        train = st.session_state.train_data.copy()
        
        if 'RUL' not in train.columns:
            st.warning("⚠️ RUL column not found. Calculating RUL...")
            train = add_RUL_column(train)
            st.session_state.train_data = train
        
        st.subheader("⚙️ Classification Setup")
        
        # Convert RUL to classes
        st.write("**RUL to Class Conversion**")
        n_classes = st.slider("Number of Classes", 2, 5, 3)
        
        def transform_to_classes(rul, n_bins):
            """Transform RUL to classes using quantile-based binning"""
            bins = pd.qcut(rul, q=n_bins, labels=False, duplicates='drop')
            return bins
        
        if st.button("Convert RUL to Classes"):
            train_class = train.copy()
            train_class['RUL_class'] = transform_to_classes(train_class['RUL'], n_classes)
            st.session_state.train_data = train_class
            st.success(f"✅ RUL converted to {n_classes} classes!")
            
            # Show class distribution
            class_dist = train_class['RUL_class'].value_counts().sort_index()
            st.bar_chart(class_dist)
        
        if 'RUL_class' in train.columns:
            # Prepare data
            drop_labels = ['unit_number', 'time_cycles', 'RUL', 'RUL_class']
            X, _ = prepare_features(train, drop_labels=drop_labels)
            y = train['RUL_class']
            
            # Train-test split
            test_size = st.slider("Test Size", 0.1, 0.4, 0.3, key='class_test')
            random_state = st.number_input("Random State", value=42, key='class_random')
            use_scaling = st.checkbox("Use Feature Scaling", value=True, key='class_scale')
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Scaling
            if use_scaling:
                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values
            
            st.subheader("🚀 Train Classification Models")
            
            class_models_to_train = st.multiselect(
                "Select Models to Train",
                ["SVC", "Random Forest", "Naive Bayes", "KNN"],
                default=["SVC", "Random Forest", "Naive Bayes", "KNN"]
            )
            
            if st.button("Train Classification Models"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                class_models = {}
                class_results = {}
                
                for idx, model_name in enumerate(class_models_to_train):
                    status_text.text(f"Training {model_name}...")
                    progress_bar.progress((idx + 1) / len(class_models_to_train))
                    
                    try:
                        if model_name == "SVC":
                            model = SVC(kernel='rbf', random_state=42)
                            model.fit(X_train_scaled, y_train)
                            y_pred_train = model.predict(X_train_scaled)
                            y_pred_test = model.predict(X_test_scaled)
                        
                        elif model_name == "Random Forest":
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train_scaled, y_train)
                            y_pred_train = model.predict(X_train_scaled)
                            y_pred_test = model.predict(X_test_scaled)
                        
                        elif model_name == "Naive Bayes":
                            model = GaussianNB()
                            model.fit(X_train_scaled, y_train)
                            y_pred_train = model.predict(X_train_scaled)
                            y_pred_test = model.predict(X_test_scaled)
                        
                        elif model_name == "KNN":
                            model = KNeighborsClassifier(n_neighbors=5)
                            model.fit(X_train_scaled, y_train)
                            y_pred_train = model.predict(X_train_scaled)
                            y_pred_test = model.predict(X_test_scaled)
                        
                        # Evaluate
                        train_metrics = evaluate_classification(y_train, y_pred_train)
                        test_metrics = evaluate_classification(y_test, y_pred_test)
                        
                        class_models[model_name] = model
                        class_results[model_name] = {
                            'train': train_metrics,
                            'test': test_metrics,
                            'y_pred_test': y_pred_test,
                            'y_test': y_test
                        }
                        
                    except Exception as e:
                        st.error(f"Error training {model_name}: {str(e)}")
                
                st.session_state.classification_models = class_models
                st.session_state.classification_results = class_results
                status_text.text("✅ Training complete!")
                progress_bar.progress(1.0)
                st.success("All classification models trained successfully!")
            
            # Display Results
            if 'classification_results' in st.session_state:
                st.subheader("📊 Classification Performance")
                
                results = st.session_state.classification_results
                
                # Metrics Table
                metrics_data = []
                for model_name, result in results.items():
                    metrics_data.append({
                        'Model': model_name,
                        'Train Accuracy': f"{result['train']['Accuracy']:.3f}",
                        'Test Accuracy': f"{result['test']['Accuracy']:.3f}",
                        'Test Precision': f"{result['test']['Precision']:.3f}",
                        'Test Recall': f"{result['test']['Recall']:.3f}",
                        'Test F1-Score': f"{result['test']['F1-Score']:.3f}"
                    })
                
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                
                # Confusion Matrix
                st.subheader("📊 Confusion Matrices")
                
                selected_model = st.selectbox("Select Model for Confusion Matrix", 
                                             list(results.keys()), key='conf_matrix')
                
                if selected_model:
                    result = results[selected_model]
                    y_test = result['y_test']
                    y_pred = result['y_pred_test']
                    
                    cm = confusion_matrix(y_test, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'{selected_model}: Confusion Matrix')
                    st.pyplot(fig)
                    
                    # Classification Report
                    st.write("**Classification Report:**")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
        else:
            st.info("Please convert RUL to classes first using the button above.")

# Model Comparison Page
elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")
    
    has_regression = 'regression_results' in st.session_state
    has_classification = 'classification_results' in st.session_state
    
    if not has_regression and not has_classification:
        st.warning("⚠️ No models trained yet. Please train models first.")
    else:
        if has_regression:
            st.subheader("🤖 Regression Models Comparison")
            reg_results = st.session_state.regression_results
            
            # Comparison Chart
            models = list(reg_results.keys())
            test_rmse = [reg_results[m]['test']['RMSE'] for m in models]
            test_r2 = [reg_results[m]['test']['R² Score'] for m in models]
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            axes[0].bar(models, test_rmse, color='skyblue', edgecolor='black')
            axes[0].set_ylabel('RMSE')
            axes[0].set_title('Test RMSE Comparison')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            axes[1].bar(models, test_r2, color='lightcoral', edgecolor='black')
            axes[1].set_ylabel('R² Score')
            axes[1].set_title('Test R² Score Comparison')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best Model
            best_rmse_model = min(models, key=lambda m: reg_results[m]['test']['RMSE'])
            best_r2_model = max(models, key=lambda m: reg_results[m]['test']['R² Score'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best RMSE Model", best_rmse_model, 
                         f"{reg_results[best_rmse_model]['test']['RMSE']:.2f}")
            with col2:
                st.metric("Best R² Model", best_r2_model,
                         f"{reg_results[best_r2_model]['test']['R² Score']:.3f}")
        
        if has_classification:
            st.subheader("📈 Classification Models Comparison")
            class_results = st.session_state.classification_results
            
            # Comparison Chart
            models = list(class_results.keys())
            test_accuracy = [class_results[m]['test']['Accuracy'] for m in models]
            test_f1 = [class_results[m]['test']['F1-Score'] for m in models]
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            axes[0].bar(models, test_accuracy, color='lightgreen', edgecolor='black')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('Test Accuracy Comparison')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].grid(True, alpha=0.3, axis='y')
            
            axes[1].bar(models, test_f1, color='orange', edgecolor='black')
            axes[1].set_ylabel('F1-Score')
            axes[1].set_title('Test F1-Score Comparison')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best Model
            best_acc_model = max(models, key=lambda m: class_results[m]['test']['Accuracy'])
            best_f1_model = max(models, key=lambda m: class_results[m]['test']['F1-Score'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Accuracy Model", best_acc_model,
                         f"{class_results[best_acc_model]['test']['Accuracy']:.3f}")
            with col2:
                st.metric("Best F1-Score Model", best_f1_model,
                         f"{class_results[best_f1_model]['test']['F1-Score']:.3f}")

# Predictions Page
elif page == "🔮 Predictions":
    st.title("🔮 Make Predictions")
    
    has_regression = 'regression_results' in st.session_state
    has_classification = 'classification_results' in st.session_state
    
    if not has_regression and not has_classification:
        st.warning("⚠️ No models trained yet. Please train models first.")
    else:
        prediction_type = st.radio("Prediction Type", ["Regression (RUL)", "Classification (RUL Class)"])
        
        if prediction_type == "Regression (RUL)" and has_regression:
            st.subheader("🤖 Regression Prediction")
            
            # Check if using pickled models
            if st.session_state.pickled_models_loaded and 'pickled_models' in st.session_state and st.session_state.pickled_models:
                model_options = {}
                model_display_names = {
                    'linear_regression': 'Linear Regression',
                    'svr': 'SVR',
                    'random_forest': 'Random Forest',
                    'xgboost': 'XGBoost'
                }
                
                for key, model in st.session_state.pickled_models.items():
                    model_options[model_display_names.get(key, key)] = (key, model)
                
                selected_display = st.selectbox("Select Model", list(model_options.keys()))
                selected_key, model = model_options[selected_display]
                scaler = st.session_state.scaler
                feature_info = st.session_state.feature_info
            else:
                selected_model = st.selectbox("Select Model", 
                                             list(st.session_state.models_trained.keys()))
                model = st.session_state.models_trained[selected_model]
                scaler = st.session_state.scaler
                feature_info = None
            
            # Input method
            input_method = st.radio("Input Method", ["Manual Entry", "Upload CSV", "Use Sample"])
            
            if input_method == "Manual Entry":
                st.write("**Enter Sensor Values:**")
                
                # Use feature info if available (from pickled models)
                if feature_info and 'drop_sensors' in feature_info:
                    drop_sensors = feature_info['drop_sensors']
                    st.info(f"Note: Sensors {drop_sensors} are excluded (as per trained model)")
                else:
                    drop_sensors = []
                
                col1, col2, col3 = st.columns(3)
                
                settings = {}
                sensors = {}
                
                with col1:
                    settings['setting_1'] = st.number_input("Setting 1", value=0.0)
                    settings['setting_2'] = st.number_input("Setting 2", value=0.0)
                    settings['setting_3'] = st.number_input("Setting 3", value=0.0)
                
                with col2:
                    for i in range(1, 11):
                        if f's_{i}' not in drop_sensors:
                            sensors[f's_{i}'] = st.number_input(f"Sensor {i}", value=0.0, key=f's{i}')
                
                with col3:
                    for i in range(11, 21):
                        if f's_{i}' not in drop_sensors:
                            sensors[f's_{i}'] = st.number_input(f"Sensor {i}", value=0.0, key=f's{i}')
                
                if st.button("Predict RUL"):
                    # Create feature vector based on model's expected features
                    if feature_info and 'feature_names' in feature_info:
                        # Use the exact feature order from the trained model
                        feature_order = feature_info['feature_names']
                        features = []
                        for feat in feature_order:
                            if feat in settings:
                                features.append(settings[feat])
                            elif feat in sensors:
                                features.append(sensors[feat])
                            else:
                                features.append(0.0)
                    else:
                        # Default feature order
                        feature_order = ['setting_1', 'setting_2', 'setting_3'] + \
                                       [f's_{i}' for i in range(1, 21) if f's_{i}' not in drop_sensors]
                        features = [settings.get(f, 0.0) for f in feature_order[:3]] + \
                                  [sensors.get(f, 0.0) for f in feature_order[3:]]
                    
                    X_input = np.array(features).reshape(1, -1)
                    
                    if scaler:
                        X_input = scaler.transform(X_input)
                    
                    prediction = model.predict(X_input)[0]
                    
                    st.success(f"**Predicted RUL: {prediction:.2f} cycles**")
                    st.info(f"Engine has approximately {prediction:.0f} cycles remaining before failure.")
            
            elif input_method == "Upload CSV":
                uploaded_file = st.file_uploader("Upload CSV file with features", type=['csv'])
                
                if uploaded_file:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Uploaded Data:**")
                    st.dataframe(df)
                    
                    if st.button("Predict"):
                        # Prepare features
                        X_input = df.copy()

                        # Align features to model expectations if we have feature_info
                        if feature_info and feature_info.get('feature_names'):
                            expected = feature_info['feature_names']
                            # Drop any unexpected columns, keep order
                            X_input = X_input[[c for c in expected if c in X_input.columns]]
                            # Add missing expected columns as zeros
                            for col in expected:
                                if col not in X_input.columns:
                                    X_input[col] = 0.0
                            X_input = X_input[expected]
                        else:
                            # Default: drop unit/time labels if present
                            X_input = X_input.drop(columns=[c for c in ['unit_number', 'time_cycles'] if c in X_input.columns], errors='ignore')
                        
                        X_input = X_input.values

                        if scaler:
                            X_input = scaler.transform(X_input)
                        
                        predictions = model.predict(X_input)
                        df['Predicted_RUL'] = predictions
                        
                        st.write("**Predictions:**")
                        st.dataframe(df)
                        
                        st.download_button(
                            "Download Predictions",
                            df.to_csv(index=False),
                            "predictions.csv",
                            "text/csv"
                        )
            
            elif input_method == "Use Sample":
                if st.session_state.data_loaded:
                    valid = st.session_state.valid_data.copy()
                    sample_idx = st.number_input("Sample Index", 0, len(valid)-1, 0)
                    
                    sample = valid.iloc[sample_idx:sample_idx+1]
                    st.write("**Sample Data:**")
                    st.dataframe(sample)
                    
                    if st.button("Predict"):
                        X_sample = sample.copy()

                        # Align features to model expectations if we have feature_info
                        if feature_info and feature_info.get('feature_names'):
                            expected = feature_info['feature_names']
                            # Drop labels and any unexpected columns, keep order
                            X_sample = X_sample.drop(columns=[c for c in ['unit_number', 'time_cycles', 'RUL', 'RUL_class'] if c in X_sample.columns], errors='ignore')
                            X_sample = X_sample[[c for c in expected if c in X_sample.columns]]
                            # Add missing expected columns as zeros
                            for col in expected:
                                if col not in X_sample.columns:
                                    X_sample[col] = 0.0
                            X_sample = X_sample[expected]
                        else:
                            # Default: drop labels only
                            X_sample, _ = prepare_features(sample, 
                                                           drop_labels=['unit_number', 'time_cycles'])
                        X_sample = X_sample.values
                        
                        if scaler:
                            X_sample = scaler.transform(X_sample)
                        
                        prediction = model.predict(X_sample)[0]
                        actual_rul = st.session_state.y_valid.iloc[sample_idx]['RUL'] if sample_idx < len(st.session_state.y_valid) else None
                        
                        st.success(f"**Predicted RUL: {prediction:.2f} cycles**")
                        if actual_rul is not None:
                            st.info(f"Actual RUL: {actual_rul} cycles")
                            error = abs(prediction - actual_rul)
                            st.metric("Prediction Error", f"{error:.2f} cycles")
        
        elif prediction_type == "Classification (RUL Class)" and has_classification:
            st.subheader("📈 Classification Prediction")
            st.info("Classification predictions coming soon...")

# Performance Analysis Page
elif page == "📈 Performance Analysis":
    st.title("📈 Performance Analysis")
    
    has_regression = 'regression_results' in st.session_state
    has_classification = 'classification_results' in st.session_state
    
    if not has_regression and not has_classification:
        st.warning("⚠️ No models trained yet. Please train models first.")
    else:
        if has_regression:
            st.subheader("🤖 Regression Model Analysis")
            
            selected_model = st.selectbox("Select Model", 
                                         list(st.session_state.regression_results.keys()),
                                         key='perf_reg')
            
            result = st.session_state.regression_results[selected_model]
            
            # Metrics comparison
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train RMSE", f"{result['train']['RMSE']:.2f}")
                st.metric("Test RMSE", f"{result['test']['RMSE']:.2f}")
            with col2:
                st.metric("Train R²", f"{result['train']['R² Score']:.3f}")
                st.metric("Test R²", f"{result['test']['R² Score']:.3f}")
            with col3:
                st.metric("Train MAE", f"{result['train']['MAE']:.2f}")
                st.metric("Test MAE", f"{result['test']['MAE']:.2f}")
            
            # Error distribution
            y_test = result['y_test']
            y_pred = result['y_pred_test']
            errors = y_test - y_pred
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
            axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
            axes[0].set_xlabel('Prediction Error')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Error Distribution')
            axes[0].grid(True, alpha=0.3)
            
            axes[1].boxplot([errors], labels=['Errors'])
            axes[1].set_ylabel('Prediction Error')
            axes[1].set_title('Error Box Plot')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        if has_classification:
            st.subheader("📈 Classification Model Analysis")
            
            selected_model = st.selectbox("Select Model",
                                         list(st.session_state.classification_results.keys()),
                                         key='perf_class')
            
            result = st.session_state.classification_results[selected_model]
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train Accuracy", f"{result['train']['Accuracy']:.3f}")
                st.metric("Test Accuracy", f"{result['test']['Accuracy']:.3f}")
            with col2:
                st.metric("Train Precision", f"{result['train']['Precision']:.3f}")
                st.metric("Test Precision", f"{result['test']['Precision']:.3f}")
            with col3:
                st.metric("Train Recall", f"{result['train']['Recall']:.3f}")
                st.metric("Test Recall", f"{result['test']['Recall']:.3f}")
            with col4:
                st.metric("Train F1-Score", f"{result['train']['F1-Score']:.3f}")
                st.metric("Test F1-Score", f"{result['test']['F1-Score']:.3f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About
**NASA Predictive Maintenance System**

Built with Streamlit for Remaining Useful Life (RUL) prediction of turbofan engines.

**Dataset:** NASA C-MAPSS FD001
            
**GitHub:** [YashJoshi2109/NASA-Turbofan-Engine-Degradation-Simulation](https://github.com/YashJoshi2109/NASA-Turbofan-Engine-Degradation-Simulation)

**Kaggle:** [NASA C-MAPSS Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

**Made by:** Yash Joshi
""")
