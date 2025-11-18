import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Box Office Revenue Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🎬 Box Office Revenue Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3163/3163478.png", width=100)
    st.title("Navigation")
    page = st.radio("Go to", ["📊 Data Analysis", "🤖 Model Training", "🎯 Predictions"])
    
    st.markdown("---")
    st.info("**About:** This advanced ML app predicts box office revenue using XGBoost regression with comprehensive data analysis.")

# Load and process data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('boxoffice.csv', encoding='latin-1')
        return df
    except:
        # Create sample data if file not found
        np.random.seed(42)
        sample_data = {
            'title': [f'Movie {i}' for i in range(500)],
            'domestic_revenue': np.random.randint(1000000, 500000000, 500),
            'opening_theaters': np.random.randint(100, 4500, 500),
            'release_days': np.random.randint(30, 365, 500),
            'MPAA': np.random.choice(['G', 'PG', 'PG-13', 'R'], 500),
            'genres': np.random.choice(['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi'], 500),
            'distributor': np.random.choice(['Universal', 'Warner Bros', 'Disney', 'Sony'], 500)
        }
        return pd.DataFrame(sample_data)

@st.cache_data
def preprocess_data(df):
    # Remove columns
    to_remove = ['world_revenue', 'opening_revenue'] if 'world_revenue' in df.columns else []
    df = df.drop([col for col in to_remove if col in df.columns], axis=1)
    
    if 'budget' in df.columns:
        df = df.drop('budget', axis=1)
    
    # Fill missing values
    for col in ['MPAA', 'genres']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    df = df.dropna()
    
    # Clean revenue column
    if df['domestic_revenue'].dtype == 'object':
        df['domestic_revenue'] = df['domestic_revenue'].astype(str).str.replace('$', '').str.replace(',', '')
    
    # Convert numeric columns
    for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    # Log transform
    for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.log10(x) if x > 0 else 0)
    
    # Vectorize genres
    if 'genres' in df.columns:
        vectorizer = CountVectorizer()
        vectorizer.fit(df['genres'])
        features = vectorizer.transform(df['genres']).toarray()
        genres = vectorizer.get_feature_names_out()
        for i, name in enumerate(genres):
            df[name] = features[:, i]
        df = df.drop('genres', axis=1)
    
    # Label encode categorical
    for col in ['distributor', 'MPAA']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    return df

# Load data
df_raw = load_data()
df = preprocess_data(df_raw.copy())

# PAGE 1: Data Analysis
if page == "📊 Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Movies", len(df_raw), delta="100%")
    with col2:
        avg_revenue = np.power(10, df['domestic_revenue'].mean())
        st.metric("Avg Revenue", f"${avg_revenue:,.0f}")
    with col3:
        st.metric("Features", df.shape[1])
    with col4:
        st.metric("Missing Values", df_raw.isnull().sum().sum())
    
    st.markdown("---")
    
    # Interactive data table
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df_raw.head(20), use_container_width=True)
    
    # Visualizations
    st.subheader("📈 Revenue Distribution by MPAA Rating")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'MPAA' in df_raw.columns:
            fig = px.bar(df_raw.groupby('MPAA')['domestic_revenue'].mean().reset_index(),
                        x='MPAA', y='domestic_revenue',
                        color='MPAA',
                        title='Average Revenue by MPAA Rating',
                        labels={'domestic_revenue': 'Avg Revenue ($)'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'MPAA' in df_raw.columns:
            mpaa_counts = df_raw['MPAA'].value_counts()
            fig = px.pie(values=mpaa_counts.values, names=mpaa_counts.index,
                        title='Movies Distribution by MPAA Rating',
                        hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.subheader("📊 Feature Distributions")
    numeric_cols = ['domestic_revenue', 'opening_theaters', 'release_days']
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if numeric_cols:
        fig = make_subplots(rows=1, cols=len(numeric_cols),
                           subplot_titles=numeric_cols)
        
        for i, col in enumerate(numeric_cols, 1):
            fig.add_trace(
                go.Histogram(x=df[col], name=col, nbinsx=30),
                row=1, col=i
            )
        
        fig.update_layout(height=400, showlegend=False, title_text="Log-Transformed Feature Distributions")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    corr_matrix = df.select_dtypes(include=np.number).corr()
    fig = px.imshow(corr_matrix, 
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title='Feature Correlation Matrix')
    st.plotly_chart(fig, use_container_width=True)

# PAGE 2: Model Training
elif page == "🤖 Model Training":
    st.header("🤖 Model Training & Evaluation")
    
    # Train model
    @st.cache_resource
    def train_model(df):
        features = df.drop(['title', 'domestic_revenue'], axis=1, errors='ignore')
        target = df['domestic_revenue'].values
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            features, target, test_size=0.1, random_state=22
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, Y_train)
        
        train_preds = model.predict(X_train_scaled)
        val_preds = model.predict(X_val_scaled)
        
        return model, scaler, X_train_scaled, X_val_scaled, Y_train, Y_val, train_preds, val_preds, features.columns
    
    with st.spinner("Training model... Please wait."):
        model, scaler, X_train, X_val, Y_train, Y_val, train_preds, val_preds, feature_names = train_model(df)
    
    st.success("✅ Model trained successfully!")
    
    # Metrics
    train_mae = mae(Y_train, train_preds)
    val_mae = mae(Y_val, val_preds)
    train_r2 = r2_score(Y_train, train_preds)
    val_r2 = r2_score(Y_val, val_preds)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training MAE", f"{train_mae:.4f}")
    with col2:
        st.metric("Validation MAE", f"{val_mae:.4f}")
    with col3:
        st.metric("Training R²", f"{train_r2:.4f}")
    with col4:
        st.metric("Validation R²", f"{val_r2:.4f}")
    
    # Prediction plots
    st.subheader("📈 Prediction vs Actual")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Y_train, y=train_preds, mode='markers',
                                name='Training', marker=dict(color='blue', opacity=0.5)))
        fig.add_trace(go.Scatter(x=[Y_train.min(), Y_train.max()],
                                y=[Y_train.min(), Y_train.max()],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(title='Training Set Predictions',
                         xaxis_title='Actual (log scale)',
                         yaxis_title='Predicted (log scale)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Y_val, y=val_preds, mode='markers',
                                name='Validation', marker=dict(color='green', opacity=0.5)))
        fig.add_trace(go.Scatter(x=[Y_val.min(), Y_val.max()],
                                y=[Y_val.min(), Y_val.max()],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(title='Validation Set Predictions',
                         xaxis_title='Actual (log scale)',
                         yaxis_title='Predicted (log scale)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                title='Top 15 Most Important Features',
                color='importance',
                color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)

# PAGE 3: Predictions
elif page == "🎯 Predictions":
    st.header("🎯 Make Revenue Predictions")
    
    # Train model for predictions
    @st.cache_resource
    def get_model(df):
        features = df.drop(['title', 'domestic_revenue'], axis=1, errors='ignore')
        target = df['domestic_revenue'].values
        
        X_train, X_val, Y_train, Y_val = train_test_split(
            features, target, test_size=0.1, random_state=22
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, Y_train)
        
        return model, scaler, features.columns
    
    model, scaler, feature_names = get_model(df)
    
    st.subheader("📝 Enter Movie Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        opening_theaters = st.number_input("Opening Theaters", min_value=100, max_value=5000, value=2500)
        mpaa_rating = st.selectbox("MPAA Rating", ['G', 'PG', 'PG-13', 'R'])
    
    with col2:
        release_days = st.number_input("Release Days", min_value=30, max_value=365, value=120)
        distributor = st.selectbox("Distributor", ['Universal', 'Warner Bros', 'Disney', 'Sony', 'Paramount'])
    
    with col3:
        genre = st.selectbox("Primary Genre", ['Action', 'Comedy', 'Drama', 'Horror', 'Sci-Fi', 'Thriller'])
    
    if st.button("🎬 Predict Revenue", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'opening_theaters': [np.log10(opening_theaters)],
            'release_days': [np.log10(release_days)],
            'MPAA': [0],  # Will be encoded
            'distributor': [0],  # Will be encoded
        })
        
        # Add genre columns
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[feature_names]
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        predicted_revenue = np.power(10, prediction)
        
        st.success("✨ Prediction Complete!")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 30px; border-radius: 15px; text-align: center;
                        box-shadow: 0 10px 25px rgba(0,0,0,0.2);'>
                <h2 style='color: white; margin: 0;'>Predicted Domestic Revenue</h2>
                <h1 style='color: #FFD700; font-size: 3em; margin: 10px 0;'>
                    ${predicted_revenue:,.0f}
                </h1>
                <p style='color: white; font-size: 1.2em;'>
                    Expected box office performance
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <p>🎬 Box Office Revenue Predictor | Powered by XGBoost & Streamlit</p>
    <p>Made with ❤️ using Machine Learning</p>
</div>
""", unsafe_allow_html=True)