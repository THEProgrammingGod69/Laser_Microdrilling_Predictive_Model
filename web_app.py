import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from llm import LaserMicrodrillingModel
import os
import plotly.express as px
from io import StringIO
import torch
import asyncio
import nest_asyncio

# Enable nested event loops
nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="Laser Microdrilling Predictor",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* Global Theme */
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    
    /* Header Styling */
    h1 {
        color: #1e3d59;
        font-weight: 600;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #17a2b8;
    }
    
    h2 {
        color: #2c5282;
        font-weight: 500;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #2d3748;
        font-weight: 500;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #17a2b8;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #138496;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Container Styling */
    .prediction-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .guide-box {
        background-color: white;
        border-left: 4px solid #17a2b8;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .chatbot-box {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f1f5f9;
    }
    
    /* Status Indicators */
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    /* Card Layout */
    .card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    
    /* Image Container */
    .hero-image-container {
        display: flex;
        justify-content: center;
        margin: 2rem 0;
        padding: 1rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
    }
    
    .hero-image {
        max-width: 600px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Feature List */
    .feature-list {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        margin: 0.5rem 0;
        padding: 0.5rem;
        background-color: #f8f9fa;
        border-radius: 4px;
    }
    
    .feature-icon {
        margin-right: 1rem;
        color: #17a2b8;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    try:
        st.session_state.model = LaserMicrodrillingModel()
        if os.path.exists('laser_models'):
            st.session_state.model.load_models()
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.session_state.model = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def create_3d_surface_plot(model, speed_range, freq_range, prediction_type='rf'):
    try:
        speeds = np.linspace(speed_range[0], speed_range[1], 50)
        freqs = np.linspace(freq_range[0], freq_range[1], 50)
        speed_mesh, freq_mesh = np.meshgrid(speeds, freqs)
        
        predictions = np.zeros_like(speed_mesh)
        for i in range(len(speeds)):
            for j in range(len(freqs)):
                predictions[j, i] = model.predict_diameter(
                    speed_mesh[j, i], 
                    freq_mesh[j, i], 
                    prediction_type
                )
        
        fig = go.Figure(data=[go.Surface(
            x=speed_mesh,
            y=freq_mesh,
            z=predictions,
            colorscale='viridis'
        )])
        
        fig.update_layout(
            title=f'{"Random Forest" if prediction_type=="rf" else "Neural Network"} Predictions',
            scene=dict(
                xaxis_title='Speed',
                yaxis_title='Frequency',
                zaxis_title='Predicted Diameter'
            ),
            width=600,
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

def show_beginner_guide():
    st.markdown("""
    <div class="guide-box">
    <h3>📚 Beginner's Guide to Laser Microdrilling Predictor</h3>
    
    <h4>What is this tool?</h4>
    This is an advanced predictive tool that helps you:
    - Predict the diameter of laser-drilled holes
    - Optimize laser parameters (speed and frequency)
    - Analyze trends and patterns in laser microdrilling
    
    <h4>How to Use This Tool:</h4>
    
    1️⃣ <b>Training the Model</b>
    - Go to "Train Model" in the sidebar
    - Upload your Excel file (EDI_OBSERVATIONS.xlsx)
    - Select the correct columns for Speed, Frequency, and Diameter
    - Click "Train Models" and wait for completion
    
    2️⃣ <b>Making Predictions</b>
    - Go to "Make Predictions"
    - Enter Speed and Frequency values
    - Click "Predict" to see the estimated hole diameter
    
    3️⃣ <b>Batch Analysis</b>
    - Use this for multiple predictions at once
    - Set ranges for Speed and Frequency
    - Get predictions for multiple combinations
    
    4️⃣ <b>Visualization</b>
    - View 3D plots of predictions
    - Compare Random Forest and Neural Network results
    - Analyze trends across different parameters
    
    5️⃣ <b>General Queries</b>
    - Use the chatbot for general questions
    - Ask about laser microdrilling concepts
    - Get help with parameter selection
    
    <h4>Tips for Best Results:</h4>
    - Use data within the trained range
    - Compare both model predictions
    - Start with small parameter ranges
    - Use the chatbot for guidance
    
    <h4>Need Help?</h4>
    - Use the chatbot below
    - Check the "About" page
    - Look for warning messages
    - Contact support if needed
    </div>
    """, unsafe_allow_html=True)

def show_chatbot_interface():
    st.markdown("""
    <div class="chatbot-box">
    <h3>🤖 AI Assistant for Laser Microdrilling</h3>
    <p>Ask me anything about laser microdrilling, parameter selection, or get help with using this tool!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.write('You: ' + message['content'])
        else:
            st.write('AI: ' + message['content'])
    
    # Chat input
    user_input = st.text_input("Ask a question:")
    if user_input:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        try:
            # Get response from the model
            response = st.session_state.model.process_general_query(user_input)
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            
            # Show the response
            st.write('AI: ' + response)
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

def main():
    st.title("🔧 Laser Microdrilling Predictor")
    
    # Sidebar navigation with direct page updates
    pages = ["Home", "Beginner's Guide", "Train Model", "Make Predictions", "Batch Analysis", "Visualization", "General Queries", "About"]
    
    # Initialize the session state for navigation if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Sidebar navigation
    selected_page = st.sidebar.radio(
        "Choose a page",
        pages,
        index=pages.index(st.session_state.current_page)
    )
    
    # Update current page
    st.session_state.current_page = selected_page

    if selected_page == "Home":
        st.markdown("""
            <div class="card">
                <h2>Welcome to Laser Microdrilling Predictor</h2>
                <p>This advanced application combines machine learning and expert knowledge to revolutionize your laser microdrilling process.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Feature List
        st.markdown("""
            <div class="feature-list">
                <h3>Key Features</h3>
                <div class="feature-item">
                    <span class="feature-icon">🎯</span>
                    <span>Predict and optimize laser microdrilling parameters</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📊</span>
                    <span>Make accurate hole diameter predictions</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📈</span>
                    <span>Analyze trends and patterns</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">🤖</span>
                    <span>Get expert guidance through AI</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Model Status Card
        st.markdown("""
            <div class="card">
                <h3>Model Status</h3>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.model and st.session_state.model.rf_model is not None:
            st.markdown("""
                <div class="success-box">
                    ✅ Models are loaded and ready!
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="warning-box">
                    ⚠️ No models loaded. Please train or load models first.
                </div>
                """, unsafe_allow_html=True)
        
        # Quick Start Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 View Beginner's Guide", key="guide_button"):
                st.session_state.current_page = "Beginner's Guide"
                st.rerun()
        with col2:
            if st.button("🤖 Ask AI Assistant", key="assistant_button"):
                st.session_state.current_page = "General Queries"
                st.rerun()

        # Hero Image at the bottom
        st.markdown("<br><br>", unsafe_allow_html=True)  # Add some spacing
        try:
            st.image("assets/laser_microdrilling.svg", 
                    caption="Laser Microdrilling Process Illustration",
                    use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
    
    elif selected_page == "Beginner's Guide":
        show_beginner_guide()
        
    elif selected_page == "General Queries":
        show_chatbot_interface()
        
    elif selected_page == "Train Model":
        st.header("Train New Model")
        
        uploaded_file = st.file_uploader("Upload your Excel dataset", type=['xlsx'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_excel(uploaded_file)
                st.write("Preview of your data:")
                st.dataframe(df.head())
                
                cols = df.columns.tolist()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    speed_col = st.selectbox("Select Speed column", cols)
                with col2:
                    freq_col = st.selectbox("Select Frequency column", cols)
                with col3:
                    diam_col = st.selectbox("Select Diameter column", cols)
                
                if st.button("Train Models"):
                    with st.spinner("Training models..."):
                        try:
                            if st.session_state.model is None:
                                st.session_state.model = LaserMicrodrillingModel()
                            st.session_state.model.load_data(uploaded_file, speed_col, freq_col, diam_col)
                            st.session_state.model.train_models()
                            st.session_state.model.evaluate_models()
                            st.session_state.model.save_models()
                            st.success("Models trained successfully!")
                            
                            # Show performance metrics
                            st.subheader("Model Performance")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Random Forest Metrics")
                                rf_pred = st.session_state.model.rf_model.predict(st.session_state.model.X_test_scaled)
                                st.metric("R² Score", f"{st.session_state.model.r2_score(st.session_state.model.y_test, rf_pred):.4f}")
                            with col2:
                                st.write("Neural Network Metrics")
                                with torch.no_grad():
                                    nn_pred = st.session_state.model.nn_model(torch.FloatTensor(st.session_state.model.X_test_scaled.values)).numpy()
                                st.metric("R² Score", f"{st.session_state.model.r2_score(st.session_state.model.y_test, nn_pred):.4f}")
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                    
    elif selected_page == "Make Predictions":
        st.header("Make Predictions")
        
        if st.session_state.model is None or st.session_state.model.rf_model is None:
            st.warning("Please train or load models first!")
        else:
            try:
                col1, col2 = st.columns(2)
                with col1:
                    speed = st.number_input("Enter Speed", min_value=0.0)
                with col2:
                    freq = st.number_input("Enter Frequency", min_value=0.0)
                
                if st.button("Predict"):
                    try:
                        rf_pred = st.session_state.model.predict_diameter(speed, freq, 'rf')
                        nn_pred = st.session_state.model.predict_diameter(speed, freq, 'nn')
                        avg_pred = (rf_pred + nn_pred) / 2
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Random Forest Prediction", f"{rf_pred:.4f}")
                        with col2:
                            st.metric("Neural Network Prediction", f"{nn_pred:.4f}")
                        with col3:
                            st.metric("Average Prediction", f"{avg_pred:.4f}")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            except Exception as e:
                st.error(f"Error in prediction interface: {str(e)}")
                    
    elif selected_page == "Batch Analysis":
        st.header("Batch Analysis")
        
        if st.session_state.model is None or st.session_state.model.rf_model is None:
            st.warning("Please train or load models first!")
        else:
            try:
                col1, col2 = st.columns(2)
                with col1:
                    speed_min = st.number_input("Minimum Speed", min_value=0.0)
                    speed_max = st.number_input("Maximum Speed", min_value=speed_min)
                with col2:
                    freq_min = st.number_input("Minimum Frequency", min_value=0.0)
                    freq_max = st.number_input("Maximum Frequency", min_value=freq_min)
                
                steps = st.slider("Number of points per parameter", min_value=2, max_value=10, value=5)
                
                if st.button("Generate Predictions"):
                    try:
                        speeds = np.linspace(speed_min, speed_max, steps)
                        freqs = np.linspace(freq_min, freq_max, steps)
                        
                        results = []
                        for speed in speeds:
                            for freq in freqs:
                                rf_pred = st.session_state.model.predict_diameter(speed, freq, 'rf')
                                nn_pred = st.session_state.model.predict_diameter(speed, freq, 'nn')
                                avg_pred = (rf_pred + nn_pred) / 2
                                results.append({
                                    'Speed': speed,
                                    'Frequency': freq,
                                    'RF Prediction': rf_pred,
                                    'NN Prediction': nn_pred,
                                    'Average': avg_pred
                                })
                        
                        results_df = pd.DataFrame(results)
                        st.write("Prediction Results:")
                        st.dataframe(results_df)
                        
                        # Download button for results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download Results",
                            csv,
                            "predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
            except Exception as e:
                st.error(f"Error in batch analysis interface: {str(e)}")
                
    elif selected_page == "Visualization":
        st.header("Prediction Visualization")
        
        if st.session_state.model is None or st.session_state.model.rf_model is None:
            st.warning("Please train or load models first!")
        else:
            try:
                if hasattr(st.session_state.model, 'X') and st.session_state.model.X is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        speed_range = st.slider(
                            "Speed Range",
                            float(st.session_state.model.X[st.session_state.model.feature_names[0]].min()),
                            float(st.session_state.model.X[st.session_state.model.feature_names[0]].max()),
                            (float(st.session_state.model.X[st.session_state.model.feature_names[0]].min()),
                             float(st.session_state.model.X[st.session_state.model.feature_names[0]].max()))
                        )
                    with col2:
                        freq_range = st.slider(
                            "Frequency Range",
                            float(st.session_state.model.X[st.session_state.model.feature_names[1]].min()),
                            float(st.session_state.model.X[st.session_state.model.feature_names[1]].max()),
                            (float(st.session_state.model.X[st.session_state.model.feature_names[1]].min()),
                             float(st.session_state.model.X[st.session_state.model.feature_names[1]].max()))
                        )
                    
                    tab1, tab2 = st.tabs(["Random Forest", "Neural Network"])
                    
                    with tab1:
                        rf_fig = create_3d_surface_plot(st.session_state.model, speed_range, freq_range, 'rf')
                        if rf_fig:
                            st.plotly_chart(rf_fig)
                    
                    with tab2:
                        nn_fig = create_3d_surface_plot(st.session_state.model, speed_range, freq_range, 'nn')
                        if nn_fig:
                            st.plotly_chart(nn_fig)
                else:
                    st.warning("Please train models with data first to enable visualization.")
            except Exception as e:
                st.error(f"Error in visualization: {str(e)}")
                
    elif selected_page == "About":
        st.header("About")
        st.markdown("""
        ### 🔬 Project Overview
        The Laser Microdrilling Predictor is an advanced application that combines cutting-edge machine learning with expert knowledge in laser processing. This tool is designed to revolutionize the way engineers and researchers approach laser microdrilling operations.

        ### 🎯 Our Goal
        Our primary objective is to optimize laser microdrilling processes by:
        - Reducing the time and resources spent on parameter optimization
        - Improving hole quality and consistency
        - Minimizing the need for trial-and-error experiments
        - Providing data-driven insights for process improvement

        ### 🤖 Machine Learning Models
        We utilize two sophisticated models for predictions:

        #### 1. Random Forest Model
        - Excellent at handling non-linear relationships
        - Robust against outliers
        - Provides feature importance insights
        - Ideal for understanding parameter interactions

        #### 2. Neural Network Model
        - Captures complex patterns in the data
        - Learns hierarchical feature representations
        - Adapts to new patterns through training
        - Optimized for high-dimensional parameter spaces

        ### 📊 The Process
        1. **Data Collection**
           - Gathering experimental data from laser microdrilling operations
           - Recording key parameters: speed, frequency, and resulting hole diameter
           - Validating data quality and consistency

        2. **Model Training**
           - Preprocessing and normalizing input data
           - Training both Random Forest and Neural Network models
           - Cross-validation for robust performance
           - Model evaluation and optimization

        3. **Prediction System**
           - Real-time parameter prediction
           - Batch analysis capabilities
           - Interactive visualization tools
           - Uncertainty estimation

        4. **Quality Assurance**
           - Continuous model validation
           - Regular performance monitoring
           - Feedback incorporation
           - Periodic model updates

        ### 💡 Applications
        - Aerospace component manufacturing
        - Medical device fabrication
        - Electronics industry
        - Research and development
        - Process optimization
        - Quality control

        ### 🔄 Continuous Improvement
        We are committed to:
        - Regular model updates based on new data
        - Feature additions based on user feedback
        - Performance optimization
        - Enhanced visualization capabilities
        - Expanded parameter range support

       
        """)

if __name__ == "__main__":
    main() 