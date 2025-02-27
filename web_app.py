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

def create_3d_surface_plot(model, speed_range, freq_range, power_val, loop_count, output_type='diameter', prediction_type='rf'):
    try:
        speeds = np.linspace(speed_range[0], speed_range[1], 50)
        freqs = np.linspace(freq_range[0], freq_range[1], 50)
        speed_mesh, freq_mesh = np.meshgrid(speeds, freqs)
        
        predictions = np.zeros_like(speed_mesh)
        for i in range(len(speeds)):
            for j in range(len(freqs)):
                if prediction_type == 'rf':
                    diameter_pred, pitch_pred = model.predict(speed_mesh[j, i], freq_mesh[j, i], power_val, loop_count, 'rf')
                else:
                    diameter_pred, pitch_pred = model.predict(speed_mesh[j, i], freq_mesh[j, i], power_val, loop_count, 'nn')
                predictions[j, i] = diameter_pred if output_type == 'diameter' else pitch_pred
        
        fig = go.Figure(data=[go.Surface(
            x=speed_mesh,
            y=freq_mesh,
            z=predictions,
            colorscale='viridis'
        )])
        
        output_label = 'Diameter' if output_type == 'diameter' else 'Pitch'
        model_label = 'Random Forest' if prediction_type == 'rf' else 'Neural Network'
        
        fig.update_layout(
            title=f'{model_label} {output_label} Predictions (Power = {power_val}, Loop Count = {loop_count})',
            scene=dict(
                xaxis_title='Speed',
                yaxis_title='Frequency',
                zaxis_title=f'Predicted {output_label}'
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
    - Predict both dimple diameter and pitch in laser-drilled dimples
    - Optimize laser parameters (speed, frequency, power, and loop count)
    - Analyze trends and patterns in laser microdrilling
    
    <h4>How to Use This Tool:</h4>
    
    1️⃣ <b>Training the Model</b>
    - Go to "Train Model" in the sidebar
    - Upload your Excel file (EDI_OBSERVATIONS.xlsx)
    - Select the correct columns for Speed, Frequency, Power, Loop Count, Diameter, and Pitch
    - Click "Train Models" and wait for completion
    
    2️⃣ <b>Making Predictions</b>
    - Go to "Make Predictions"
    - Enter Speed, Frequency, Power, and Loop Count values
    - Click "Predict" to see the estimated dimple diameter and pitch
    
    3️⃣ <b>Batch Analysis</b>
    - Use this for multiple predictions at once
    - Set ranges for Speed, Frequency, Power, and Loop Count
    - Get predictions for multiple combinations
    
    4️⃣ <b>Visualization</b>
    - View 3D plots of predictions
    - Compare Random Forest and Neural Network results
    - Analyze trends across different parameters
    - Visualize both diameter and pitch predictions
    
    5️⃣ <b>General Queries</b>
    - Use the chatbot for general questions
    - Ask about laser microdrilling concepts
    - Get help with parameter selection
    
    <h4>Tips for Best Results:</h4>
    - Use data within the trained range
    - Compare both model predictions
    - Start with small parameter ranges
    - Consider the relationship between parameters
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
    <h4>Type "laser" to get started!</h4>
    <p>Welcome to the AI Assistant! You can ask me anything about laser microdrilling, parameter selection, or get help with using this tool.</p>
    <p>Here are some topics you might want to explore:</p>
    <ul>
        <li><b>Laser Parameters:</b> Inquire about the optimal settings for speed, frequency, power, and loop count.</li>
        <li><b>Model Predictions:</b> Ask how to interpret the predictions for dimple diameter and pitch.</li>
        <li><b>Training Models:</b> Get guidance on how to train the models with your data.</li>
        <li><b>Batch Analysis:</b> Learn how to perform batch predictions for multiple parameter combinations.</li>
        <li><b>Visualization:</b> Understand how to visualize the results and trends in your data.</li>
    </ul>
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
                    <span>Predict hole diameter and pitch based on laser parameters</span>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">📊</span>
                    <span>Optimize speed, frequency, and power settings</span>
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
        
        if st.session_state.model and st.session_state.model.rf_model_diameter is not None:
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

        # Add some spacing at the bottom
        st.markdown("<br><br>", unsafe_allow_html=True)
    
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
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    speed_col = st.selectbox("Select Speed column", cols)
                with col2:
                    freq_col = st.selectbox("Select Frequency column", cols)
                with col3:
                    power_col = st.selectbox("Select Power column", cols)
                with col4:
                    loop_col = st.selectbox("Select Loop Count column", cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    diam_col = st.selectbox("Select Diameter column", cols)
                with col2:
                    pitch_col = st.selectbox("Select Pitch column", cols)
                
                if st.button("Train Models"):
                    with st.spinner("Training models..."):
                        try:
                            if st.session_state.model is None:
                                st.session_state.model = LaserMicrodrillingModel()
                            st.session_state.model.load_data(df, speed_col, freq_col, power_col, diam_col, pitch_col)
                            st.session_state.model.train_models()
                            st.session_state.model.evaluate_models()
                            st.session_state.model.save_models()
                            st.success("Models trained successfully!")
                            
                            # Show performance metrics
                            st.subheader("Model Performance")
                            
                            # Diameter Models Performance
                            st.write("Diameter Prediction Models")
                            col1, col2 = st.columns(2)
                            with col2:
                                st.write("Neural Network Metrics")
                                with torch.no_grad():
                                    nn_pred_diameter = st.session_state.model.nn_model_diameter(torch.FloatTensor(st.session_state.model.X_test_scaled.values)).numpy()
                                st.metric("R² Score", f"{st.session_state.model.r2_score(st.session_state.model.y_test_diameter, nn_pred_diameter):.4f}")
                            
                            # Pitch Models Performance
                            st.write("Pitch Prediction Models")
                            col1, col2 = st.columns(2)
                            with col2:
                                st.write("Neural Network Metrics")
                                with torch.no_grad():
                                    nn_pred_pitch = st.session_state.model.nn_model_pitch(torch.FloatTensor(st.session_state.model.X_test_scaled.values)).numpy()
                                st.metric("R² Score", f"{st.session_state.model.r2_score(st.session_state.model.y_test_pitch, nn_pred_pitch):.4f}")
                        except Exception as e:
                            st.error(f"Error during training: {str(e)}")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                    
    elif selected_page == "Make Predictions":
        st.header("Make Predictions")
        
        if st.session_state.model is None or st.session_state.model.rf_model_diameter is None:
            st.warning("Please train or load models first!")
        else:
            try:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    speed = st.number_input("Enter Speed", min_value=0.0)
                with col2:
                    freq = st.number_input("Enter Frequency", min_value=0.0)
                with col3:
                    power = st.number_input("Enter Power", min_value=0.0)
                with col4:
                    loop_count = st.number_input("Enter Loop Count", min_value=1, step=1)
                
                if st.button("Predict"):
                    try:
                        # Remove loop_count from the predict calls
                        rf_diameter, rf_pitch = st.session_state.model.predict(speed, freq, power, 'rf')
                        nn_diameter, nn_pitch = st.session_state.model.predict(speed, freq, power, 'nn')
                        avg_diameter = (rf_diameter + nn_diameter) / 2
                        avg_pitch = (rf_pitch + nn_pitch) / 2
                        
                        # Display Diameter Predictions
                        st.subheader("Dimple Diameter Predictions")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Random Forest", f"{rf_diameter:.4f}")
                        with col2:
                            st.metric("Neural Network", f"{nn_diameter:.4f}")
                        with col3:
                            st.metric("Average", f"{avg_diameter:.4f}")
                        
                        # Display Pitch Predictions
                        st.subheader("Dimple Pitch Predictions")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Random Forest", f"{rf_pitch:.4f}")
                        with col2:
                            st.metric("Neural Network", f"{nn_pitch:.4f}")
                        with col3:
                            st.metric("Average", f"{avg_pitch:.4f}")
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
            except Exception as e:
                st.error(f"Error in prediction interface: {str(e)}")
                    
    elif selected_page == "Batch Analysis":
        st.header("Batch Analysis")
        
        if st.session_state.model is None or st.session_state.model.rf_model_diameter is None:
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
                
                col1, col2 = st.columns(2)
                with col1:
                    power_min = st.number_input("Minimum Power", min_value=0.0)
                    power_max = st.number_input("Maximum Power", min_value=power_min)
                with col2:
                    loop_min = st.number_input("Minimum Loop Count", min_value=1, step=1)
                    loop_max = st.number_input("Maximum Loop Count", min_value=loop_min, step=1)
                
                steps = st.number_input("Number of steps", min_value=2, value=5)
                
                if st.button("Generate Predictions"):
                    try:
                        speeds = np.linspace(speed_min, speed_max, steps)
                        freqs = np.linspace(freq_min, freq_max, steps)
                        powers = np.linspace(power_min, power_max, steps)
                        loops = np.linspace(loop_min, loop_max, steps, dtype=int)
                        
                        results = []
                        for speed in speeds:
                            for freq in freqs:
                                for power in powers:
                                    for loop in loops:
                                        rf_diameter, rf_pitch = st.session_state.model.predict(speed, freq, power, loop, 'rf')
                                        nn_diameter, nn_pitch = st.session_state.model.predict(speed, freq, power, loop, 'nn')
                                        avg_diameter = (rf_diameter + nn_diameter) / 2
                                        avg_pitch = (rf_pitch + nn_pitch) / 2
                                        results.append({
                                            'Speed': speed,
                                            'Frequency': freq,
                                            'Power': power,
                                            'Loop Count': loop,
                                            'RF Diameter': rf_diameter,
                                            'NN Diameter': nn_diameter,
                                            'Average Diameter': avg_diameter,
                                            'RF Pitch': rf_pitch,
                                            'NN Pitch': nn_pitch,
                                            'Average Pitch': avg_pitch
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
        
        if st.session_state.model is None or st.session_state.model.rf_model_diameter is None:
            st.warning("Please train or load models first!")
        else:
            try:
                if hasattr(st.session_state.model, 'X') and st.session_state.model.X is not None:
                    # Parameter selection
                    col1, col2, col3, col4 = st.columns(4)
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
                    with col3:
                        power_val = st.slider(
                            "Power Value",
                            float(st.session_state.model.X[st.session_state.model.feature_names[2]].min()),
                            float(st.session_state.model.X[st.session_state.model.feature_names[2]].max()),
                            float(st.session_state.model.X[st.session_state.model.feature_names[2]].mean())
                        )
                    with col4:
                        loop_count = st.number_input("Enter Loop Count", min_value=1, step=1)
                    
                    # Output type selection
                    output_type = st.radio(
                        "Select Output to Visualize",
                        ["Diameter", "Pitch"],
                        horizontal=True
                    )
                    
                    # Model tabs
                    tab1, tab2 = st.tabs(["Random Forest", "Neural Network"])
                    
                    with tab1:
                        rf_diameter, rf_pitch = st.session_state.model.predict(speed_range[0], freq_range[0], power_val, loop_count, 'rf')
                        rf_fig = create_3d_surface_plot(
                            st.session_state.model,
                            speed_range,
                            freq_range,
                            power_val,
                            loop_count,
                            output_type.lower(),
                            'rf'
                        )
                        if rf_fig:
                            st.plotly_chart(rf_fig)
                    
                    with tab2:
                        nn_diameter, nn_pitch = st.session_state.model.predict(speed_range[0], freq_range[0], power_val, loop_count, 'nn')
                        nn_fig = create_3d_surface_plot(
                            st.session_state.model,
                            speed_range,
                            freq_range,
                            power_val,
                            loop_count,
                            output_type.lower(),
                            'nn'
                        )
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
        - Predicting both dimple diameter and pitch accurately
        - Optimizing speed, frequency, power, and loop count parameters
        - Reducing the time and resources spent on parameter optimization
        - Improving dimple quality and consistency
        - Minimizing the need for trial-and-error experiments
        - Providing data-driven insights for process improvement

        ### 🤖 Machine Learning Models
        We utilize two sophisticated models for predictions:

        #### 1. Random Forest Models
        - Separate models for diameter and pitch prediction
        - Excellent at handling non-linear relationships
        - Robust against outliers
        - Provides feature importance insights
        - Ideal for understanding parameter interactions

        #### 2. Neural Network Models
        - Dedicated networks for diameter and pitch prediction
        - Captures complex patterns in the data
        - Learns hierarchical feature representations
        - Adapts to new patterns through training
        - Optimized for high-dimensional parameter spaces

        ### 📊 Input Parameters
        1. **Speed**
           - Controls the movement rate
           - Affects processing time and quality
           - Key factor in dimple formation

        2. **Frequency**
           - Determines pulse repetition rate
           - Influences energy distribution
           - Critical for dimple characteristics

        3. **Power**
           - Controls energy input
           - Affects material removal rate
           - Key for dimple depth and quality

        4. **Loop Count**
           - Determines number of passes
           - Influences dimple depth
           - Affects overall processing time

        ### 🎯 Output Predictions
        1. **Dimple Diameter**
           - Accurate prediction of average dimple diameter
           - Both RF and NN model estimates
           - Ensemble averaging for better accuracy

        2. **Dimple Pitch**
           - Precise prediction of dimple spacing
           - Multiple model predictions
           - Optimized for manufacturing requirements

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