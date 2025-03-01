import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from torch.utils.data import Dataset
import warnings
import joblib
import os
warnings.filterwarnings('ignore')  # Suppress warnings

class LaserMicrodrillingModel:
    def __init__(self):
        self.rf_model_diameter = None
        self.nn_model_diameter = None
        self.rf_model_pitch = None
        self.nn_model_pitch = None
        self.scaler = None
        self.feature_names = None
        self.target_names = None
        self.nlp_pipeline = None
        
    def load_data(self, data, speed_col, freq_col, power_col, diam_col, pitch_col):
        """Load and prepare data for training"""
        print("Loading dataset...")
        self.data = data  # Use the passed DataFrame directly
        self.feature_names = [speed_col, freq_col, power_col]
        self.target_names = [diam_col, pitch_col]
        
        # Create feature DataFrame
        self.X = pd.DataFrame(self.data[[speed_col, freq_col, power_col]])
        self.y_diameter = self.data[diam_col]
        self.y_pitch = self.data[pitch_col]
        
        # Split and scale data
        self.X_train, self.X_test, self.y_train_diameter, self.y_test_diameter, self.y_train_pitch, self.y_test_pitch = train_test_split(
            self.X, self.y_diameter, self.y_pitch, test_size=0.2, random_state=42
        )
        
        self.scaler = StandardScaler()
        self.X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
    def train_models(self):
        """Train both Random Forest and Neural Network models for diameter and pitch"""
        print("Training models...")
        
        # Train Random Forest models
        print("Training Random Forest models...")
        self.rf_model_diameter = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model_pitch = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.rf_model_diameter.fit(self.X_train_scaled, self.y_train_diameter)
        self.rf_model_pitch.fit(self.X_train_scaled, self.y_train_pitch)
        
        # Train Neural Network models
        print("Training Neural Network models...")
        
        # Define NN architecture for both diameter and pitch
        class NeuralNetwork(torch.nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(input_size, 64),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(64, 32),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(32, 1)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Initialize and train NN for diameter
        self.nn_model_diameter = NeuralNetwork(len(self.feature_names))
        self._train_neural_network(self.nn_model_diameter, self.y_train_diameter, "diameter")
        
        # Initialize and train NN for pitch
        self.nn_model_pitch = NeuralNetwork(len(self.feature_names))
        self._train_neural_network(self.nn_model_pitch, self.y_train_pitch, "pitch")
    
    def _train_neural_network(self, model, target_data, target_name):
        """Helper function to train neural networks"""
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        X_tensor = torch.FloatTensor(self.X_train_scaled.values)
        y_tensor = torch.FloatTensor(target_data.values.reshape(-1, 1))
        
        epochs = 100
        batch_size = 32
        
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], {target_name.capitalize()} Loss: {loss.item():.4f}")
    
    def evaluate_models(self):
        """Evaluate both models' performance"""
        print("\nEvaluating models...")
        
        # Evaluate Random Forest models
        rf_pred_diameter = self.rf_model_diameter.predict(self.X_test_scaled)
        rf_pred_pitch = self.rf_model_pitch.predict(self.X_test_scaled)
        
        print("\nRandom Forest Performance:")
        print(f"Diameter - R² Score: {r2_score(self.y_test_diameter, rf_pred_diameter):.4f}")
        print(f"Pitch - R² Score: {r2_score(self.y_test_pitch, rf_pred_pitch):.4f}")
        
        # Evaluate Neural Network models
        self.nn_model_diameter.eval()
        self.nn_model_pitch.eval()
        with torch.no_grad():
            nn_pred_diameter = self.nn_model_diameter(torch.FloatTensor(self.X_test_scaled.values)).numpy()
            nn_pred_pitch = self.nn_model_pitch(torch.FloatTensor(self.X_test_scaled.values)).numpy()
        
        print("\nNeural Network Performance:")
        print(f"Diameter - R² Score: {r2_score(self.y_test_diameter, nn_pred_diameter):.4f}")
        print(f"Pitch - R² Score: {r2_score(self.y_test_pitch, nn_pred_pitch):.4f}")
    
    def predict(self, speed, frequency, power, model_type='rf'):
        """Predict both diameter and pitch for given parameters"""
        input_data = pd.DataFrame([[speed, frequency, power]], columns=self.feature_names)
        input_scaled = self.scaler.transform(input_data)
        
        if model_type.lower() == 'rf':
            diameter_pred = self.rf_model_diameter.predict(input_scaled)[0]
            pitch_pred = self.rf_model_pitch.predict(input_scaled)[0]
        elif model_type.lower() == 'nn':
            self.nn_model_diameter.eval()
            self.nn_model_pitch.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_scaled)
                diameter_pred = self.nn_model_diameter(input_tensor).item()
                pitch_pred = self.nn_model_pitch(input_tensor).item()
        else:
            raise ValueError("model_type must be 'rf' or 'nn'")
        
        return diameter_pred, pitch_pred
    
    def process_general_query(self, query):
        """
        Process general queries about laser microdrilling using predefined responses.
        
        Args:
            query (str): The user's question or query
            
        Returns:
            str: The AI-generated response
        """
        try:
            # Define comprehensive responses for frequently asked questions
            common_responses = {
                "what is laser microdrilling": "Laser microdrilling is a precision manufacturing process that uses focused laser beams to create very small holes in materials. It's commonly used in aerospace, electronics, and medical device manufacturing.",
                "how does laser microdrilling work": "Laser microdrilling works by focusing a high-power laser beam onto a material's surface. The laser's energy vaporizes or melts the material, creating precise holes. The process is controlled by parameters like laser power, speed, and frequency.",
                "what is the principle": "The principle of laser microdrilling is based on material removal through laser-matter interaction. When a focused laser beam hits the material surface, it causes localized heating, melting, and vaporization, creating a hole through controlled material removal.",
                "what is a dimple": "A dimple refers to the small, concave indentation created in a material during the laser microdrilling process. It is critical in applications requiring precise hole sizes and shapes.",
                "what is pitch": "Pitch refers to the distance between two adjacent dimples or holes in a material. It is crucial for applications requiring specific spacing for functionality.",
                "what is a hatch pattern": "A hatch pattern is a specific arrangement of laser passes used to create a series of dimples or holes in a material. It affects the efficiency of the drilling process and the quality of the final product.",
                "what is spacing": "Spacing refers to the distance between individual dimples or holes, important for ensuring that the holes do not interfere with each other.",
                "what are laser parameters": "Key laser parameters include power, frequency, speed, and pulse duration, all of which influence the quality and characteristics of the drilled holes.",
                "what materials can be processed": "Laser microdrilling can process metals, ceramics, polymers, composites, semiconductors, and glass.",
                "what are the applications": "Applications include aerospace components, medical devices, electronic circuit boards, fuel injection nozzles, and microfluidic devices.",
                "how to control hole size": "Hole size can be controlled by adjusting laser power, focus position, pulse frequency, and drilling speed.",
                "what affects drilling speed": "Drilling speed is affected by material thickness, material properties, laser power, and required hole quality.",
                "what is the importance of thermal conductivity": "Thermal conductivity affects how heat is dissipated during the drilling process, influencing dimple quality.",
                "what is the absorption coefficient": "The absorption coefficient determines how well the material absorbs laser energy, impacting the efficiency of the drilling process.",
                "how to improve accuracy": "Improve accuracy by ensuring proper material preparation, system calibration, and parameter optimization.",
                "what are common problems": "Common issues include irregular hole shapes, taper formation, heat affected zones, and incomplete penetration.",
                "how to fix quality issues": "Fix quality issues by adjusting parameters, performing system maintenance, and ensuring proper material preparation.",
                "what are advanced techniques": "Advanced techniques include trepanning, percussion drilling, and multi-pulse strategies.",
                "what are safety measures": "Safety measures include implementing laser safety protocols, using personal protective equipment (PPE), and ensuring proper ventilation.",
                "what is the environmental impact": "Environmental considerations include energy efficiency, waste management, and emission control.",
                "what are cost factors": "Cost factors include equipment investment, operating costs, maintenance, and material costs.",
                "what are required skills": "Required skills include laser operation, process control, quality inspection, and safety procedures.",
                "what are quality standards": "Quality standards include ISO requirements, industry specifications, and customer requirements.",
                "what is system integration": "System integration involves setting up production lines, control systems, and data collection mechanisms.",
                "what are the economic benefits": "Economic benefits include high productivity, reduced waste, consistent quality, and lower labor costs.",
                "what is the future of laser microdrilling": "Future trends include AI integration, process automation, and sustainable practices."
            }
            
            # Check if the query matches any common questions
            query_lower = query.lower()
            for key, response in common_responses.items():
                if key in query_lower:
                    return response
            
            # For queries about specific parameters
            if "dimple" in query_lower:
                return "A dimple refers to the small, concave indentation created in a material during the laser microdrilling process. It is critical in applications requiring precise hole sizes and shapes."
            
            if "pitch" in query_lower:
                return "Pitch refers to the distance between two adjacent dimples or holes in a material, crucial for applications requiring specific spacing for functionality."
            
            if "hatch pattern" in query_lower:
                return "A hatch pattern is a specific arrangement of laser passes used to create a series of dimples or holes in a material, affecting the efficiency of the drilling process."
            
            if "spacing" in query_lower:
                return "Spacing refers to the distance between individual dimples or holes, important for ensuring that the holes do not interfere with each other."
            
            # Add more terms as needed...
            
            # If no specific match, provide a general response
            return ("I understand you're asking about laser microdrilling. While I can provide general guidance, "
                   "I recommend using our prediction tools for specific parameter recommendations. "
                   "You can also try asking more specific questions about speed, frequency, hole diameter, "
                   "process optimization, materials, applications, or quality control.")
            
        except Exception as e:
            return f"I apologize, but I encountered an error processing your query. Please try rephrasing your question or contact support. Error: {str(e)}"
    
    def save_models(self):
        """Save all models and scaler"""
        print("\nSaving models...")
        if not os.path.exists('laser_models'):
            os.makedirs('laser_models')
        
        # Save Random Forest models
        joblib.dump(self.rf_model_diameter, 'laser_models/rf_model_diameter.joblib')
        joblib.dump(self.rf_model_pitch, 'laser_models/rf_model_pitch.joblib')
        
        # Save Neural Network models
        torch.save(self.nn_model_diameter.state_dict(), 'laser_models/nn_model_diameter.pth')
        torch.save(self.nn_model_pitch.state_dict(), 'laser_models/nn_model_pitch.pth')
        
        # Save scaler
        joblib.dump(self.scaler, 'laser_models/scaler.joblib')
    
    def load_models(self):
        """Load all saved models and scaler"""
        print("\nLoading models...")
        try:
            # Load Random Forest models
            self.rf_model_diameter = joblib.load('laser_models/rf_model_diameter.joblib')
            self.rf_model_pitch = joblib.load('laser_models/rf_model_pitch.joblib')
            
            # Load Neural Network models
            class NeuralNetwork(torch.nn.Module):
                def __init__(self, input_size=3):
                    super().__init__()
                    self.layers = torch.nn.Sequential(
                        torch.nn.Linear(input_size, 64),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(64, 32),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(0.2),
                        torch.nn.Linear(32, 1)
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            self.nn_model_diameter = NeuralNetwork()
            self.nn_model_pitch = NeuralNetwork()
            
            self.nn_model_diameter.load_state_dict(torch.load('laser_models/nn_model_diameter.pth'))
            self.nn_model_pitch.load_state_dict(torch.load('laser_models/nn_model_pitch.pth'))
            
            # Load scaler
            self.scaler = joblib.load('laser_models/scaler.joblib')
            
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = LaserMicrodrillingModel()
    
    # Load and prepare data
    print("Loading dataset...")
    data = pd.read_excel('EDI_OBSERVATIONS.xlsx')
    
    # Display information about the dataset
    print("\nDataset Info:")
    print(data.info())
    
    print("\nColumn names in the dataset:")
    for col in data.columns:
        print(f"- {col}")
    
    print("\nFirst few rows of the dataset:")
    print(data.head())
    
    # Get column names from user
    print("\nPlease check the column names above and press Enter to continue...")
    input()
    
    speed_col = input("Enter the exact name of the Speed column from above: ")
    freq_col = input("Enter the exact name of the Frequency column from above: ")
    power_col = input("Enter the exact name of the Power column from above: ")
    diam_col = input("Enter the exact name of the Average Diameter column from above: ")
    pitch_col = input("Enter the exact name of the Pitch column from above: ")
    
    # Train models
    model.load_data(data, speed_col, freq_col, power_col, diam_col, pitch_col)
    model.train_models()
    model.evaluate_models()
    
    # Save models
    model.save_models()
    
    # Example predictions
    print("\nExample Predictions:")
    test_speeds = [model.X[speed_col].mean(), model.X[speed_col].min(), model.X[speed_col].max()]
    test_freqs = [model.X[freq_col].mean(), model.X[freq_col].min(), model.X[freq_col].max()]
    test_powers = [model.X[power_col].mean(), model.X[power_col].min(), model.X[power_col].max()]
    
    for speed in test_speeds:
        for freq in test_freqs:
            for power in test_powers:
                diameter_pred, pitch_pred = model.predict(speed, freq, power, 'nn')
                print(f"\nInput: {speed_col}={speed:.2f}, {freq_col}={freq:.2f}, {power_col}={power:.2f}")
                print(f"Predicted Diameter: {diameter_pred:.4f}")
                print(f"Predicted Pitch: {pitch_pred:.4f}")
    
    # Example of general task processing
    print("\nExample of general task processing:")
    query = "What are the key factors affecting laser microdrilling?"
    response = model.process_general_query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")
