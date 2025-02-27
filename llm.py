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
        
    def load_data(self, speed_col, freq_col, power_col, diam_col, pitch_col):
        """Load and prepare data for training"""
        print("Loading dataset...")
        self.data = pd.read_excel('EDI_OBSERVATIONS.xlsx')
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
                # Basic Concepts and Definitions
                "what is laser microdrilling": "Laser microdrilling is a precision manufacturing process that uses focused laser beams to create very small holes in materials. It's commonly used in aerospace, electronics, and medical device manufacturing.",
                "how does laser microdrilling work": "Laser microdrilling works by focusing a high-power laser beam onto a material's surface. The laser's energy vaporizes or melts the material, creating precise holes. The process is controlled by parameters like laser power, speed, and frequency.",
                "what is the principle": "The principle of laser microdrilling is based on material removal through laser-matter interaction. When a focused laser beam hits the material surface, it causes localized heating, melting, and vaporization, creating a hole through controlled material removal.",
                
                # Process Parameters and Controls
                "what are the key parameters": "Key parameters in laser microdrilling include: 1) Laser power, 2) Pulse frequency, 3) Drilling speed, 4) Focus position, 5) Assist gas pressure, 6) Pulse duration, 7) Beam quality, 8) Material properties.",
                "how to control hole size": "Hole size in laser microdrilling can be controlled through: 1) Laser power adjustment, 2) Focus position optimization, 3) Pulse frequency selection, 4) Drilling speed control, 5) Multiple pass strategies, 6) Beam shaping techniques.",
                "what affects drilling speed": "Drilling speed is affected by: 1) Material thickness, 2) Material properties, 3) Laser power, 4) Required hole quality, 5) Assist gas parameters, 6) Focal length, 7) Beam characteristics.",
                
                # Materials and Applications
                "what materials can be processed": "Laser microdrilling can process: 1) Metals (steel, aluminum, titanium), 2) Ceramics, 3) Polymers, 4) Composites, 5) Semiconductors, 6) Glass, 7) Aerospace alloys, 8) Medical-grade materials.",
                "what are the applications": "Applications include: 1) Aerospace components, 2) Electronic circuit boards, 3) Medical devices, 4) Fuel injection nozzles, 5) Filtration systems, 6) Cooling holes in turbine blades, 7) Microfluidic devices, 8) Precision instruments.",
                "industry uses": "Industries using laser microdrilling: 1) Aerospace, 2) Automotive, 3) Electronics, 4) Medical devices, 5) Energy sector, 6) Chemical processing, 7) Research facilities, 8) Defense.",
                
                # Quality and Optimization
                "what affects hole quality": "Hole quality factors: 1) Laser parameters, 2) Material properties, 3) Process conditions, 4) Environmental factors, 5) System calibration, 6) Assist gas flow, 7) Focus position, 8) Material preparation.",
                "how to improve accuracy": "Improve accuracy by: 1) Proper material preparation, 2) System calibration, 3) Parameter optimization, 4) Environmental control, 5) Regular maintenance, 6) Quality monitoring, 7) Process validation.",
                "quality control methods": "Quality control methods: 1) Visual inspection, 2) Microscopy, 3) Dimensional measurement, 4) Roundness testing, 5) Surface roughness analysis, 6) Statistical process control, 7) Automated inspection.",
                
                # Process Optimization
                "how to optimize process": "Process optimization involves: 1) Parameter study, 2) DOE approach, 3) Statistical analysis, 4) Quality monitoring, 5) Process validation, 6) Continuous improvement, 7) Feedback control.",
                "best practices": "Best practices include: 1) Regular calibration, 2) Material preparation, 3) Parameter verification, 4) Quality checks, 5) Process documentation, 6) Operator training, 7) Maintenance schedule.",
                "process monitoring": "Monitor: 1) Laser power, 2) Beam quality, 3) Process emissions, 4) Temperature, 5) Assist gas flow, 6) Focus position, 7) Material positioning.",
                
                # Troubleshooting
                "common problems": "Common issues: 1) Irregular hole shape, 2) Taper formation, 3) Heat affected zone, 4) Recast layer, 5) Spatter formation, 6) Incomplete penetration, 7) Surface damage.",
                "how to fix quality issues": "Fix quality issues by: 1) Parameter adjustment, 2) System maintenance, 3) Material preparation, 4) Process validation, 5) Environmental control, 6) Tool optimization.",
                "process defects": "Common defects: 1) Hole taper, 2) Surface roughness, 3) Recast layer, 4) Microcracks, 5) Heat affected zone, 6) Spatter, 7) Incomplete penetration.",
                
                # Advanced Topics
                "advanced techniques": "Advanced techniques: 1) Trepanning, 2) Helical drilling, 3) Percussion drilling, 4) Multi-pulse strategies, 5) Beam shaping, 6) Process monitoring.",
                "new developments": "Recent developments: 1) Ultrafast lasers, 2) Beam shaping, 3) Process monitoring, 4) Automation, 5) Quality control, 6) Material processing.",
                "future trends": "Future trends: 1) AI integration, 2) Process automation, 3) Quality monitoring, 4) Sustainable processing, 5) New materials, 6) Industry 4.0 integration.",
                
                # Safety and Environment
                "safety measures": "Safety measures: 1) Laser safety protocols, 2) PPE requirements, 3) Enclosure systems, 4) Ventilation, 5) Training programs, 6) Emergency procedures.",
                "environmental impact": "Environmental considerations: 1) Energy efficiency, 2) Waste management, 3) Emission control, 4) Resource optimization, 5) Sustainable practices.",
                "safety equipment": "Required safety equipment: 1) Laser safety glasses, 2) Enclosures, 3) Ventilation systems, 4) Warning signs, 5) Emergency stops, 6) Monitoring systems.",
                
                # Equipment and Setup
                "equipment needed": "Required equipment: 1) Laser source, 2) Beam delivery system, 3) Motion system, 4) Control system, 5) Assist gas supply, 6) Monitoring devices.",
                "system setup": "System setup includes: 1) Laser alignment, 2) Focus adjustment, 3) Gas supply connection, 4) Control system setup, 5) Safety system verification.",
                "maintenance requirements": "Maintenance needs: 1) Regular calibration, 2) Optics cleaning, 3) Alignment checks, 4) Gas system maintenance, 5) Safety system verification.",
                
                # Process Selection
                "when to use": "Use laser microdrilling when: 1) High precision needed, 2) Small hole sizes required, 3) Complex materials, 4) High production rates, 5) Quality critical.",
                "advantages": "Advantages: 1) High precision, 2) No tool wear, 3) Complex geometries, 4) Various materials, 5) Automated process, 6) Consistent quality.",
                "limitations": "Limitations: 1) Initial cost, 2) Material restrictions, 3) Heat effects, 4) Energy consumption, 5) Technical expertise needed.",
                
                # Cost and Economics
                "cost factors": "Cost factors: 1) Equipment investment, 2) Operating costs, 3) Maintenance, 4) Material costs, 5) Labor, 6) Energy consumption.",
                "economic benefits": "Economic benefits: 1) High productivity, 2) Reduced waste, 3) Consistent quality, 4) Lower labor costs, 5) Flexible production.",
                "roi calculation": "ROI considerations: 1) Equipment costs, 2) Operating expenses, 3) Production rates, 4) Quality improvements, 5) Reduced waste.",
                
                # Training and Skills
                "required skills": "Required skills: 1) Laser operation, 2) Process control, 3) Quality inspection, 4) Safety procedures, 5) Troubleshooting, 6) Documentation.",
                "training needs": "Training requirements: 1) Safety protocols, 2) Equipment operation, 3) Process control, 4) Quality assessment, 5) Maintenance procedures.",
                "operator qualifications": "Operator needs: 1) Technical background, 2) Safety certification, 3) Process knowledge, 4) Quality control skills, 5) Documentation ability.",
                
                # Quality Standards
                "quality standards": "Quality standards: 1) ISO requirements, 2) Industry specifications, 3) Customer requirements, 4) Internal standards, 5) Safety regulations.",
                "certification requirements": "Certifications: 1) ISO 9001, 2) Industry specific, 3) Safety standards, 4) Environmental compliance, 5) Quality system requirements.",
                "inspection methods": "Inspection methods: 1) Visual inspection, 2) Dimensional measurement, 3) Surface analysis, 4) Metallographic examination, 5) Non-destructive testing.",
                
                # Process Integration
                "system integration": "Integration aspects: 1) Production line setup, 2) Control systems, 3) Data collection, 4) Quality monitoring, 5) Process automation.",
                "automation options": "Automation includes: 1) Material handling, 2) Process control, 3) Quality inspection, 4) Data collection, 5) Report generation.",
                "industry 4.0": "Industry 4.0 features: 1) Data analytics, 2) Process monitoring, 3) Predictive maintenance, 4) Quality control, 5) Remote operation.",
                
                # Research and Development
                "research areas": "Research focus: 1) New materials, 2) Process optimization, 3) Quality improvement, 4) Cost reduction, 5) Sustainability.",
                "recent advances": "Recent advances: 1) Ultrafast lasers, 2) Process monitoring, 3) Automation, 4) Quality control, 5) New applications.",
                "future developments": "Future developments: 1) AI integration, 2) Smart manufacturing, 3) New materials, 4) Process optimization, 5) Sustainability.",
                
                # Specific Parameters
                "speed effects": "Speed affects: 1) Hole quality, 2) Production rate, 3) Heat effects, 4) Material removal, 5) Process stability.",
                "frequency impact": "Frequency impacts: 1) Hole size, 2) Quality, 3) Heat effects, 4) Material removal, 5) Process efficiency.",
                "power settings": "Power settings affect: 1) Penetration depth, 2) Processing speed, 3) Heat effects, 4) Quality, 5) Material removal.",
                
                # Process Monitoring
                "monitoring methods": "Monitoring includes: 1) Power measurement, 2) Process emissions, 3) Temperature control, 4) Quality inspection, 5) Data collection.",
                "quality control": "Quality control: 1) In-process monitoring, 2) Post-process inspection, 3) Statistical analysis, 4) Documentation, 5) Feedback control.",
                "process control": "Process control: 1) Parameter monitoring, 2) Quality checks, 3) Feedback systems, 4) Data analysis, 5) Adjustment procedures.",
                
                # Maintenance and Service
                "maintenance schedule": "Maintenance includes: 1) Daily checks, 2) Weekly cleaning, 3) Monthly calibration, 4) Quarterly service, 5) Annual overhaul.",
                "service requirements": "Service needs: 1) Regular inspection, 2) Component replacement, 3) Calibration, 4) Safety checks, 5) Performance validation.",
                "troubleshooting guide": "Troubleshooting: 1) Visual inspection, 2) Parameter check, 3) System diagnostics, 4) Component testing, 5) Performance validation.",
                
                # Documentation and Records
                "required documentation": "Documentation needs: 1) Process parameters, 2) Quality records, 3) Maintenance logs, 4) Training records, 5) Safety procedures.",
                "record keeping": "Keep records of: 1) Process parameters, 2) Quality data, 3) Maintenance activities, 4) Training, 5) Safety incidents.",
                "quality records": "Quality records: 1) Process parameters, 2) Inspection results, 3) Non-conformance reports, 4) Corrective actions, 5) Validation data.",
                
                # Tool Usage
                "how to use this tool": "To use this tool: 1) Upload your dataset, 2) Select columns, 3) Train models, 4) Make predictions, 5) Analyze results, 6) Optimize parameters.",
                "what do the models predict": "Our models predict hole diameter based on: 1) Drilling speed, 2) Laser frequency, using both Random Forest and Neural Network approaches.",
                "model capabilities": "Model capabilities: 1) Diameter prediction, 2) Parameter optimization, 3) Process analysis, 4) Quality estimation, 5) Performance evaluation."
            }
            
            # Check if the query matches any common questions
            query_lower = query.lower()
            for key, response in common_responses.items():
                if any(word in query_lower for word in key.split()):
                    return response
            
            # For queries about specific parameters
            if "speed" in query_lower:
                return "Drilling speed is a crucial parameter that affects hole quality. Higher speeds generally result in smaller hole diameters but may affect quality. Use our prediction tool to find optimal speed values for your application."
            
            if "frequency" in query_lower:
                return "Laser frequency determines how many pulses are delivered per second. Higher frequencies can lead to more precise holes but may require optimization. Our model can help you find the best frequency for your needs."
            
            if "diameter" in query_lower:
                return "Hole diameter is the key output parameter we predict. It's influenced by both speed and frequency. Our models can help you achieve the desired diameter by suggesting optimal parameter combinations."
            
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
    model.load_data(speed_col, freq_col, power_col, diam_col, pitch_col)
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
