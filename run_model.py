from llm import LaserMicrodrillingModel

def main():
    # Initialize model (note the parentheses)
    model = LaserMicrodrillingModel()
    
    try:
        # Load and prepare data
        print("Loading dataset...")
        data_file = 'EDI_OBSERVATIONS.xlsx'
        
        # Get column names from user
        print("\nPlease enter the exact column names from your Excel file:")
        speed_col = input("Speed column name: ")
        freq_col = input("Frequency column name: ")
        diam_col = input("Average Diameter column name: ")
        
        # Train and evaluate models
        model.load_data(data_file, speed_col, freq_col, diam_col)
        model.train_models()
        model.evaluate_models()
        
        # Save the trained models
        model.save_models()
        
        # Make some example predictions
        print("\nMaking example predictions...")
        # Test with minimum, mean, and maximum values
        test_speeds = [model.X[speed_col].mean(), model.X[speed_col].min(), model.X[speed_col].max()]
        test_freqs = [model.X[freq_col].mean(), model.X[freq_col].min(), model.X[freq_col].max()]
        
        for speed in test_speeds:
            for freq in test_freqs:
                rf_pred = model.predict_diameter(speed, freq, 'rf')
                nn_pred = model.predict_diameter(speed, freq, 'nn')
                print(f"\nInput: {speed_col}={speed:.2f}, {freq_col}={freq:.2f}")
                print(f"Random Forest Prediction: {rf_pred:.4f}")
                print(f"Neural Network Prediction: {nn_pred:.4f}")
        
        # Try a general query
        print("\nTesting general query capability...")
        query = "What are the key factors affecting laser microdrilling?"
        response = model.process_general_query(query)
        print(f"Query: {query}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please make sure:")
        print("1. The Excel file 'EDI_OBSERVATIONS.xlsx' is in the same directory")
        print("2. The column names you entered match exactly with those in the Excel file")
        print("3. All required packages are installed")

if __name__ == "__main__":
    main() 