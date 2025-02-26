from llm import LaserMicrodrillingModel
import os

def print_menu():
    print("\n=== Laser Microdrilling Model Interface ===")
    print("1. Train new models")
    print("2. Load existing models")
    print("3. Make single prediction")
    print("4. Make batch predictions")
    print("5. Ask general question about laser microdrilling")
    print("6. Compare model predictions")
    print("7. View model performance metrics")
    print("8. Exit")
    return input("Select an option (1-8): ")

def get_column_names():
    print("\nPlease enter the exact column names from your Excel file:")
    speed_col = input("Speed column name: ")
    freq_col = input("Frequency column name: ")
    diam_col = input("Average Diameter column name: ")
    return speed_col, freq_col, diam_col

def main():
    model = LaserMicrodrillingModel()
    
    while True:
        choice = print_menu()
        
        if choice == '1':
            # Train new models
            try:
                speed_col, freq_col, diam_col = get_column_names()
                model.load_data('EDI_OBSERVATIONS.xlsx', speed_col, freq_col, diam_col)
                model.train_models()
                model.evaluate_models()
                model.save_models()
                print("\nModels trained and saved successfully!")
            except Exception as e:
                print(f"\nError during training: {str(e)}")

        elif choice == '2':
            # Load existing models
            if os.path.exists('laser_models'):
                if model.load_models():
                    print("\nModels loaded successfully!")
            else:
                print("\nNo saved models found. Please train new models first.")

        elif choice == '3':
            # Make single prediction
            if model.rf_model is None:
                print("\nPlease load or train models first!")
                continue
                
            try:
                print("\nEnter values for prediction:")
                speed = float(input("Enter speed value: "))
                freq = float(input("Enter frequency value: "))
                
                print("\nPredictions:")
                rf_pred = model.predict_diameter(speed, freq, 'rf')
                nn_pred = model.predict_diameter(speed, freq, 'nn')
                
                print(f"Random Forest Prediction: {rf_pred:.4f}")
                print(f"Neural Network Prediction: {nn_pred:.4f}")
                print(f"Average Prediction: {(rf_pred + nn_pred)/2:.4f}")
            except ValueError:
                print("\nPlease enter valid numerical values!")
            except Exception as e:
                print(f"\nError making prediction: {str(e)}")

        elif choice == '4':
            # Make batch predictions
            if model.rf_model is None:
                print("\nPlease load or train models first!")
                continue
                
            try:
                print("\nEnter range for batch predictions:")
                speed_min = float(input("Minimum speed: "))
                speed_max = float(input("Maximum speed: "))
                freq_min = float(input("Minimum frequency: "))
                freq_max = float(input("Maximum frequency: "))
                steps = int(input("Number of steps (e.g., 5 for 5x5 grid): "))
                
                import numpy as np
                speeds = np.linspace(speed_min, speed_max, steps)
                freqs = np.linspace(freq_min, freq_max, steps)
                
                print("\nBatch Predictions:")
                print("Speed\tFrequency\tRF Prediction\tNN Prediction\tAverage")
                print("-" * 70)
                
                for speed in speeds:
                    for freq in freqs:
                        rf_pred = model.predict_diameter(speed, freq, 'rf')
                        nn_pred = model.predict_diameter(speed, freq, 'nn')
                        avg_pred = (rf_pred + nn_pred) / 2
                        print(f"{speed:.2f}\t{freq:.2f}\t\t{rf_pred:.4f}\t{nn_pred:.4f}\t{avg_pred:.4f}")
            except ValueError:
                print("\nPlease enter valid numerical values!")
            except Exception as e:
                print(f"\nError making batch predictions: {str(e)}")

        elif choice == '5':
            # Ask general question
            if model.nlp_pipeline is None:
                print("\nNLP model not initialized. Only numerical predictions are available.")
                continue
                
            try:
                query = input("\nEnter your question about laser microdrilling: ")
                response = model.process_general_query(query)
                print(f"\nResponse: {response}")
            except Exception as e:
                print(f"\nError processing query: {str(e)}")

        elif choice == '6':
            # Compare model predictions
            if model.rf_model is None:
                print("\nPlease load or train models first!")
                continue
                
            try:
                model.evaluate_models()
                print("\nVisualization saved as 'model_predictions_comparison.png'")
            except Exception as e:
                print(f"\nError comparing models: {str(e)}")

        elif choice == '7':
            # View model performance metrics
            if model.rf_model is None:
                print("\nPlease load or train models first!")
                continue
                
            try:
                model.evaluate_models()
            except Exception as e:
                print(f"\nError calculating metrics: {str(e)}")

        elif choice == '8':
            print("\nThank you for using the Laser Microdrilling Model!")
            break

        else:
            print("\nInvalid choice! Please select a number between 1 and 8.")

if __name__ == "__main__":
    main() 