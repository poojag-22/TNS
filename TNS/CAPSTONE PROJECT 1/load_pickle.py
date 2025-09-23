import pickle

# This code will open and read your 'scaler.pkl' file.
# It must be in the same folder as this script.

try:
    # Open the file in 'read binary' mode ('rb')
    with open('scaler.pkl', 'rb') as file:
        
        # Load the object from the pickle file
        loaded_scaler = pickle.load(file)
        
        # Print the loaded object to see what's inside
        print("Successfully loaded 'scaler.pkl'")
        print("The object is:")
        print(loaded_scaler)

except FileNotFoundError:
    print("Error: Make sure 'scaler.pkl' is in the same folder as this script.")
except Exception as e:
    print(f"An error occurred while loading the file: {e}")