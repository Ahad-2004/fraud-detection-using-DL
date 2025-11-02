import pickle
import sys

# Try to load and re-save the preprocessor with a compatible protocol
input_file = "professional_preprocessor.pkl"
output_file = "professional_preprocessor_fixed.pkl"

print(f"Attempting to fix {input_file}...")

try:
    # Try loading with Python 3.13 compatible settings
    with open(input_file, 'rb') as f:
        import pickle5
        preprocessor = pickle5.load(f)
    print("Loaded with pickle5")
except:
    try:
        # Try with joblib which is often used for sklearn objects
        import joblib
        preprocessor = joblib.load(input_file)
        print("Loaded with joblib")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative method...")
        
        # Last resort: try to load with older Python
        import subprocess
        print("Please install Python 3.10 or 3.11 and run this script with that version")
        sys.exit(1)

# Save with a compatible protocol
with open(output_file, 'wb') as f:
    pickle.dump(preprocessor, f, protocol=4)

print(f"Successfully saved fixed preprocessor to {output_file}")
print("Now update main.py to use 'professional_preprocessor_fixed.pkl'")
