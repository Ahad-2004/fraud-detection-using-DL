import joblib
import pandas as pd
import numpy as np

print("Testing preprocessor loading...")

try:
    preprocessor = joblib.load('professional_preprocessor.pkl')
    print("✅ Preprocessor loaded successfully!")
    print(f"Preprocessor type: {type(preprocessor)}")
    
    # Create test data matching your training format
    test_data = pd.DataFrame([{
        'age_of_driver': 24,
        'safty_rating': 0,  # Note: 'safty_rating' not 'safety_rating'
        'annual_income': 47305,
        'vehicle_price': 1768,
        'age_of_vehicle': 10,
        'past_num_of_claims': 6,
        'claim_est_payout': 11703,
        'vehicle_weight': 2524,
        'marital_status': '1.0',
        'high_education_ind': '0',
        'address_change_ind': '0',
        'witness_present_ind': '0.0',
        'liab_prct': '0-20',
        'policy_report_filed_ind': '0',
        'gender': 'M',
        'living_status': 'Rent',
        'claim_day_of_week': 'Monday',
        'accident_site': 'Highway',
        'channel': 'Phone',
        'vehicle_category': 'Sedan',
        'vehicle_color': 'Black'
    }])
    
    print("\nTest data columns:", test_data.columns.tolist())
    
    # Transform the data
    transformed = preprocessor.transform(test_data)
    print(f"\n✅ Transformation successful!")
    print(f"Output shape: {transformed.shape}")
    print(f"Expected shape: (1, 146)")
    
    if transformed.shape[1] == 146:
        print("\n🎉 SUCCESS! Preprocessor creates exactly 146 features as expected!")
    else:
        print(f"\n⚠️ WARNING: Expected 146 features but got {transformed.shape[1]}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
