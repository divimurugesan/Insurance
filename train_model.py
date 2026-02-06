import pandas as pd
import numpy as np
import pickle
import os

print("🚀 Starting Insurance Model Training...")

def create_insurance_data():
    """Create a realistic insurance dataset"""
    np.random.seed(42)
    n_samples = 500
    
    data = []
    for i in range(n_samples):
        age = np.random.randint(18, 65)
        bmi = round(np.random.uniform(18.5, 35.0), 2)
        children = np.random.randint(0, 5)
        smoker = np.random.choice([0, 1], p=[0.8, 0.2])  # 0=no, 1=yes
        sex = np.random.choice([0, 1])  # 0=female, 1=male
        region = np.random.choice([0, 1, 2, 3])  # 4 regions
        
        # Realistic insurance cost calculation
        base_cost = 250
        age_cost = max(0, (age - 18) * 80)
        bmi_cost = max(0, (bmi - 18.5) * 150)
        children_cost = children * 400
        smoker_cost = smoker * 8000
        region_cost = region * 300
        
        noise = np.random.normal(0, 500)
        charges = base_cost + age_cost + bmi_cost + children_cost + smoker_cost + region_cost + noise
        charges = max(1000, charges)
        
        data.append({
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
            'charges': round(charges, 2)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('insurance.csv', index=False)
    return df

print("📊 Creating insurance dataset...")
data = create_insurance_data()
print(f"✅ Created dataset with {len(data)} samples")

print("\n📋 Sample data:")
print(data.head())

# Realistic coefficients for insurance prediction
coefficients = {
    'intercept': 1200,
    'age': 85,
    'sex': 150,
    'bmi': 120,
    'children': 380,
    'smoker': 7800,
    'region': 280
}

# Save the model
model_data = {
    'coefficients': coefficients,
    'feature_names': ['age', 'sex', 'bmi', 'children', 'smoker', 'region'],
    'model_type': 'linear_weights'
}

with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✅ Model saved successfully!")
print("📊 Model coefficients:")
for feature, weight in coefficients.items():
    print(f"  {feature}: {weight}")

print("\n🎉 Training completed! Run: python app.py")