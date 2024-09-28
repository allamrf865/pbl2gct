# Step 1: Import Necessary Libraries
import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 2: Load and Display Dataset
base_path = 'Dataset2_Edema.xlsx'  # Ensure the path is correct
try:
    df = pd.read_excel(base_path)
    st.write("### Sample Data")
    st.dataframe(df.head())  # Display the first few rows of the DataFrame

    # Display column names for reference
    st.write("### Columns in the Dataset")
    st.write(df.columns.tolist())

except FileNotFoundError:
    st.error(f"Dataset '{base_path}' not found. Please ensure the file is in the correct path.")

# Step 3: Define and Add Target_Class Based on Predefined Rules
def add_target_class(df):
    # Define rules for each diagnosis (customize based on your data)
    conditions = [
        (df['Serum_Albumin_g/dL'] < 2.5) & (df['Proteinuria_Level'] > 3000),  # Minimal Change Disease
        (df['Blood_Pressure_Systolic'] > 140) & (df['Blood_Pressure_Diastolic'] > 90) & (df['Heart_Disease'] == 1),  # Heart Failure
        (df['Portal_Hypertension'] == 1) & (df['Serum_Albumin_g/dL'] < 3),  # Liver Cirrhosis
        (df['Serum_Creatinine_mg/dL'] > 1.5) | (df['Chronic_Renal_Failure'] == 1),  # Chronic Kidney Disease
        (df['Overfill_Edema'] == 1),  # Nephrotic Syndrome Edema
        (df['BMI'] < 18.5) & (df['Serum_Albumin_g/dL'] < 3),  # Malnutrition
        (df['Congestive_Heart_Failure'] == 1)  # Heart-related Edema
    ]

    # Corresponding target classes
    choices = [
        'Minimal Change Disease',
        'Heart Failure',
        'Liver Cirrhosis',
        'Chronic Kidney Disease',
        'Nephrotic Syndrome Edema',
        'Malnutrition',
        'Congestive Heart Failure'
    ]

    # Create the Target_Class column
    df['Target_Class'] = pd.Series('Other')  # Default class if no conditions are met
    for cond, choice in zip(conditions, choices):
        df.loc[cond, 'Target_Class'] = choice

    return df

# Apply the function to add Target_Class
df = add_target_class(df)

# Step 4: Preprocess and Train Model
def preprocess_and_train(df):
    # Define the columns to be used as features
    FEATURE_COLUMNS = [
        'Age', 'Weight_kg', 'Height_cm', 'Blood_Pressure_Systolic', 'Blood_Pressure_Diastolic',
        'Serum_Albumin_g/dL', 'Proteinuria_Level', 'Serum_Creatinine_mg/dL', 'BMI',
        'Heart_Disease', 'Portal_Hypertension', 'Chronic_Renal_Failure', 'Overfill_Edema',
        'Congestive_Heart_Failure'
    ]

    # Filter the DataFrame to include only the feature columns and the target
    df = df[FEATURE_COLUMNS + ['Target_Class']]

    # Convert categorical columns to numeric
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.Categorical(df[col]).codes

    # Check and handle NaNs or infinite values
    df = df.replace([float('inf'), float('-inf')], float('nan')).dropna()

    # Separate features and target
    X = df[FEATURE_COLUMNS]
    y = df['Target_Class']

    # Normalize target classes
    y = y.map({
        'Minimal Change Disease': 0, 
        'Other': 1, 
        'Heart Failure': 2, 
        'Liver Cirrhosis': 3, 
        'Chronic Kidney Disease': 4,
        'Nephrotic Syndrome Edema': 5,
        'Malnutrition': 6,
        'Congestive Heart Failure': 7
    }).fillna(0).astype(int)  # Ensure no NaNs in y

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42, 
        use_label_encoder=False, eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)

    return xgb_model, X_test, y_test

# Train the model
xgb_model, X_test, y_test = preprocess_and_train(df)

if xgb_model is not None:
    # Step 5: Create Input Fields for User Prediction
    st.write("## Input Patient Data for Diagnosis")

    # Create input fields
    age = st.number_input("Age", min_value=0)
    weight_kg = st.number_input("Weight (kg)", min_value=0.0)
    height_cm = st.number_input("Height (cm)", min_value=0.0)
    blood_pressure_systolic = st.number_input("Blood Pressure Systolic (mmHg)", min_value=0)
    blood_pressure_diastolic = st.number_input("Blood Pressure Diastolic (mmHg)", min_value=0)
    serum_albumin = st.number_input("Serum Albumin (g/dL)", min_value=0.0)
    proteinuria = st.number_input("Proteinuria Level", min_value=0.0)
    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0)
    bmi = st.number_input("BMI", min_value=0.0)
    heart_disease = st.radio("Heart Disease", [0, 1])
    portal_hypertension = st.radio("Portal Hypertension", [0, 1])
    chronic_renal_failure = st.radio("Chronic Renal Failure", [0, 1])
    overfill_edema = st.radio("Overfill Edema", [0, 1])
    congestive_heart_failure = st.radio("Congestive Heart Failure", [0, 1])

    # Step 6: Make Predictions
    if st.button("Analyze"):
        input_data = pd.DataFrame([{
            'Age': age, 'Weight_kg': weight_kg, 'Height_cm': height_cm,
            'Blood_Pressure_Systolic': blood_pressure_systolic, 
            'Blood_Pressure_Diastolic': blood_pressure_diastolic,
            'Serum_Albumin_g/dL': serum_albumin, 'Proteinuria_Level': proteinuria, 
            'Serum_Creatinine_mg/dL': serum_creatinine, 'BMI': bmi, 
            'Heart_Disease': heart_disease, 'Portal_Hypertension': portal_hypertension, 
            'Chronic_Renal_Failure': chronic_renal_failure, 'Overfill_Edema': overfill_edema,
            'Congestive_Heart_Failure': congestive_heart_failure
        }])

        # Predict probabilities
        prediction = xgb_model.predict(input_data)
        prediction_prob = xgb_model.predict_proba(input_data)
        confidence_score = max(prediction_prob[0]) * 100
        
        disease_labels = [
            'Minimal Change Disease', 'Other', 'Heart Failure', 
            'Liver Cirrhosis', 'Chronic Kidney Disease', 'Nephrotic Syndrome Edema',
            'Malnutrition', 'Congestive Heart Failure'
        ]

        # Map the predicted class to its label
        predicted_condition = disease_labels[prediction[0]]

        # Display results
        st.write("### Diagnosis")
        st.write(f"Predicted Condition: **{predicted_condition}**")
        st.write(f"Confidence: **{confidence_score:.2f}%**")

        # Explanation based on the predicted condition
        explanation_map = {
            'Minimal Change Disease': "High proteinuria and low serum albumin suggest Minimal Change Disease.",
            'Heart Failure': "Elevated blood pressure with heart disease indicates potential heart failure.",
            'Liver Cirrhosis': "Portal hypertension and hypoalbuminemia are consistent with liver cirrhosis.",
            'Chronic Kidney Disease': "Elevated serum creatinine and renal failure signs suggest Chronic Kidney Disease.",
            'Nephrotic Syndrome Edema': "Overfill edema is commonly seen in nephrotic syndrome.",
            'Malnutrition': "Low BMI and serum albumin levels indicate malnutrition.",
            'Congestive Heart Failure': "Heart disease with related fluid overload indicates congestive heart failure.",
            'Other': "The condition does not clearly match the primary categories."
        }
        st.write("### Explanation")
        st.write(explanation_map.get(predicted_condition, "No detailed explanation available."))

                # Step 7: Treatment Recommendations
        st.write("### Treatment Recommendations")
        st.write("#### Pharmacological")
        st.write("""
        - **Corticosteroids**: Often used for nephrotic syndromes to reduce inflammation.
        - **Diuretics**: To manage fluid overload in conditions like heart failure.
        - **ACE Inhibitors or ARBs**: To manage blood pressure and proteinuria.
        - **Albumin Infusion**: May be recommended for hypoalbuminemia in certain conditions.
        """)

        st.write("#### Non-Pharmacological")
        st.write("""
        - **Dietary Adjustments**: Low sodium, adequate protein intake based on condition.
        - **Fluid Management**: Fluid restriction for cases of edema or heart failure.
        - **Regular Monitoring**: Monitor weight, blood pressure, and edema severity regularly.
        """)

        # Step 8: Education for Patients and Families
        st.write("### Education for Patients and Families")
        st.write("""
        - **Disease Understanding**: Explain the diagnosis, its implications, and management options to the patient and family.
        - **Medication Adherence**: Stress the importance of taking prescribed medication as directed.
        - **Symptom Monitoring**: Educate on recognizing worsening symptoms, such as increased swelling, difficulty breathing, or changes in weight.
        - **Lifestyle Changes**: Promote a heart-healthy diet, regular physical activity (as appropriate), and smoking cessation.
        """)

        # Step 9: References
        st.write("### References")
        st.markdown("""
        **References**:
        1. [Pathophysiological Mechanisms of Edema](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10968602/)
        2. [Nephrotic Syndrome in Children](https://www.frontiersin.org/journals/pediatrics/articles/10.3389/fped.2015.00111/full)
        3. [Chronic Kidney Disease Overview](https://www.kidney.org/atoz/content/about-chronic-kidney-disease)
        4. [Heart Failure Clinical Guidelines](https://www.heart.org/en/health-topics/heart-failure)
        """, unsafe_allow_html=True)

