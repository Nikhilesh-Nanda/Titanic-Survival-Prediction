import streamlit as st
import pandas as pd
import pickle

# Load the trained model
# Ensure 'titanic_survival_model.pkl' is in the same directory as this app.py file
try:
    with open('titanic_survival_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'titanic_survival_model.pkl' not found. Please ensure the model file is in the same directory.")
    st.stop()

# Define preprocessing functions (matching the notebook's preprocessing)
def map_age_group(val):
    if pd.isna(val): # Handle potential NaN, though age is filled in training
        return None
    val = int(val) # Age was floored in preprocessing
    if val >= 60: return "O"
    elif (val > 35) and (val < 60): return "M"
    elif val <= 35: return "Y"

# Streamlit App Title
st.title('Titanic Survival Prediction')
st.write('Enter passenger details to predict survival.')

# Input widgets for features
with st.sidebar:
    st.header('Passenger Details')
    pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3], format_func=lambda x: f'Class {x}')
    sex_input = st.radio('Sex', ['male', 'female'])
    age = st.slider('Age', 0.0, 80.0, 29.0, step=1.0)
    sibsp = st.number_input('Number of Siblings/Spouses Aboard (SibSp)', 0, 8, 0)
    parch = st.number_input('Number of Parents/Children Aboard (Parch)', 0, 6, 0)
    fare = st.number_input('Fare (Ticket Price)', 0.0, 512.0, 32.0, step=0.1)
    embarked_input = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'], format_func=lambda x: {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}[x])

# Preprocess inputs to match model training
sex = 0 if sex_input == 'male' else 1
family_size = sibsp + parch

# Map Embarked to numerical (0, 1, 2)
embarked_map = {'S': 0.0, 'C': 1.0, 'Q': 2.0} # Note: In the notebook, it was `{'S':0,'C':1,'Q':2}` but `my_data['Embarked']` became float after fillna. Use float for consistency.
embarked = embarked_map[embarked_input]

# Map AgeGroup to numerical (1, 2, 3)
age_group_str = map_age_group(age)
age_group_map = {'Y': 1, 'M': 2, 'O': 3}
age_group = age_group_map.get(age_group_str, 1) # Default to 'Y' if None

# Create a DataFrame for prediction (ensure column order and types match training data)
input_data = pd.DataFrame([[pclass, sex, int(age), family_size, fare, embarked, age_group]],
                            columns=['Pclass', 'Sex', 'Age', 'familySize', 'Fare', 'Embarked', 'AgeGroup'])

# Make prediction
if st.button('Predict Survival'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.success(f"The passenger is likely to survive! (Probability: {prediction_proba[0][1]:.2f})")
    else:
        st.error(f"The passenger is likely \'not\' to survive. (Probability: {prediction_proba[0][0]:.2f})")

st.write("**Note:** 0 = Not Survived, 1 = Survived")
st.write("0 = Male, 1 = Female")
