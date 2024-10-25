import joblib
import streamlit as st
import pandas as pd

# Load your trained model
model = joblib.load('catboost_model.pkl')  # Ensure this file exists

# Title for the web app
st.title("Marketing Campaign Conversion Prediction App")

# Create inputs for the user to provide the features
st.write("Please provide the following input features for prediction:")

age = st.number_input("Age", min_value=0, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Income", min_value=0, value=50000)
campaign_channel = st.selectbox("Campaign Channel", ["Email", "Social Media", "PPC", "Referral", "SEO"])
campaign_type = st.selectbox("Campaign Type", ["Awareness", "Consideration", "Retention", "Conversion"])
ad_spend = st.number_input("Ad Spend", min_value=0, value=1000)
click_through_rate = st.number_input("Click Through Rate", min_value=0.0, max_value=1.0, value=0.05)
conversion_rate = st.number_input("Conversion Rate", min_value=0.0, max_value=1.0, value=0.02)
website_visits = st.number_input("Website Visits", min_value=0, value=200)
pages_per_visit = st.number_input("Pages per Visit", min_value=0.0, max_value=10.0, value=3.5)
time_on_site = st.number_input("Time on Site (in seconds)", min_value=0, value=120)
social_shares = st.number_input("Social Shares", min_value=0, value=10)
email_opens = st.number_input("Email Opens", min_value=0, value=50)
email_clicks = st.number_input("Email Clicks", min_value=0, value=10)
previous_purchases = st.number_input("Previous Purchases", min_value=0, value=5)
loyalty_points = st.number_input("Loyalty Points", min_value=0, value=200)

# Create a dictionary to store the input data
input_data = {
    'Age': age,
    'Gender': gender,
    'Income': income,
    'CampaignChannel': campaign_channel,
    'CampaignType': campaign_type,
    'AdSpend': ad_spend,
    'ClickThroughRate': click_through_rate,
    'ConversionRate': conversion_rate,
    'WebsiteVisits': website_visits,
    'PagesPerVisit': pages_per_visit,
    'TimeOnSite': time_on_site,
    'SocialShares': social_shares,
    'EmailOpens': email_opens,
    'EmailClicks': email_clicks,
    'PreviousPurchases': previous_purchases,
    'LoyaltyPoints': loyalty_points
}

# Convert the dictionary to a DataFrame for prediction
input_df = pd.DataFrame([input_data])

# One-hot encode the categorical columns to match the model
input_df = pd.get_dummies(input_df)

# Ensure all expected columns are present by adding missing columns with default value 0
expected_columns = model.feature_names_
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder the columns to match the model's expected feature order
input_df = input_df[expected_columns]

# Make the prediction when the user clicks the "Predict" button
if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write(f"The predicted conversion is: {prediction[0]}")
