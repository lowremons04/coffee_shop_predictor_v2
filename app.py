# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Coffee Shop Business Intelligence",
    page_icon="☕",
    layout="wide"
)

# --- Caching the Model and Encoder ---
@st.cache_resource
def load_model_and_encoder():
    """Load the trained pipeline and label encoder from disk."""
    try:
        pipeline = joblib.load('coffee_purchase_predictor.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        return pipeline, label_encoder
    except FileNotFoundError:
        st.error("Model or encoder files not found. Make sure 'coffee_purchase_predictor.joblib' and 'label_encoder.joblib' are in the same directory.")
        return None, None

pipeline, label_encoder = load_model_and_encoder()

# --- Application Title and Description ---
st.title("☕ Coffee Shop Business Intelligence Dashboard")
st.markdown("""
Welcome to your business dashboard. This tool provides three key functions:
1.  **Single Customer Prediction:** Predict the next coffee purchase for an individual customer.
2.  **Batch Forecasting:** Upload a CSV of customer data to predict next purchases for many customers at once.
3.  **Inventory & Insights:** Get a demand forecast for the upcoming week and view historical favorites based on your uploaded data.
""")

# --- Create Tabs for different functionalities ---
tab1, tab2, tab3 = st.tabs(["👤 Single Customer Prediction", "📈 Batch Forecasting", "📊 Inventory & Insights"])


# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==============================================================================
with tab1:
    st.header("Predict Next Purchase for a Single Customer")
    
    with st.form("single_customer_form"):
        # Create columns for a cleaner layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Purchase History")
            total_visits = st.slider('Total Visits', 1, 100, 10)
            total_spent = st.slider('Total Spent (R)', 50.0, 5000.0, 500.0, step=10.0)
            days_since_last_visit = st.slider('Days Since Last Visit', 0, 365, 30)
            avg_spent_per_visit = total_spent / total_visits if total_visits > 0 else 0
            st.metric("Average Spent per Visit (R)", f"{avg_spent_per_visit:.2f}")

        with col2:
            st.subheader("Customer Habits")
            favorite_weekday = st.selectbox('Favorite Weekday', 
                                            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], key='single_weekday')
            favorite_time_of_day = st.selectbox('Favorite Time of Day', 
                                                ['Morning', 'Afternoon', 'Evening'], key='single_time')

        st.subheader("Previous Coffee Counts")
        
        # Gracefully handle if model is not loaded yet
        if pipeline is not None and label_encoder is not None:
            coffee_types = label_encoder.classes_
            # Dynamically create columns for a neat layout
            num_columns = min(len(coffee_types), 5) # Max 5 columns per row
            coffee_cols_rows = [st.columns(num_columns) for _ in range((len(coffee_types) + num_columns - 1) // num_columns)]
            coffee_counts = {}

            # Flatten the list of columns
            all_cols = [col for row in coffee_cols_rows for col in row]

            for i, coffee in enumerate(coffee_types):
                with all_cols[i]:
                    input_label = coffee
                    col_name = f'count_{coffee.lower().replace(" ", "_")}'
                    coffee_counts[col_name] = st.number_input(input_label, min_value=0, max_value=50, value=2, key=f'single_{col_name}')
        else:
            st.warning("Model not loaded. Cannot display coffee types.")
            coffee_counts = {}

        # Submit button for the form
        submitted = st.form_submit_button("Predict Next Purchase")
        
        if submitted:
            if pipeline is not None and label_encoder is not None:
                # Get the full list of required columns from the model
                required_model_cols = pipeline.named_steps['preprocessor'].feature_names_in_
                
                data = {
                    'total_visits': total_visits, 'total_spent': total_spent,
                    'days_since_last_visit': days_since_last_visit, 'avg_spent_per_visit': avg_spent_per_visit,
                    'favorite_weekday': favorite_weekday, 'favorite_time_of_day': favorite_time_of_day,
                    **coffee_counts
                }
                
                input_df = pd.DataFrame(data, index=[0])

                # Add any missing columns that the model expects but were not in the form
                for col in required_model_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Ensure the column order is correct
                input_df_aligned = input_df[required_model_cols]
                
                prediction_encoded = pipeline.predict(input_df_aligned)
                prediction_label = label_encoder.inverse_transform(prediction_encoded)
                
                st.success(f"🎉 The model predicts the customer will buy a **{prediction_label[0]}** next!")
                st.balloons()
            else:
                st.warning("Model is not loaded. Cannot make a prediction.")


# ==============================================================================
# TAB 2 & 3: BATCH FORECASTING AND INSIGHTS
# ==============================================================================
# We define a placeholder for the results to share between tabs
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None

with tab2:
    st.header("Batch Forecasting with CSV File")
    st.write("Upload a CSV file with customer data to predict their next purchases.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df_original = pd.read_csv(uploaded_file)
            st.write("✅ **CSV Uploaded Successfully!** Here's a preview:")
            st.dataframe(batch_df_original.head())

            # Get the full list of feature names the model was trained on
            required_model_cols = pipeline.named_steps['preprocessor'].feature_names_in_
            
            # Define the essential columns that MUST be in the user's uploaded file
            core_cols = ['total_visits', 'total_spent', 'days_since_last_visit', 'avg_spent_per_visit', 'favorite_weekday', 'favorite_time_of_day']
            
            # Check if all essential columns are present
            if all(col in batch_df_original.columns for col in core_cols):
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Preparing data and predicting..."):
                        
                        # --- ROBUSTNESS FIX START ---
                        # We make a copy to avoid modifying the original DataFrame in memory
                        batch_df_processed = batch_df_original.copy()

                        # Add any missing product count columns and fill with 0
                        for col in required_model_cols:
                            if col not in batch_df_processed.columns:
                                batch_df_processed[col] = 0
                        
                        # Ensure the column order is exactly what the model expects
                        batch_df_aligned = batch_df_processed[required_model_cols]
                        # --- ROBUSTNESS FIX END ---

                        predictions_encoded = pipeline.predict(batch_df_aligned)
                        predictions_labels = label_encoder.inverse_transform(predictions_encoded)
                        
                        # Add the prediction to the original DataFrame for display
                        results_df = batch_df_original.copy()
                        results_df['predicted_next_purchase'] = predictions_labels
                        st.session_state.batch_results = results_df # Save to session state
                        
                        st.success("✅ **Batch Prediction Complete!**")
                        st.dataframe(results_df)

                        # Provide a download button for the results
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=results_df.to_csv(index=False).encode('utf-8'),
                            file_name='predicted_purchases.csv',
                            mime='text/csv',
                        )
            else:
                # The improved error message
                st.error("❌ **CSV Error:** Your file is missing one or more essential columns.")
                st.write("**Essential columns required:**")
                st.json(core_cols)
                st.write("**Columns found in your file:**")
                st.json(batch_df_original.columns.tolist())
                st.info("Please check your CSV file. The very first line must be the header, and it must contain all the essential column names.")

        except Exception as e:
            st.error(f"An error occurred: {e}")

with tab3:
    st.header("Inventory Demand Forecast & Customer Insights")
    st.write("This tab uses the results from the 'Batch Forecasting' tab to generate insights.")
    
    if st.session_state.batch_results is not None:
        results_df = st.session_state.batch_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            # --- Feature 2: Stock Count Prediction ---
            st.subheader("Predicted Demand for Next Week")
            st.write("Based on the predicted next purchase for each customer in your uploaded file.")
            
            demand_forecast = results_df['predicted_next_purchase'].value_counts().reset_index()
            demand_forecast.columns = ['Coffee Type', 'Predicted Number of Sales']
            
            st.bar_chart(demand_forecast, x='Coffee Type', y='Predicted Number of Sales', color="#FF8C00")
            st.write("This chart estimates the number of units you might sell for each coffee type, assuming each customer in the batch visits once next week. Use this to guide your inventory stocking.")

        with col2:
            # --- Feature 3: Top Favorite Drinks (Historical) ---
            st.subheader("Historical Customer Favorites")
            st.write("Based on the total purchase counts from your uploaded file.")

            count_cols = [col for col in results_df.columns if col.startswith('count_')]
            if count_cols:
                historical_favorites = results_df[count_cols].sum().sort_values(ascending=False)
                historical_favorites.index = [idx.replace('count_', '').replace('_', ' ').title() for idx in historical_favorites.index]
                historical_favorites = historical_favorites.reset_index()
                historical_favorites.columns = ['Coffee Type', 'Total Historical Purchases']

                st.bar_chart(historical_favorites, x='Coffee Type', y='Total Historical Purchases', color="#008080")
                st.write("This chart shows which drinks have been the most popular historically among the customers in your uploaded file.")
            else:
                st.warning("Could not find historical purchase count columns (e.g., 'count_latte') in the uploaded file.")
    else:
        st.info("ℹ️ Please upload a CSV and run a batch prediction in the **'Batch Forecasting'** tab to see insights here.")

# In app.py, modify the loading function

@st.cache_resource
def load_all_models():
    """Load all trained models from disk."""
    try:
        pipeline = joblib.load('coffee_purchase_predictor.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        forecast_models = joblib.load('demand_forecasting_models.joblib')
        return pipeline, label_encoder, forecast_models
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e.filename}. Please ensure all .joblib files are present.")
        return None, None, None

pipeline, label_encoder, forecast_models = load_all_models()
