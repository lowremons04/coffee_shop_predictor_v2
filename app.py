# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from prophet import Prophet # Required for the new forecasting tab

# --- Page Configuration ---
st.set_page_config(
    page_title="Coffee Shop Business Intelligence",
    page_icon="☕",
    layout="wide"
)

# --- Caching Models ---
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

# --- Application Title and Description ---
st.title("☕ Coffee Shop Business Intelligence Dashboard")
st.markdown("""
Welcome to your business dashboard. This tool provides predictive insights and forecasting to help you manage your coffee shop more effectively.
""")

# --- Create Tabs for different functionalities ---
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Single Customer Prediction", 
    "📈 Batch Forecasting", 
    "📊 Customer Insights", 
    "📅 Demand Forecast"
])


# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==============================================================================
with tab1:
    st.header("Predict Next Purchase for a Single Customer")
    
    with st.form("single_customer_form"):
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
        if pipeline is not None and label_encoder is not None:
            coffee_types = label_encoder.classes_
            num_columns = min(len(coffee_types), 5)
            coffee_cols_rows = [st.columns(num_columns) for _ in range((len(coffee_types) + num_columns - 1) // num_columns)]
            coffee_counts = {}
            all_cols = [col for row in coffee_cols_rows for col in row]
            for i, coffee in enumerate(coffee_types):
                with all_cols[i]:
                    col_name = f'count_{coffee.lower().replace(" ", "_")}'
                    coffee_counts[col_name] = st.number_input(coffee, min_value=0, max_value=50, value=2, key=f'single_{col_name}')
        else:
            st.warning("Model not loaded. Cannot display coffee types.")
            coffee_counts = {}

        submitted = st.form_submit_button("Predict Next Purchase")
        if submitted:
            if pipeline is not None and label_encoder is not None:
                required_model_cols = pipeline.named_steps['preprocessor'].feature_names_in_
                data = {'total_visits': total_visits, 'total_spent': total_spent, 'days_since_last_visit': days_since_last_visit, 
                        'avg_spent_per_visit': avg_spent_per_visit, 'favorite_weekday': favorite_weekday, 
                        'favorite_time_of_day': favorite_time_of_day, **coffee_counts}
                input_df = pd.DataFrame(data, index=[0])
                for col in required_model_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
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
# Sidebar for "What-If" Scenarios
st.sidebar.title("📈 What-If Scenario Simulator")
st.sidebar.write("Simulate the effect of a weekly promotion on customer predictions in the 'Batch Forecasting' tab.")

promo_drink = "None"
if label_encoder:
    coffee_types_for_promo = sorted([c for c in label_encoder.classes_])
    promo_drink = st.sidebar.selectbox("Select a drink to promote:", options=["None"] + coffee_types_for_promo)

boost_amount = 0
if promo_drink != "None":
    boost_amount = st.sidebar.slider(f"Promotion Strength (add to '{promo_drink}' count):", min_value=1, max_value=10, value=3)

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
            
            required_model_cols = pipeline.named_steps['preprocessor'].feature_names_in_
            core_cols = ['total_visits', 'total_spent', 'days_since_last_visit', 'avg_spent_per_visit', 'favorite_weekday', 'favorite_time_of_day']
            
            if all(col in batch_df_original.columns for col in core_cols):
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Preparing data and predicting..."):
                        batch_df_processed = batch_df_original.copy()

                        # --- Apply the Promotional Boost ---
                        if promo_drink != "None":
                            promo_col_name = f"count_{promo_drink.lower().replace(' ', '_')}"
                            if promo_col_name in batch_df_processed.columns:
                                st.info(f"Applying a promotional boost of +{boost_amount} to '{promo_drink}'.")
                                batch_df_processed[promo_col_name] = batch_df_processed[promo_col_name] + boost_amount
                            else:
                                st.warning(f"Could not apply promotion. Column '{promo_col_name}' not found.")

                        for col in required_model_cols:
                            if col not in batch_df_processed.columns:
                                batch_df_processed[col] = 0
                        batch_df_aligned = batch_df_processed[required_model_cols]

                        predictions_encoded = pipeline.predict(batch_df_aligned)
                        predictions_labels = label_encoder.inverse_transform(predictions_encoded)
                        
                        results_df = batch_df_original.copy()
                        results_df['predicted_next_purchase'] = predictions_labels
                        st.session_state.batch_results = results_df
                        
                        st.success("✅ **Batch Prediction Complete!**")
                        st.dataframe(results_df)
                        st.download_button(label="Download Predictions as CSV", data=results_df.to_csv(index=False).encode('utf-8'),
                                           file_name='predicted_purchases.csv', mime='text/csv')
            else:
                st.error("❌ **CSV Error:** Your file is missing one or more essential columns.")
                st.write("**Essential columns required:**"); st.json(core_cols)
                st.write("**Columns found in your file:**"); st.json(batch_df_original.columns.tolist())
                st.info("Please check your CSV file's header.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with tab3:
    st.header("Customer Insights from Batch Data")
    st.write("This tab uses the results from the 'Batch Forecasting' tab to generate insights.")
    if st.session_state.batch_results is not None:
        results_df = st.session_state.batch_results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted Demand (from Batch)")
            st.write("Based on the predicted next purchase for each customer in your uploaded file.")
            demand_forecast = results_df['predicted_next_purchase'].value_counts().reset_index()
            demand_forecast.columns = ['Coffee Type', 'Predicted Number of Sales']
            st.bar_chart(demand_forecast, x='Coffee Type', y='Predicted Number of Sales', color="#FF8C00")

        with col2:
            st.subheader("Historical Customer Favorites")
            st.write("Based on the total purchase counts from your uploaded file.")
            count_cols = [col for col in results_df.columns if col.startswith('count_')]
            if count_cols:
                historical_favorites = results_df[count_cols].sum().sort_values(ascending=False)
                historical_favorites.index = [idx.replace('count_', '').replace('_', ' ').title() for idx in historical_favorites.index]
                historical_favorites = historical_favorites.reset_index()
                historical_favorites.columns = ['Coffee Type', 'Total Historical Purchases']
                st.bar_chart(historical_favorites, x='Coffee Type', y='Total Historical Purchases', color="#008080")
            else:
                st.warning("Could not find historical purchase count columns (e.g., 'count_latte') in the uploaded file.")
    else:
        st.info("ℹ️ Please upload a CSV and run a batch prediction in the **'Batch Forecasting'** tab to see insights here.")

# ==============================================================================
# TAB 4: DEMAND FORECAST
# ==============================================================================
with tab4:
    st.header("📅 Weekly Demand Forecast")
    st.write("Predict the total number of units sold for each product over the next week using a time series model.")

    if forecast_models:
        product_to_forecast = st.selectbox("Select a Coffee Type to Forecast:", options=sorted(forecast_models.keys()))
        forecast_days = st.slider("Days to forecast", 7, 30, 7)

        if st.button(f"Forecast {product_to_forecast} Sales", type="primary"):
            with st.spinner("Generating forecast..."):
                model = forecast_models[product_to_forecast]
                future = model.make_future_dataframe(periods=forecast_days)
                forecast = model.predict(future)
                
                st.success("✅ Forecast Generated!")
                st.subheader(f"Forecast Plot for {product_to_forecast}")
                fig = model.plot(forecast)
                ax = fig.gca()
                ax.set_title(f"Sales Forecast for {product_to_forecast}", size=20)
                ax.set_xlabel("Date", size=15)
                ax.set_ylabel("Predicted Sales", size=15)
                st.pyplot(fig)

                st.subheader("Forecast Data")
                forecast_data_display = forecast[forecast['ds'] >= pd.to_datetime('today').normalize()]
                st.dataframe(
                    forecast_data_display[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                        'ds': 'Date', 'yhat': 'Predicted Sales',
                        'yhat_lower': 'Lower Estimate', 'yhat_upper': 'Upper Estimate'
                    }).set_index('Date')
                )
    else:
        st.warning("Forecasting models are not loaded. Ensure 'demand_forecasting_models.joblib' is present.")
