import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# Title
st.title("Spare Parts Sales Prediction Dashboard")

# Step 1: Collect data from ERP system
st.header("1. Data Collection")
uploaded_file = st.file_uploader("Upload 'spare_parts_sales_with_issues.csv'", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("**Data Info:**")
    st.write(df.info())
    st.write("**First 5 rows:**")
    st.write(df.head())
    st.write(f"**Total records:** {len(df)}")

    # Step 2: Clean and normalize data
    st.header("2. Data Cleaning and Normalization")
    df['UnitsSold'] = df.groupby('PartName')['UnitsSold'].transform(lambda x: x.fillna(x.median()))
    df['Temperature'] = df.groupby('Region')['Temperature'].transform(lambda x: x.fillna(x.median()))
    duplicates = df.duplicated(subset=['PartID', 'PartName', 'Region', 'Week']).sum()
    df['Week'] = pd.to_datetime(df['Week'], format='%m/%d/%y')
    df['PartName_Encoded'] = LabelEncoder().fit_transform(df['PartName'])
    df['Region_Encoded'] = LabelEncoder().fit_transform(df['Region'])
    scaler = MinMaxScaler()
    df[['UnitsSold', 'Temperature']] = scaler.fit_transform(df[['UnitsSold', 'Temperature']])
    st.write(f"**Duplicates:** {duplicates}")
    st.write("**Cleaned Data Info:**")
    st.write(df.info())
    df.to_csv('cleaned_spare_parts_sales.csv', index=False)
    st.download_button("Download cleaned data", data=open('cleaned_spare_parts_sales.csv', 'rb'), file_name='cleaned_spare_parts_sales.csv')

    # Step 3: Analyze seasonal trends
    st.header("3. Seasonal Trends Analysis")
    df['Month'] = df['Week'].dt.month
    partname_counts = df['PartName'].value_counts().to_dict()
    region_counts = df['Region'].value_counts().to_dict()
    monthly_sales = df.groupby('Month')['UnitsSold'].mean().to_dict()
    correlation = df[['UnitsSold', 'Temperature', 'IsHolidayWeek', 'WarrantyClaims']].corr()['UnitsSold']
    st.write("**PartName Distribution:**", partname_counts)
    st.write("**Region Distribution:**", region_counts)
    st.write("**Average UnitsSold by Month:**", monthly_sales)
    st.write("**Correlation with UnitsSold:**", correlation.to_dict())

    # Step 4: Visualize using Chart.js (mimicking Power BI)
    st.header("4. Visualizations")
    charts = [
        {
            "id": "partname_chart",
            "config": {
                "type": "pie",
                "data": {
                    "labels": ["Brake Pad", "Oil Filter", "Battery", "Air Filter", "Spark Plug"],
                    "datasets": [{
                        "label": "PartName",
                        "data": [50, 48, 40, 32, 30],
                        "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF"],
                        "borderColor": "#FFFFFF",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "plugins": {
                        "title": {"display": true, "text": "PartName Distribution"},
                        "legend": {"position": "right"}
                    }
                }
            }
        },
        {
            "id": "units_sold_chart",
            "config": {
                "type": "bar",
                "data": {
                    "labels": ["0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"],
                    "datasets": [{
                        "label": "Count",
                        "data": [14, 23, 25, 24, 26, 28, 22, 19, 14, 5],
                        "backgroundColor": "#36A2EB",
                        "borderColor": "#FFFFFF",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {"beginAtZero": true, "title": {"display": true, "text": "Count"}},
                        "x": {"title": {"display": true, "text": "UnitsSold (Normalized)"}}
                    },
                    "plugins": {"title": {"display": true, "text": "UnitsSold Distribution"}}
                }
            }
        },
        {
            "id": "region_chart",
            "config": {
                "type": "bar",
                "data": {
                    "labels": ["East", "North", "South", "West"],
                    "datasets": [{
                        "label": "Average UnitsSold",
                        "data": [0.48, 0.45, 0.47, 0.44],
                        "backgroundColor": "#FFCE56",
                        "borderColor": "#FFFFFF",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "y": {"beginAtZero": true, "title": {"display": true, "text": "Average UnitsSold"}},
                        "x": {"title": {"display": true, "text": "Region"}}
                    },
                    "plugins": {"title": {"display": true, "text": "Average UnitsSold by Region"}}
                }
            }
        },
        {
            "id": "month_chart",
            "config": {
                "type": "line",
                "data": {
                    "labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    "datasets": [{
                        "label": "Average UnitsSold",
                        "data": [0.46, 0.43, 0.45, 0.47, 0.46, 0.48, 0.47, 0.46, 0.47, 0.46, 0.50, 0.45],
                        "backgroundColor": "#4BC0C0",
                        "borderColor": "#4BC0C0",
                        "fill": false
                    }]
                },
                "options": {
                    "scales": {
                        "y": {"beginAtZero": true, "title": {"display": true, "text": "Average UnitsSold"}},
                        "x": {"title": {"display": true, "text": "Month"}}
                    },
                    "plugins": {"title": {"display": true, "text": "UnitsSold by Month"}}
                }
            }
        },
        {
            "id": "temp_chart",
            "config": {
                "type": "scatter",
                "data": {
                    "datasets": [{
                        "label": "UnitsSold vs Temperature",
                        "data": [
                            {"x": 0.94, "y": 0.02}, {"x": 0.12, "y": 0.04}, {"x": 0.52, "y": 0.78}, {"x": 0.05, "y": 0.32},
                            {"x": 0.93, "y": 0.76}, {"x": 0.39, "y": 0.27}, {"x": 0.58, "y": 0.13}, {"x": 0.35, "y": 0.57}
                        ],
                        "backgroundColor": "#9966FF",
                        "borderColor": "#FFFFFF",
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "x": {"title": {"display": true, "text": "Temperature (Normalized)"}},
                        "y": {"title": {"display": true, "text": "UnitsSold (Normalized)"}, "beginAtZero": true}
                    },
                    "plugins": {"title": {"display": true, "text": "UnitsSold vs Temperature"}}
                }
            }
        }
    ]
    for chart in charts:
        st.components.v1.html(f"""
        <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
        <canvas id='{chart["id"]}'></canvas>
        <script>
        const ctx = document.getElementById('{chart["id"]}').getContext('2d');
        new Chart(ctx, {json.dumps(chart["config"])});
        </script>
        """, height=400)

    # Step 5: Train Linear Regression model
    st.header("5. Model Training")
    X = df[['Temperature', 'IsHolidayWeek', 'WarrantyClaims', 'PartName_Encoded', 'Region_Encoded', 'Month']]
    y = df['UnitsSold']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("**Coefficients:**", model.coef_)
    st.write("**Intercept:**", model.intercept_)

    # Step 6: Evaluate accuracy using MAE and RMSE
    st.header("6. Model Evaluation")
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")

    # Prediction interface
    st.header("Predict UnitsSold")
    temp = st.slider("Temperature (Normalized)", 0.0, 1.0, 0.5)
    holiday = st.selectbox("IsHolidayWeek", [0, 1])
    warranty = st.slider("WarrantyClaims", 0, 4, 2)
    part = st.selectbox("PartName", df['PartName'].unique())
    region = st.selectbox("Region", df['Region'].unique())
    month = st.selectbox("Month", range(1, 13))
    input_data = pd.DataFrame({
        'Temperature': [temp],
        'IsHolidayWeek': [holiday],
        'WarrantyClaims': [warranty],
        'PartName_Encoded': [df[df['PartName'] == part]['PartName_Encoded'].iloc[0]],
        'Region_Encoded': [df[df['Region'] == region]['Region_Encoded'].iloc[0]],
        'Month': [month]
    })
    prediction = model.predict(input_data)[0]
    st.write(f"**Predicted UnitsSold (Normalized):** {prediction:.4f}")
else:
    st.warning("Please upload 'spare_parts_sales_with_issues.csv' to proceed.")
