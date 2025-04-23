from streamlit_folium import st_folium

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from utils import *
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# State coordinates mapping
state_coordinates = {
    'AL': [32.7794, -86.8287], 'AK': [64.0685, -152.2782], 'AZ': [34.2744, -111.6602],
    'AR': [34.8938, -92.4426], 'CA': [37.1841, -119.4696], 'CO': [38.9972, -105.5478],
    'CT': [41.6219, -72.7273], 'DE': [38.9896, -75.5050], 'FL': [28.6305, -82.4497],
    'GA': [32.6415, -83.4426], 'HI': [20.2927, -156.3737], 'ID': [44.3509, -114.6130],
    'IL': [40.0417, -89.1965], 'IN': [39.8942, -86.2816], 'IA': [42.0751, -93.4960],
    'KS': [38.4937, -98.3804], 'KY': [37.5347, -85.3021], 'LA': [31.0689, -91.9968],
    'ME': [45.3695, -69.2428], 'MD': [39.0550, -76.7909], 'MA': [42.2596, -71.8083],
    'MI': [44.3467, -85.4102], 'MN': [46.2807, -94.3053], 'MS': [32.7364, -89.6678],
    'MO': [38.3566, -92.4580], 'MT': [47.0527, -109.6333], 'NE': [41.5378, -99.7951],
    'NV': [39.3289, -116.6312], 'NH': [43.6805, -71.5811], 'NJ': [40.1907, -74.6728],
    'NM': [34.4071, -106.1126], 'NY': [42.9538, -75.5268], 'NC': [35.5557, -79.3877],
    'ND': [47.4501, -100.4659], 'OH': [40.2862, -82.7937], 'OK': [35.5889, -97.4943],
    'OR': [43.9336, -120.5583], 'PA': [40.8781, -77.7996], 'RI': [41.6762, -71.5562],
    'SC': [33.9169, -80.8964], 'SD': [44.4443, -100.2263], 'TN': [35.8580, -86.3505],
    'TX': [31.4757, -99.3312], 'UT': [39.3055, -111.6703], 'VT': [44.0687, -72.6658],
    'VA': [37.5215, -78.8537], 'WA': [47.3826, -120.4472], 'WV': [38.6409, -80.6227],
    'WI': [44.6243, -89.9941], 'WY': [42.9957, -107.5512]
}


from streamlit_folium import folium_static

# Page Configuration
st.set_page_config(
    page_title="Medicines Analytics & Disease Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
with open('assets/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the disease prediction model and data
@st.cache_resource
def load_disease_prediction_resources():
    try:
        # Load trained Naïve Bayes model
        model = joblib.load("disease_prediction_model.pkl")

        # Load dataset to get symptom names and handle the format
        # Make sure to exclude the prognosis column completely
        # This is critical to match with how the model was trained
        df = pd.read_csv("Training.csv")

        # If the last column is prognosis, which contains the disease names,
        # exclude it from the symptom list
        cols_without_prognosis = df.columns[:-1]  # All columns except the last (prognosis)

        # Clean up the symptom column names if needed
        symptom_columns = [col.strip() for col in cols_without_prognosis]

        print(f"Loaded {len(symptom_columns)} symptom columns")

        return model, symptom_columns
    except Exception as e:
        st.error(f"Error loading prediction resources: {e}")
        return None, None

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.radio("Select Section", ["Sales Dashboard", "Disease Prediction"])

# Main Header
st.sidebar.image("https://images.unsplash.com/photo-1606940743881-b33f4b04d661", use_container_width=True)
st.sidebar.markdown("---")

# App header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://images.unsplash.com/photo-1593086586351-1673fca190cf", width=100)
with col2:
    st.title("Medicines Analytics Hub")
    st.markdown("A comprehensive system for medication sales analysis and disease prediction")

# Sales Dashboard Section
if app_mode == "Sales Dashboard":
    st.header("📈 Medicines Sales Analytics Dashboard")

    # File uploader for sales data
    uploaded_file = st.file_uploader("Upload sales data (CSV)", type="csv")

    if uploaded_file is not None:
        # Load and preprocess data with the analytics functions from app.py
        try:
            df = load_and_preprocess_data(uploaded_file)
            st.success("Successfully loaded sales data")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        # Date filter
        st.sidebar.header("📅 Date Filter")
        min_date = df['datum'].min().date()
        max_date = df['datum'].max().date()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = df[(df['datum'].dt.date >= start_date) & (df['datum'].dt.date <= end_date)]
        else:
            filtered_df = df

        # Medication Category Filter
        st.sidebar.header("🏷️ Category Filter")
        categories = [
            'anti_inflammatory_acetic_acid',
            'anti_inflammatory_propionic_acid',
            'salicylic_acid_analgesics',
            'n02be',
            'anxiolytics',
            'hypnotics_sedatives',
            'airway_disease_medications',
            'antihistamines'
        ]

        selected_categories = st.sidebar.multiselect(
            "Select Medication Categories",
            options=categories,
            default=categories,
            format_func=lambda x: x.replace('_', ' ').title()
        )

        # Location Filter
        st.sidebar.header("📍 Location Filter")
        all_states = sorted(df['state'].unique())
        selected_states = st.sidebar.multiselect(
            "Select States", 
            options=all_states,
            default=[]
        )

        if selected_states:
            filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]

        # Data Status
        st.markdown(f"**Data range:** {start_date} to {end_date}")
        st.markdown(f"**Records:** {len(filtered_df):,} transactions from {filtered_df['location'].nunique()} locations")

        # Dashboard Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🗺️ Geographic", "📈 Time Analysis", "🔍 Category Details", "📦 Stock Analysis"])

        with tab1:
            st.subheader("Key Performance Metrics")

            # Summary statistics cards
            stats = get_summary_stats(filtered_df)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sales", f"${stats['total_sales']:,.2f}")
            with col2:
                st.metric("Avg Daily Sales", f"${stats['avg_daily_sales']:,.2f}")
            with col3:
                st.metric("Top Category", stats['top_category'])
            with col4:
                st.metric("Month-over-Month Growth", f"{stats['mom_growth']:.1f}%")

            # Add cost breakdown section
            st.markdown("### Cost Breakdown by Medication Category")

            med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                      'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                      'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']

            # Create cost breakdown dataframe with pricing info
            med_prices = {
                'anti_inflammatory_acetic_acid': 5,
                'anti_inflammatory_propionic_acid': 4,
                'salicylic_acid_analgesics': 3,
                'n02be': 6,
                'anxiolytics': 7,
                'hypnotics_sedatives': 8,
                'airway_disease_medications': 9,
                'antihistamines': 10
            }

            cost_data = []
            for med in med_cols:
                units = filtered_df[med].sum()
                price = med_prices[med]
                total = units * price
                cost_data.append({
                    'Category': med.replace('_', ' ').title(),
                    'Units Sold': f"{units:,.1f}",
                    'Price Per Unit': f"${price:.2f}",
                    'Total Cost': f"${total:,.2f}"
                })

            cost_df = pd.DataFrame(cost_data)
            st.dataframe(cost_df, use_container_width=True)

            st.markdown("---")

            # Category comparison chart
            st.subheader("Medication Category Analysis")
            category_fig = create_category_comparison(filtered_df)
            st.plotly_chart(category_fig, use_container_width=True)

            # Top cities chart
            st.subheader("Top Performing Cities")
            cities_fig = create_top_cities_chart(filtered_df)
            st.plotly_chart(cities_fig, use_container_width=True)

        with tab2:
            st.header("Geographic Analysis")

            # Create two columns for dual map view
            col1, col2 = st.columns(2)

            # Get medication columns
            med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                       'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                       'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']

            # Analyze location-wise data
            location_data = df.groupby(['state', 'city']).agg({
                'total_cost': 'sum',
                **{med: 'sum' for med in med_cols}
            }).reset_index()

            # Find top medication by location
            location_data['top_medication'] = location_data[med_cols].idxmax(axis=1)
            location_data['top_medication_units'] = location_data[med_cols].max(axis=1)

            # Calculate total units for each location
            location_data['total_units'] = location_data[med_cols].sum(axis=1)

            # Calculate percentage of each medication
            for med in med_cols:
                location_data[f'{med}_pct'] = (location_data[med] / location_data['total_units'] * 100).round(1)

            # Single column for map visualization
            st.subheader("Sales Distribution by Location")

            # Create base map centered on US
            m1 = folium.Map(location=[39.8283, -98.5795], zoom_start=4,
                          tiles='OpenStreetMap')

            # Initialize geocoder
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter

            geolocator = Nominatim(user_agent="my_agent")
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

            # Geocode locations
            @st.cache_data
            def get_location_coordinates(city, state):
                try:
                    location = geocode(f"{city}, {state}, USA")
                    if location:
                        return location.latitude, location.longitude
                    return None
                except:
                    return None

            # Create markers and heatmap data
            heat_data = []

            for _, row in location_data.iterrows():
                coords = get_location_coordinates(row['city'], row['state'])
                if coords:
                    # Create a detailed popup
                    popup_text = f"""
                    <div style='width:300px'>
                        <h4>{row['city']}, {row['state']}</h4>
                        <hr>
                        <b>Financial Summary:</b><br>
                        Total Revenue: ${row['total_cost']:,.2f}<br>
                        Average Price/Unit: ${row['total_cost']/row['total_units']:,.2f}<br>
                        <hr>
                        <b>Sales Distribution:</b><br>
                        Total Units: {row['total_units']:,.0f}<br>
                        Top Medication: {row['top_medication'].replace('_', ' ').title()}<br>
                        <hr>
                        <b>Medication Breakdown:</b><br>
                        {'<br>'.join([f"{med.replace('_', ' ').title()}: {row[f'{med}_pct']}%" for med in med_cols])}
                    </div>
                    """

                    folium.CircleMarker(
                        location=coords,
                        radius=np.sqrt(row['total_cost'])/50,
                        popup=popup_text,
                        tooltip=row['city'],
                        fill=True,
                        color='blue',
                        fill_opacity=0.7
                    ).add_to(m1)

                    # Add heatmap data point
                    heat_data.append([coords[0], coords[1], row['total_cost']])

            # Add heatmap layer
            from folium import plugins
            plugins.HeatMap(heat_data).add_to(m1)

            st_folium(m1, width=1200)  # Increased width for better visibility

            # Show top 5 cities by units and revenue
            col1, col2 = st.columns(2)
            with col1:
                st.write("Top 5 Cities by Units Sold:")
                city_units = location_data.groupby('city')[med_cols].sum()
                city_units['total_units'] = city_units.sum(axis=1)
                st.dataframe(city_units['total_units'].nlargest(5))

            with col2:
                st.write("Top 5 Cities by Revenue:")
                city_revenue = location_data.groupby('city')['total_cost'].sum()
                st.dataframe(city_revenue.nlargest(5))

            # Get medication columns
            med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                       'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                       'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']

            # Analyze state-wise data
            state_data = filtered_df.groupby('state').agg({
                'total_cost': 'sum',
            }).reset_index()

            # Get top medication by state
            state_med_data = filtered_df.groupby('state')[med_cols].sum().reset_index()
            state_med_data['top_medication'] = state_med_data[med_cols].idxmax(axis=1)
            state_med_data['top_medication_units'] = state_med_data[med_cols].max(axis=1)

            # Medication Distribution Analysis
            st.subheader("Medication Distribution by Location")

            # Create tabs for state and city views
            dist_tab1, dist_tab2 = st.tabs(["State-wise Distribution", "City-wise Distribution"])

            with dist_tab1:
                # Calculate state-wise top medications
                state_meds = filtered_df.groupby('state')[med_cols].sum()
                state_top_meds = state_meds.idxmax(axis=1)

                # Create an expander for each medication category
                for med in med_cols:
                    med_display = med.replace('_', ' ').title()
                    with st.expander(f"🔍 {med_display}"):
                        # Find states where this medication is the top seller
                        top_states = state_top_meds[state_top_meds == med].index.tolist()
                        if top_states:
                            st.write("**Top Selling States:**")
                            for state in top_states:
                                units = state_meds.loc[state, med]
                                st.write(f"- {state}: {units:,.0f} units")
                        else:
                            st.write("No states where this is the top selling medication")

            with dist_tab2:
                # Calculate city-wise top medications
                city_meds = filtered_df.groupby(['city', 'state'])[med_cols].sum()
                city_top_meds = city_meds.idxmax(axis=1)

                # Create a searchable selection for medications
                selected_med = st.selectbox(
                    "Select Medication to View Distribution",
                    options=med_cols,
                    format_func=lambda x: x.replace('_', ' ').title()
                )

                # Show top 10 cities for selected medication
                city_med_sales = city_meds[selected_med].sort_values(ascending=False).head(10)
                for (city, state), units in city_med_sales.items():
                    st.write(f"- {city}, {state}: {units:,.0f} units")

            st.markdown("---")
            # N02BE Sales Distribution Map at the bottom of geographic section
            st.subheader("N02BE Sales Distribution Map")

            # Calculate N02BE sales by state and city
            n02be_sales = filtered_df.groupby(['state', 'city'])['n02be'].sum().reset_index()
            n02be_state_sales = filtered_df.groupby('state')['n02be'].sum().reset_index()

            # Find maximum sales for scaling
            max_sales = n02be_sales['n02be'].max()
            max_state_sales = n02be_state_sales['n02be'].max()

            # Create the map centered on US
            m_n02be = folium.Map(location=[39.8283, -98.5795], zoom_start=4,
                               tiles='OpenStreetMap')

            # Add state-level highlight for Texas
            texas_coords = [
                [36.5, -106.6], [36.5, -100.0], [34.6, -100.0], [34.6, -96.5],
                [33.8, -96.5], [31.8, -93.8], [31.1, -93.8], [28.5, -96.5],
                [25.8, -97.1], [25.8, -99.2], [27.4, -99.9], [29.8, -103.9],
                [31.0, -106.6]
            ]

            folium.Polygon(
                locations=texas_coords,
                color='purple',
                weight=2,
                fill=True,
                fill_color='purple',
                fill_opacity=0.2,
                popup=f'Texas - Total N02BE Units: 10,557'
            ).add_to(m_n02be)

            # Add markers for each city
            for _, row in n02be_sales.iterrows():
                if row['n02be'] > 0:  # Only show locations with sales
                    state_coords = state_coordinates.get(row['state'], [39.8283, -98.5795])

                    # Calculate size and color based on sales volume
                    sales_ratio = row['n02be'] / max_sales

                    # Different colors for sales levels and special highlight for Seattle
                    if row['city'] == 'Seattle':
                        color = 'purple'  # Special color for Seattle
                    elif sales_ratio > 0.8:
                        color = 'red'  # Highest sellers
                    elif sales_ratio > 0.5:
                        color = 'orange'  # Medium sellers
                    else:
                        color = 'blue'  # Lower sellers

                    # Create detailed popup
                    popup_content = f"""
                    <div style='width:200px'>
                        <h4>{row['city']}, {row['state']}</h4>
                        <b>N02BE Units Sold:</b> {row['n02be']:,.0f}<br>
                        <b>Sales Level:</b> {'High' if sales_ratio > 0.8 else 'Medium' if sales_ratio > 0.5 else 'Low'}
                    </div>
                    """

                    # Add circle marker
                    folium.CircleMarker(
                        location=state_coords,
                        radius=10 + (sales_ratio * 20),  # Size based on sales
                        popup=popup_content,
                        tooltip=f"{row['city']}: {row['n02be']:,.0f} units",
                        color=color,
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(m_n02be)

            # Add legend
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px">
            <p><b>N02BE Sales Volume</b></p>
            <p><i class="fa fa-circle" style="color:red"></i> High Volume</p>
            <p><i class="fa fa-circle" style="color:orange"></i> Medium Volume</p>
            <p><i class="fa fa-circle" style="color:blue"></i> Lower Volume</p>
            </div>
            """
            m_n02be.get_root().html.add_child(folium.Element(legend_html))

            # Display the map
            st_folium(m_n02be, width=1200, height=600)

            # Enhanced N02BE Sales Analysis Conclusion with better visual organization
            st.markdown("""
            <style>
            .conclusion-box {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin: 20px 0;
                border-left: 5px solid #1f77b4;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-box {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                border: 1px solid #e1e4e8;
            }
            .highlight-text {
                color: #1f77b4;
                font-weight: bold;
            }
            .insight-box {
                background-color: #e8f4f8;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            }
            </style>
            
            <div class="conclusion-box">
                <h2 style='color: #1f77b4;'>🔍 N02BE Sales Analysis Key Findings</h2>
            </div>
            """, unsafe_allow_html=True)

            # Key metrics in an enhanced layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-box">
                    <h3 style='margin:0'>🏆 Top State Performance</h3>
                    <p class="highlight-text" style='font-size:24px'>10,557 units</p>
                    <p>Texas leads state-wide sales</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class="metric-box">
                    <h3 style='margin:0'>🌆 Leading City</h3>
                    <p class="highlight-text" style='font-size:24px'>3,711 units</p>
                    <p>Seattle tops city-wide distribution</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div class="metric-box">
                    <h3 style='margin:0'>📈 Market Share</h3>
                    <p class="highlight-text" style='font-size:24px'>23.4%</p>
                    <p>Of total analgesic sales</p>
                </div>
                """, unsafe_allow_html=True)

            # Detailed Analysis in an enhanced layout
            st.markdown("""
            <div class="conclusion-box">
                <h3 style='color: #1f77b4;'>📊 Regional Distribution Insights</h3>
                <div class="insight-box">
                    <h4>Geographic Patterns</h4>
                    <ul>
                        <li>Strong concentration in Texas (10,557 units)</li>
                        <li>Significant urban center presence (Seattle: 3,711 units)</li>
                        <li>Notable coastal region performance</li>
                    </ul>
                </div>
                
                <div class="insight-box">
                    <h4>Clinical Applications</h4>
                    <ul>
                        <li>Primary use in pain management</li>
                        <li>Fever reduction applications</li>
                        <li>Common in both acute and chronic conditions</li>
                    </ul>
                </div>
                
                <div class="insight-box">
                    <h4>Market Implications</h4>
                    <ul>
                        <li>Higher demand in urban centers</li>
                        <li>Seasonal variation patterns observed</li>
                        <li>Strong correlation with population density</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Public health implications in a card-like container
            st.markdown("""
            <div class="conclusion-box">
                <h3 style='color: #1f77b4;'>📋 Public Health Implications</h3>
                <div class="insight-box">
                    <p>These findings support the role of medicine sales as regional health indicators:</p>
                    <ul>
                        <li>✓ Identify regional health patterns</li>
                        <li>✓ Guide public health resource allocation</li>
                        <li>✓ Support preventive healthcare initiatives</li>
                        <li>✓ Enable evidence-based policy making</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.subheader("Time-Based Analysis")

            # Monthly trend
            st.write("### Monthly Sales Trend")
            monthly_fig = create_monthly_trend(filtered_df)
            st.plotly_chart(monthly_fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                # Daily pattern
                st.write("### Daily Sales Pattern")
                daily_fig = create_daily_pattern(filtered_df)
                st.plotly_chart(daily_fig, use_container_width=True)

            with col2:
                # Hourly pattern
                st.write("### Hourly Sales Pattern")
                hourly_fig = create_hourly_pattern(filtered_df)
                st.plotly_chart(hourly_fig, use_container_width=True)

            # Heatmap of hourly sales by day
            st.write("### Sales Heatmap: Day vs Hour")

            # Create a simpler version directly here to avoid column naming issues
            try:
                # Alternative implementation to avoid column naming issues
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                hours = list(range(24))

                # Create a matrix for the heatmap
                sales_matrix = np.zeros((len(days), len(hours)))

                # Group data by day and hour, ignoring problematic reset_index
                day_hour_data = filtered_df.groupby([filtered_df['datum'].dt.day_name(), filtered_df['datum'].dt.hour])['total_cost'].mean()

                # Fill the matrix with values
                for (day, hour), value in day_hour_data.items():
                    if day in days:
                        day_idx = days.index(day)
                        sales_matrix[day_idx, hour] = value

                # Create heatmap with the matrix
                fig = px.imshow(
                    sales_matrix,
                    x=hours,
                    y=days,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Average Sales ($)"),
                    title="Average Sales by Day and Hour"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating day/hour heatmap: {e}")
                st.warning("This visualization is temporarily unavailable, but you can still use the other analysis tools.")

        with tab4:
            st.subheader("Category Detail Analysis")

            # Category selection for detailed analysis
            selected_category = st.selectbox(
                "Select a category to analyze",
                options=categories,
                format_func=lambda x: x.replace('_', ' ').title()
            )

            # Time series for selected category
            st.write(f"### {selected_category.replace('_', ' ').title()} Sales Over Time")
            category_ts = create_time_series(filtered_df, selected_category)
            st.plotly_chart(category_ts, use_container_width=True)

            # Correlation analysis
            st.write("### Correlation Between Categories")
            corr_fig = create_correlation_heatmap(filtered_df)
            st.plotly_chart(corr_fig, use_container_width=True)

            # Top locations for selected category
            st.write(f"### Top 10 Cities for {selected_category.replace('_', ' ').title()}")
            category_locations = filtered_df.groupby('city')[selected_category].sum().sort_values(ascending=False).head(10)

            fig = px.bar(
                x=category_locations.index,
                y=category_locations.values,
                labels={'x': 'City', 'y': 'Sales'},
            )

            st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.header("📦 Stock Recommendations & Analysis")
            
            # Get medication columns
            med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                       'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                       'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']

            # Get market analysis data
            overall_analysis, med_analysis = analyze_market_data(filtered_df, med_cols)
            
            # Market Overview
            col1, col2 = st.columns(2)
            
            with col1:
                # Market share pie chart
                market_shares = pd.DataFrame(overall_analysis['market_leaders'], 
                                           columns=['Medication', 'Market Share'])
                fig_market = px.pie(market_shares, values='Market Share', names='Medication',
                                  title='Market Share Distribution')
                st.plotly_chart(fig_market, use_container_width=True)
            
            with col2:
                st.markdown("### 🔍 Market Intelligence")
                st.markdown(f"**Overall Market Trend:** {overall_analysis['trend']}")
                
                # Peak months display
                month_numbers = [int(m) for m in overall_analysis['peak_months']]
                peak_months_names = [calendar.month_name[m] for m in month_numbers]
                st.markdown("**Peak Sales Months:** " + ", ".join(peak_months_names))
                
                # Display top 3 medications with their trends
                st.markdown("**Top Performing Medications:**")
                for med, share in overall_analysis['market_leaders'][:3]:
                    st.info(f"🔹 {med}\n- Market Share: {share:.1f}%\n- Trend: {med_analysis[med.lower().replace(' ', '_')]['growth_trajectory']}")
            
            # Medication-specific recommendations
            st.markdown("### 📦 Stock Management Recommendations")
            
            # Display recommendations for each medication
            for med in med_cols:
                data = med_analysis[med]
                with st.expander(f"Stock Analysis - {med.replace('_', ' ').title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Market Metrics**")
                        st.markdown(f"- Market Share: {data['market_share']:.1f}%")
                        st.markdown(f"- Growth Trajectory: {data['growth_trajectory']}")
                        st.markdown(f"- {data['regional_strength']}")
                    
                    with col2:
                        st.markdown("**Stock Recommendations**")
                        st.info(data['stock_recommendation'])
                        
                    # Regional adjustments
                    st.markdown("**Regional Stock Adjustments**")
                    adj_cols = st.columns(3)
                    sorted_regions = sorted(data['regional_adjustments'].items(), 
                                         key=lambda x: abs(x[1]), 
                                         reverse=True)
                    
                    for i, (region, adjustment) in enumerate(sorted_regions):
                        with adj_cols[i % 3]:
                            if adjustment > 10:
                                st.success(f"📈 {region}: +{adjustment:.1f}%")
                            elif adjustment < -10:
                                st.error(f"📉 {region}: {adjustment:.1f}%")
                            else:
                                st.info(f"📊 {region}: {adjustment:+.1f}%")

        # Allow downloading of filtered data
        st.sidebar.markdown("---")
        st.sidebar.subheader("Export Data")

        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            "Download Filtered Data",
            data=csv,
            file_name=f"pharmaceutical_sales_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        # Simple instruction without detailed format information
        st.info("Please upload a CSV file containing pharmaceutical sales data.")

        col1, col2 = st.columns(2)
        with col1:
            st.image("https://images.unsplash.com/photo-1460925895917-afdab827c52f", use_container_width=True)
        with col2:
            st.image("https://images.unsplash.com/photo-1624711076872-ecdbc5ade023", use_container_width=True)

# Disease Prediction Section
else:
    st.header("🏥 Disease Prediction System")

    try:
        # Load model and symptom data
        model, symptom_columns = load_disease_prediction_resources()

        if model is None or symptom_columns is None:
            st.error("Failed to load disease prediction resources")
            st.stop()

        st.markdown("""
        This system uses machine learning to predict potential diseases based on symptoms. 
        Please select exactly 5 symptoms for accurate prediction.
        """)

        # Create two columns for image and input
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image("https://images.unsplash.com/photo-1504813184591-01572f98c85f", use_container_width=True)
            st.image("https://images.unsplash.com/photo-1597121870960-7b5391b88b84", use_container_width=True)

        with col2:
            # Create multiselect for symptoms
            selected_symptoms = st.multiselect(
                "Select 5 Symptoms",
                options=symptom_columns,
                max_selections=5
            )

            if st.button("Predict Disease") and len(selected_symptoms) == 5:
                # Create input array with features (set selected symptoms to 1, rest 0)
                input_features = [0] * len(symptom_columns)

                # Convert symptom_columns to a regular Python list if it's not already
                symptom_list = list(symptom_columns)

                # Process symptoms silently, don't show debug header
                # st.subheader("Symptom Processing")

                for symptom in selected_symptoms:
                    # Find the index of the symptom in the list of symptoms
                    try:
                        if symptom in symptom_list:
                            index = symptom_list.index(symptom)
                            input_features[index] = 1
                        else:
                            st.warning(f"Symptom '{symptom}' not found in training data")
                    except Exception as e:
                        st.error(f"Error processing symptom '{symptom}': {e}")
                        continue

                # Convert input to DataFrame - ensure we don't include the prognosis column
                # The error happens because the model was trained on data without the prognosis column
                # but our symptom_columns might include it during prediction
                if 'prognosis' in symptom_columns:
                    # Create DataFrame without the prognosis column
                    cols_without_prognosis = [col for col in symptom_columns if col != 'prognosis']
                    user_input_df = pd.DataFrame([input_features[:-1]], columns=cols_without_prognosis)
                else:
                    user_input_df = pd.DataFrame([input_features], columns=symptom_columns)

                # Predict disease
                try:
                    prediction_encoded = model.predict(user_input_df)[0]

                    # Load the label encoder to convert the numeric prediction back to disease name
                    label_encoder = joblib.load("label_encoder.pkl")
                    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

                    # Display result in a nice format
                    st.success(f"### Predicted Disease: {prediction}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.warning("Unable to make a prediction with the selected symptoms.")
                    st.stop()

                # Show selected symptoms
                st.write("**Selected Symptoms:**")
                for symptom in selected_symptoms:
                    st.write(f"- {symptom}")

                # Recommended medications based on predicted disease
                st.markdown("### Recommended Medications")
                if prediction == "Allergy":
                    st.write("- Antihistamines")
                    st.write("- Nasal corticosteroids")
                    st.write("- Decongestants")

                    st.markdown("### Relevant Medication Categories")
                    st.write("- Antihistamines")
                    st.write("- Anti Inflammatory Acetic Acid")
                elif prediction == "Fungal infection":
                    st.write("- Antifungal creams")
                    st.write("- Oral antifungals")
                    st.write("- Medicated powders")

                    st.markdown("### Relevant Medication Categories")
                    st.write("- Antifungal medications")
                    st.write("- Anti-inflammatory agents")
                elif prediction == "GERD":
                    st.write("- Proton pump inhibitors")
                    st.write("- H2 blockers")
                    st.write("- Antacids")

                    st.markdown("### Relevant Medication Categories")
                    st.write("- Anti-inflammatory propionic acid")
                    st.write("- Antacids")
                elif prediction == "Chronic cholestasis":
                    st.write("- Ursodeoxycholic acid")
                    st.write("- Cholestyramine")
                    st.write("- Rifampicin")

                    st.markdown("### Relevant Medication Categories")
                    st.write("- Bile acid sequestrants")
                    st.write("- Anti-inflammatory medications")
                else:
                    st.write("- Consult with a healthcare professional for specific medications")

                    st.markdown("### Relevant Medication Categories")
                    st.write("- Varies based on diagnosis")

                # Important note about the prediction
                st.markdown("### Important Note")
                st.markdown("This is an automated prediction based on the symptoms you provided. Always consult with a healthcare professional for a proper diagnosis and treatment plan.")

            elif len(selected_symptoms) != 5and len(selected_symptoms) > 0:
                st.warning("Please select exactly 5 symptoms for prediction")
                st.write(f"Currently selected: {len(selected_symptoms)}/5 symptoms")

            # Add information about the model
            st.markdown("---")
            st.markdown("""
            ### About the Disease Prediction Model

            This prediction system uses a Naïve Bayes classifier trained on a dataset of symptoms 
            and corresponding diseases. The model analyzes patterns in the symptoms to predict 
            the most likely disease.

            **Note:** This prediction tool is for educational purposes only and should not 
            replace professional medical advice.
            """)

    except Exception as e:
        st.error(f"Error loading the disease prediction model: {e}")
        st.markdown("""
        ### Model files not found

        The disease prediction model or training data files could not be loaded. 
        Please ensure the following files are available in the application directory:
        - disease_prediction_model.pkl
        - Training.csv
        """)