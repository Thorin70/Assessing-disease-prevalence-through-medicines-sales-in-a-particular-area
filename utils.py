import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
import numpy as np
from datetime import datetime, timedelta
import calendar

# Function to load and preprocess data
def load_and_preprocess_data(file):
    """Load and preprocess the sales data"""
    df = pd.read_csv(file)
    # Convert datum column to datetime
    df['datum'] = pd.to_datetime(df['datum'], format='mixed')
    
    # Get medication columns
    med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']
    
    # Medication pricing per unit
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
    
    # Calculate total units if not present
    if 'total_units' not in df.columns:
        df['total_units'] = df[med_cols].sum(axis=1)
    
    # Calculate cost breakdown and total calculated cost
    for med in med_cols:
        df[f'{med}_cost'] = df[med] * med_prices[med]
    
    # Calculate total calculated cost (sum of individual medication costs)
    if 'calculated_cost' not in df.columns:
        df['calculated_cost'] = sum(df[f'{med}_cost'] for med in med_cols)
    
    # Use provided total_cost if available, otherwise use calculated cost
    if 'total_cost' not in df.columns:
        df['total_cost'] = df['calculated_cost']
    
    # Calculate average price per unit
    df['avg_price_per_unit'] = df['total_cost'] / df['total_units']
    df['avg_price_per_unit'] = df['avg_price_per_unit'].fillna(0)
    
    return df

def get_first_six_months_data(df):
    """Filter data for first 6 months"""
    min_date = df['datum'].min()
    six_months_later = min_date + pd.DateOffset(months=6)
    return df[(df['datum'] >= min_date) & (df['datum'] <= six_months_later)]

def create_monthly_comparison(df):
    """Create monthly comparison visualization"""
    monthly_sales = df.groupby(df['datum'].dt.strftime('%Y-%m'))['total_cost'].sum().reset_index()
    monthly_sales.columns = ['Month', 'Total Sales']
    
    fig = px.bar(monthly_sales, x='Month', y='Total Sales',
                title='Monthly Sales Comparison',
                labels={'Total Sales': 'Total Sales ($)', 'Month': 'Month'},
                color='Total Sales')
    return fig

def create_category_trend(df):
    """Create category trend analysis"""
    med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']
    
    # Group by month and sum medication sales
    monthly_category = df.groupby(df['datum'].dt.strftime('%Y-%m'))[med_cols].sum().reset_index()
    
    # Melt the dataframe for plotting
    melted_df = pd.melt(monthly_category, id_vars=['datum'], value_vars=med_cols, 
                   var_name='Medication Category', value_name='Units Sold')
    
    # Create line chart
    fig = px.line(melted_df, x='datum', y='Units Sold', color='Medication Category',
                 title='Monthly Trend by Medication Category',
                 labels={'datum': 'Month', 'Units Sold': 'Units Sold'})
    return fig

def create_weekly_sales_pattern(df):
    """Create weekly sales pattern visualization"""
    # Group by day of week and calculate average sales
    weekday_avg = df.groupby(df['datum'].dt.day_name())['total_cost'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    # Create bar chart
    fig = px.bar(x=weekday_avg.index, y=weekday_avg.values,
                title='Average Daily Sales by Day of Week',
                labels={'x': 'Day of Week', 'y': 'Average Sales ($)'})
    return fig

def get_six_month_summary(df):
    """Get summary statistics for first 6 months"""
    six_month_data = get_first_six_months_data(df)
    
    med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']
    
    # Calculate metrics
    total_sales = six_month_data['total_cost'].sum()
    avg_daily_sales = six_month_data.groupby(six_month_data['datum'].dt.date)['total_cost'].sum().mean()
    
    # Get top medication category
    category_sales = six_month_data[med_cols].sum()
    top_category = category_sales.idxmax()
    
    # Calculate month-over-month growth
    monthly_sales = six_month_data.groupby(six_month_data['datum'].dt.strftime('%Y-%m'))['total_cost'].sum()
    if len(monthly_sales) >= 2:
        last_month = monthly_sales.iloc[-1]
        prev_month = monthly_sales.iloc[-2]
        mom_growth = ((last_month - prev_month) / prev_month * 100) if prev_month > 0 else 0
    else:
        mom_growth = 0
    
    return {
        'total_sales': total_sales,
        'avg_daily_sales': avg_daily_sales,
        'top_category': top_category.replace('_', ' ').title(),
        'mom_growth': mom_growth
    }

def create_time_series(df, category):
    """Create time series plot for selected category"""
    # Group by date and calculate sum for the selected category
    daily_data = df.groupby(df['datum'].dt.date)[category].sum().reset_index()
    
    # Create time series plot
    fig = px.line(daily_data, x='datum', y=category,
                 title=f'{category.replace("_", " ").title()} Sales Over Time',
                 labels={'datum': 'Date', category: 'Units Sold'})
    return fig

def create_geographic_heatmap(df):
    """Create geographic heatmap of sales"""
    # Group by state and calculate total sales
    state_sales = df.groupby('state')['total_cost'].sum().reset_index()
    
    # Create base map
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    # Add heatmap to map
    for _, row in state_sales.iterrows():
        state = row['state']
        sales = row['total_cost']
        
        # Pseudo-random position based on state name
        # This is a simplified approach - in a real app, you'd use actual coordinates
        lat = 39.8283 + (hash(state) % 10) / 10
        lon = -98.5795 + (hash(state[::-1]) % 10) / 10
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=np.sqrt(sales) / 100,
            popup=f"{state}: ${sales:,.2f}",
            fill=True,
            color='red',
            fill_opacity=0.7
        ).add_to(m)
    
    return m

def create_category_comparison(df):
    """Create bar chart comparing different medication categories"""
    med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']
    
    # Calculate total sales for each category
    category_sales = df[med_cols].sum().sort_values(ascending=False)
    
    # Create bar chart
    fig = px.bar(x=category_sales.index, y=category_sales.values,
                title='Sales Comparison by Medication Category',
                labels={'x': 'Medication Category', 'y': 'Units Sold'},
                color=category_sales.values)
    
    # Format x-axis labels
    fig.update_xaxes(tickangle=45, ticktext=[cat.replace('_', ' ').title() for cat in category_sales.index],
                   tickvals=category_sales.index)
    
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap between medication categories"""
    med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']
    
    # Calculate correlation matrix
    corr_matrix = df[med_cols].corr()
    
    # Create heatmap
    fig = px.imshow(corr_matrix,
                   labels=dict(color="Correlation"),
                   x=[col.replace('_', ' ').title() for col in corr_matrix.columns],
                   y=[col.replace('_', ' ').title() for col in corr_matrix.columns],
                   title="Medication Sales Correlation")
    
    return fig

def create_weekly_pattern(df):
    """Create weekly sales pattern visualization"""
    # Group by day of week and hour to create heatmap
    # Use unique column names to avoid the 'datum already exists' error
    hourly_day = df.groupby([df['datum'].dt.day_name(), df['datum'].dt.hour])['total_cost'].mean().reset_index()
    # Rename columns explicitly to avoid conflicts
    hourly_day = hourly_day.rename(columns={'datum': 'Day', 0: 'Hour', 'total_cost': 'Average Sales'})
    
    # Create pivot table
    pivot_table = hourly_day.pivot(index='Day', columns='Hour', values='Average Sales')
    
    # Reorder days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(days_order)
    
    # Create heatmap
    fig = px.imshow(pivot_table,
                   labels=dict(x="Hour of Day", y="Day of Week", color="Average Sales ($)"),
                   x=pivot_table.columns,
                   y=pivot_table.index,
                   title="Average Sales by Day and Hour")
    
    return fig

def get_summary_stats(df):
    """Calculate summary statistics"""
    med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']
    
    # Calculate metrics
    total_sales = df['total_cost'].sum()
    avg_daily_sales = df.groupby(df['datum'].dt.date)['total_cost'].sum().mean()
    
    # Get top medication category
    category_sales = df[med_cols].sum()
    top_category = category_sales.idxmax()
    
    # Calculate month-over-month growth
    monthly_sales = df.groupby(df['datum'].dt.strftime('%Y-%m'))['total_cost'].sum()
    if len(monthly_sales) >= 2:
        last_month = monthly_sales.iloc[-1]
        prev_month = monthly_sales.iloc[-2]
        mom_growth = ((last_month - prev_month) / prev_month * 100) if prev_month > 0 else 0
    else:
        mom_growth = 0
    
    return {
        'total_sales': total_sales,
        'avg_daily_sales': avg_daily_sales,
        'top_category': top_category.replace('_', ' ').title(),
        'mom_growth': mom_growth
    }

def create_top_cities_chart(df):
    """Create visualization of top 10 cities by sales"""
    # Group by city and calculate total sales
    city_sales = df.groupby('city')['total_cost'].sum().reset_index()
    
    # Get top 10 cities
    top_cities = city_sales.nlargest(10, 'total_cost')
    
    # Create bar chart
    fig = px.bar(top_cities, x='city', y='total_cost',
                title='Top 10 Cities by Sales',
                labels={'city': 'City', 'total_cost': 'Total Sales ($)'},
                color='total_cost')
    
    return fig

def create_monthly_trend(df):
    """Create monthly sales trend visualization"""
    # Group by month and calculate total sales
    monthly_data = df.groupby(df['datum'].dt.strftime('%Y-%m'))['total_cost'].sum().reset_index()
    monthly_data.columns = ['Month', 'Total Sales']
    
    # Create line chart
    fig = px.line(monthly_data, x='Month', y='Total Sales',
                 title='Monthly Sales Trend',
                 labels={'Month': 'Month', 'Total Sales': 'Total Sales ($)'})
    
    return fig

def create_hourly_pattern(df):
    """Create hourly sales pattern visualization"""
    # Group by hour and calculate average sales
    hourly_avg = df.groupby(df['datum'].dt.hour)['total_cost'].mean().reset_index()
    hourly_avg.columns = ['Hour', 'Average Sales']
    
    # Create bar chart
    fig = px.bar(hourly_avg, x='Hour', y='Average Sales',
                title='Average Sales by Hour of Day',
                labels={'Hour': 'Hour of Day', 'Average Sales': 'Average Sales ($)'})
    
    return fig

def create_daily_pattern(df):
    """Create daily sales pattern visualization"""
    # Group by day of week and calculate average sales
    weekday_avg = df.groupby(df['datum'].dt.day_name())['total_cost'].mean().reset_index()
    weekday_avg.columns = ['Day', 'Average Sales']
    
    # Set correct order for days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_avg['Day'] = pd.Categorical(weekday_avg['Day'], categories=days_order, ordered=True)
    weekday_avg = weekday_avg.sort_values('Day')
    
    # Create bar chart
    fig = px.bar(weekday_avg, x='Day', y='Average Sales',
                title='Average Sales by Day of Week',
                labels={'Day': 'Day of Week', 'Average Sales': 'Average Sales ($)'})
    
    return fig

def create_hourly_heatmap(df):
    """Create hourly sales heatmap by day"""
    # Group by day of week and hour to create heatmap
    hourly_day = df.groupby([df['datum'].dt.day_name(), df['datum'].dt.hour])['total_cost'].mean().reset_index()
    # Rename columns explicitly to avoid conflicts
    hourly_day = hourly_day.rename(columns={df['datum'].dt.day_name().name: 'Day', df['datum'].dt.hour.name: 'Hour', 'total_cost': 'Average Sales'})
    
    # Create pivot table
    pivot_table = hourly_day.pivot(index='Day', columns='Hour', values='Average Sales')
    
    # Reorder days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex(days_order)
    
    # Create heatmap
    fig = px.imshow(pivot_table,
                   labels=dict(x="Hour of Day", y="Day of Week", color="Average Sales ($)"),
                   x=pivot_table.columns,
                   y=pivot_table.index,
                   title="Average Sales by Day and Hour")
    
    return fig

def create_category_growth(df):
    """Analyze category growth over time"""
    med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']
    
    # Group by month and calculate category sales
    monthly_category = df.groupby(df['datum'].dt.strftime('%Y-%m'))[med_cols].sum()
    
    # Calculate growth rates
    growth_rates = {}
    for col in med_cols:
        if len(monthly_category) >= 2:
            first_month = monthly_category[col].iloc[0]
            last_month = monthly_category[col].iloc[-1]
            if first_month > 0:
                growth = ((last_month - first_month) / first_month * 100)
            else:
                growth = 0
        else:
            growth = 0
        
        growth_rates[col.replace('_', ' ').title()] = growth
    
    # Create bar chart
    growth_df = pd.DataFrame({
        'Category': list(growth_rates.keys()),
        'Growth Rate (%)': list(growth_rates.values())
    })
    
    fig = px.bar(growth_df, x='Category', y='Growth Rate (%)',
                title='Medication Category Growth Rates',
                color='Growth Rate (%)')
    
    return fig

def create_city_trends(df):
    """Analyze sales trends for top cities"""
    # Get top 5 cities by total sales
    top_cities = df.groupby('city')['total_cost'].sum().nlargest(5).index.tolist()
    
    # Filter data for top cities
    top_city_data = df[df['city'].isin(top_cities)]
    
    # Group by city and month
    city_monthly = top_city_data.groupby(['city', top_city_data['datum'].dt.strftime('%Y-%m')])['total_cost'].sum().reset_index()
    city_monthly.columns = ['City', 'Month', 'Total Sales']
    
    # Create line chart
    fig = px.line(city_monthly, x='Month', y='Total Sales', color='City',
                 title='Monthly Sales Trends for Top 5 Cities',
                 labels={'Month': 'Month', 'Total Sales': 'Total Sales ($)'})
    
    return fig

def analyze_seasonal_patterns(df):
    """Analyze seasonal patterns in medication categories"""
    med_cols = ['anti_inflammatory_acetic_acid', 'anti_inflammatory_propionic_acid',
                'salicylic_acid_analgesics', 'n02be', 'anxiolytics',
                'hypnotics_sedatives', 'airway_disease_medications', 'antihistamines']
    
    # Add season column based on month
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['datum'].dt.month.apply(get_season)
    
    # Group by season and calculate category sales
    seasonal_sales = df.groupby('season')[med_cols].sum().reset_index()
    
    # Melt dataframe for plotting
    melted_df = pd.melt(seasonal_sales, id_vars=['season'], value_vars=med_cols,
                    var_name='Medication Category', value_name='Units Sold')
    
    # Create bar chart
    fig = px.bar(melted_df, x='season', y='Units Sold', color='Medication Category',
                barmode='group',
                title='Seasonal Patterns in Medication Sales',
                labels={'season': 'Season', 'Units Sold': 'Units Sold'})
    
    return fig

def analyze_market_data(df, med_cols):
    """Analyze overall market trends and individual medication performance"""
    import calendar
    
    # Calculate total sales for market share
    total_sales = df[med_cols].sum().sum()
    market_leaders = []
    
    for med in med_cols:
        med_sales = df[med].sum()
        market_share = (med_sales / total_sales) * 100
        market_leaders.append([med.replace('_', ' ').title(), market_share])
    
    # Sort market leaders by market share
    market_leaders = sorted(market_leaders, key=lambda x: x[1], reverse=True)
    
    # Determine peak months
    monthly_sales = df.groupby(df['datum'].dt.month)[med_cols].sum().sum(axis=1)
    peak_months = monthly_sales.nlargest(3).index.tolist()
    
    # Analyze overall trend using weighted moving average
    daily_sales = df.groupby('datum')[med_cols].sum().sum(axis=1)
    ma_7 = daily_sales.rolling(window=7).mean()
    overall_trend = "Upward" if ma_7.iloc[-7:].mean() > ma_7.iloc[7:14].mean() else "Downward"
    
    overall_analysis = {
        'market_leaders': market_leaders,
        'peak_months': peak_months,
        'trend': overall_trend
    }
    
    # Analyze individual medications
    med_analysis = {}
    
    for med in med_cols:
        # Calculate market share
        med_sales = df[med].sum()
        market_share = (med_sales / total_sales) * 100
        
        # Calculate trend using 7-day moving averages for more stability
        daily_med_sales = df.groupby('datum')[med].sum()
        ma_7 = daily_med_sales.rolling(window=7).mean()
        
        # Compare recent period to previous period
        recent_avg = ma_7.iloc[-7:].mean()
        prev_avg = ma_7.iloc[7:14].mean()
        
        # Calculate growth percentage
        if prev_avg > 0:
            growth = ((recent_avg - prev_avg) / prev_avg) * 100
        else:
            growth = 0
            
        # Determine trajectory with market share consideration
        if market_share > 30:  # Major products
            if growth > 5:
                trajectory = "Strong Growth"
            elif growth > -5:
                trajectory = "Stable with High Market Share"  # Changed for high market share items
            else:
                trajectory = "Slight Volume Adjustment"  # Changed from "Declining"
        else:  # Regular products
            if growth > 15:
                trajectory = "Strong Growth"
            elif growth > 5:
                trajectory = "Moderate Growth"
            elif growth > -5:
                trajectory = "Stable"
            else:
                trajectory = "Declining"
        
        # Regional analysis
        regional_sales = df.groupby('state')[med].sum()
        total_regional = regional_sales.sum()
        regional_share = (regional_sales / total_regional) * 100
        
        # Calculate recommended regional adjustments
        avg_share = 100 / len(regional_sales)
        adjustments = {state: (share - avg_share) for state, share in regional_share.items()}
        
        # Determine regional strength
        top_region = regional_share.idxmax()
        top_share = regional_share.max()
        
        med_analysis[med] = {
            'market_share': market_share,
            'growth_trajectory': trajectory,
            'regional_strength': f"Strongest in {top_region} ({top_share:.1f}% of sales)",
            'regional_adjustments': adjustments,
            'stock_recommendation': generate_stock_recommendation(trajectory, market_share, growth)
        }
    
    return overall_analysis, med_analysis

def generate_stock_recommendation(trajectory, market_share, growth):
    """Generate stock management recommendations based on metrics"""
    base_message = ""
    # For high market share items (>30%), prioritize maintaining stock levels even with small declines
    if market_share > 30:
        if growth > 5:
            base_message = f"Increase stock levels by {min(20, abs(growth))}% to maintain market dominance"
        elif growth > -10:
            base_message = "Maintain current stock levels with safety buffer due to high market share"
        else:
            base_message = "Optimize inventory while maintaining market leadership position - consider minor adjustments of 5-10%"
    # For regular market share items
    else:
        if trajectory == "Strong Growth":
            base_message = f"Increase stock levels by {min(30, abs(growth))}% to meet growing demand"
        elif trajectory == "Moderate Growth":
            base_message = f"Gradually increase stock by {min(15, abs(growth))}% over next quarter"
        elif trajectory == "Stable":
            if market_share > 20:
                base_message = "Maintain current stock levels with regular monitoring"
            else:
                base_message = "Maintain current levels with slight buffer for demand spikes"
        else:  # Declining
            if market_share > 15:
                base_message = f"Review stock levels - consider {min(10, abs(growth))}% reduction while maintaining safety stock"
            else:
                base_message = f"Optimize inventory - reduce stock by {min(15, abs(growth))}% with careful monitoring"
    
    # Add seasonal advice if significant changes
    if abs(growth) > 15:
        base_message += "\nConsider seasonal factors when adjusting inventory levels."
    
    return base_message