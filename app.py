import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import hashlib

# Page configuration
st.set_page_config(
    page_title="🚀 Advanced AI FP&A Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff4757 0%, #c44569 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff3742;
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff9500;
    }
    .alert-success {
        background: linear-gradient(135deg, #2ed573 0%, #17c0eb 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #00d2d3;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedFPASystem:
    def __init__(self):
        self.data = None
        self.forecasts = {}
        self.scenarios = {}
        self.model_history = []
        
    def load_superstore_dataset(self, uploaded_file=None):
        """Load Superstore dataset from file upload or GitHub"""
        try:
            if uploaded_file is not None:
                # Load from uploaded file
                data = pd.read_csv(uploaded_file)
                st.success("✅ Dataset uploaded successfully!")
            else:
                # Try to load from GitHub repository
                try:
                    data = pd.read_csv('superstore_dataset.csv')
                    st.success("✅ Superstore dataset loaded from repository!")
                except FileNotFoundError:
                    st.info("📁 Superstore dataset not found in repository. Using sample data.")
                    return self.generate_enhanced_sample_data()
            
            # Process Superstore data
            return self.process_superstore_data(data)
            
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return self.generate_enhanced_sample_data()
    
    def process_superstore_data(self, data):
        """Process Superstore dataset for FP&A analysis"""
        processed_data = data.copy()
        
        # Convert date columns
        date_columns = ['Order Date', 'order_date', 'Date', 'date']
        for date_col in date_columns:
            if date_col in processed_data.columns:
                processed_data['Date'] = pd.to_datetime(processed_data[date_col])
                break
        else:
            # If no date column found, create one
            processed_data['Date'] = pd.date_range(start='2021-01-01', periods=len(processed_data), freq='D')
        
        # Standardize column names for FP&A analysis
        column_mapping = {
            'Sales': 'Revenue',
            'sales': 'Revenue',
            'Profit': 'Net_Margin',
            'profit': 'Net_Margin',
            'Quantity': 'Units_Sold',
            'quantity': 'Units_Sold',
            'Category': 'Product_Category',
            'category': 'Product_Category',
            'Sub-Category': 'Product_Subcategory',
            'Region': 'Region',
            'region': 'Region',
            'Segment': 'Customer_Segment',
            'segment': 'Customer_Segment',
            'Customer ID': 'Customer_ID',
            'customer_id': 'Customer_ID',
            'State': 'State',
            'City': 'City'
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in processed_data.columns:
                processed_data = processed_data.rename(columns={old_col: new_col})
        
        # Ensure we have required columns
        if 'Revenue' not in processed_data.columns:
            # Try to find a revenue-like column
            revenue_candidates = [col for col in processed_data.columns if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'amount', 'total'])]
            if revenue_candidates:
                processed_data['Revenue'] = pd.to_numeric(processed_data[revenue_candidates[0]], errors='coerce')
            else:
                st.warning("⚠️ No revenue column found. Using sample data.")
                return self.generate_enhanced_sample_data()
        
        # Ensure Revenue is numeric
        processed_data['Revenue'] = pd.to_numeric(processed_data['Revenue'], errors='coerce')
        
        # Handle Units_Sold
        if 'Units_Sold' not in processed_data.columns:
            quantity_candidates = [col for col in processed_data.columns if any(keyword in col.lower() for keyword in ['quantity', 'units', 'qty'])]
            if quantity_candidates:
                processed_data['Units_Sold'] = pd.to_numeric(processed_data[quantity_candidates[0]], errors='coerce')
            else:
                # Calculate units sold from revenue (assuming average price)
                processed_data['Units_Sold'] = (processed_data['Revenue'] / np.random.uniform(50, 200, len(processed_data))).astype(int)
        
        processed_data['Units_Sold'] = pd.to_numeric(processed_data['Units_Sold'], errors='coerce')
        
        # Handle Net_Margin (Profit)
        if 'Net_Margin' not in processed_data.columns:
            profit_candidates = [col for col in processed_data.columns if any(keyword in col.lower() for keyword in ['profit', 'margin'])]
            if profit_candidates:
                processed_data['Net_Margin'] = pd.to_numeric(processed_data[profit_candidates[0]], errors='coerce')
            else:
                # Estimate profit as percentage of revenue
                processed_data['Net_Margin'] = processed_data['Revenue'] * np.random.uniform(0.1, 0.3, len(processed_data))
        
        processed_data['Net_Margin'] = pd.to_numeric(processed_data['Net_Margin'], errors='coerce')
        
        # Calculate Cost of Goods
        processed_data['Cost_of_Goods'] = processed_data['Revenue'] - processed_data['Net_Margin']
        
        # Calculate derived financial metrics
        processed_data['Gross_Margin'] = processed_data['Net_Margin']  # Simplified assumption
        processed_data['Gross_Margin_Percent'] = (processed_data['Gross_Margin'] / processed_data['Revenue']) * 100
        processed_data['Net_Margin_Percent'] = (processed_data['Net_Margin'] / processed_data['Revenue']) * 100
        
        # Add business operational metrics
        processed_data['Marketing_Spend'] = processed_data['Revenue'] * np.random.uniform(0.08, 0.15, len(processed_data))
        processed_data['Sales_Team_Size'] = np.random.randint(20, 40, len(processed_data))
        processed_data['Revenue_per_Employee'] = processed_data['Revenue'] / processed_data['Sales_Team_Size']
        
        # Customer metrics
        processed_data['Customer_Acquisition_Cost'] = np.random.uniform(40, 250, len(processed_data))
        processed_data['Customer_Lifetime_Value'] = np.random.uniform(600, 2000, len(processed_data))
        processed_data['CAC_LTV_Ratio'] = processed_data['Customer_Lifetime_Value'] / processed_data['Customer_Acquisition_Cost']
        
        # Market and competitive metrics
        processed_data['Price_per_Unit'] = processed_data['Revenue'] / processed_data['Units_Sold']
        processed_data['Marketing_Efficiency'] = processed_data['Revenue'] / processed_data['Marketing_Spend']
        processed_data['Market_Share'] = np.random.uniform(0.1, 0.3, len(processed_data))
        processed_data['Competitor_Price'] = processed_data['Price_per_Unit'] * np.random.uniform(0.9, 1.1, len(processed_data))
        
        # Economic and digital metrics
        processed_data['Economic_Index'] = 100 + np.cumsum(np.random.normal(0, 0.3, len(processed_data)))
        processed_data['Website_Traffic'] = np.random.randint(8000, 25000, len(processed_data))
        processed_data['Conversion_Rate'] = np.random.uniform(0.02, 0.08, len(processed_data))
        
        # Fill missing categorical columns
        if 'Region' not in processed_data.columns:
            processed_data['Region'] = np.random.choice(['North', 'South', 'East', 'West'], len(processed_data))
        if 'Product_Category' not in processed_data.columns:
            processed_data['Product_Category'] = np.random.choice(['Technology', 'Furniture', 'Office Supplies'], len(processed_data))
        if 'Customer_Segment' not in processed_data.columns:
            processed_data['Customer_Segment'] = np.random.choice(['Consumer', 'Corporate', 'Home Office'], len(processed_data))
        
        processed_data['Sales_Channel'] = np.random.choice(['Online', 'Retail', 'B2B'], len(processed_data))
        
        # Clean data
        processed_data = processed_data.dropna(subset=['Revenue', 'Units_Sold', 'Net_Margin'])
        processed_data = processed_data[processed_data['Revenue'] > 0]
        processed_data = processed_data[processed_data['Units_Sold'] > 0]
        
        # Sort by date
        processed_data = processed_data.sort_values('Date').reset_index(drop=True)
        
        st.info(f"📊 Processed {len(processed_data)} records from {processed_data['Date'].min().strftime('%Y-%m-%d')} to {processed_data['Date'].max().strftime('%Y-%m-%d')}")
        
        return processed_data
    
    def generate_enhanced_sample_data(self):
        """Generate realistic sample data"""
        np.random.seed(42)
        
        # Generate 2 years of daily data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        n_days = len(dates)
        
        # Create complex patterns
        base_trend = np.linspace(80000, 180000, n_days)
        seasonal_yearly = 20000 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        seasonal_monthly = 8000 * np.sin(2 * np.pi * np.arange(n_days) / 30.44)
        weekly_pattern = 5000 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        
        # Add economic cycles and market events
        economic_cycle = 10000 * np.sin(2 * np.pi * np.arange(n_days) / 180)
        market_shocks = np.zeros(n_days)
        
        # Simulate market events
        shock_dates = [100, 300, 500, 650]
        for shock_day in shock_dates:
            if shock_day < n_days:
                shock_impact = np.random.uniform(-0.2, 0.3)
                shock_duration = 30
                for i in range(shock_duration):
                    if shock_day + i < n_days:
                        market_shocks[shock_day + i] = shock_impact * base_trend[shock_day + i] * np.exp(-i/10)
        
        noise = np.random.normal(0, 6000, n_days)
        revenue = base_trend + seasonal_yearly + seasonal_monthly + weekly_pattern + economic_cycle + market_shocks + noise
        revenue = np.maximum(revenue, 30000)
        
        # Generate correlated business metrics
        data = pd.DataFrame({
            'Date': dates,
            'Revenue': revenue,
            'Units_Sold': (revenue / np.random.uniform(75, 125, n_days)).astype(int),
            'Cost_of_Goods': revenue * np.random.uniform(0.35, 0.65, n_days),
            'Marketing_Spend': revenue * np.random.uniform(0.06, 0.18, n_days),
            'Sales_Team_Size': np.random.randint(12, 45, n_days),
            'Customer_Acquisition_Cost': np.random.uniform(40, 250, n_days),
            'Customer_Lifetime_Value': np.random.uniform(600, 2000, n_days),
            'Market_Share': np.random.uniform(0.08, 0.25, n_days),
            'Competitor_Price': np.random.uniform(80, 160, n_days),
            'Economic_Index': 100 + np.cumsum(np.random.normal(0, 0.3, n_days)),
            'Website_Traffic': np.random.randint(5000, 25000, n_days),
            'Conversion_Rate': np.random.uniform(0.02, 0.08, n_days),
            'Region': np.random.choice(['North America', 'Europe', 'Asia-Pacific', 'Latin America'], n_days, p=[0.4, 0.3, 0.2, 0.1]),
            'Product_Category': np.random.choice(['Technology', 'Office Supplies', 'Furniture'], n_days, p=[0.45, 0.35, 0.20]),
            'Sales_Channel': np.random.choice(['E-commerce', 'Retail', 'B2B Direct', 'Partner'], n_days, p=[0.4, 0.3, 0.2, 0.1]),
            'Customer_Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], n_days, p=[0.5, 0.3, 0.2])
        })
        
        # Calculate derived metrics
        data['Gross_Margin'] = data['Revenue'] - data['Cost_of_Goods']
        data['Net_Margin'] = data['Gross_Margin'] - data['Marketing_Spend']
        data['Gross_Margin_Percent'] = (data['Gross_Margin'] / data['Revenue']) * 100
        data['Net_Margin_Percent'] = (data['Net_Margin'] / data['Revenue']) * 100
        data['Revenue_per_Employee'] = data['Revenue'] / data['Sales_Team_Size']
        data['CAC_LTV_Ratio'] = data['Customer_Lifetime_Value'] / data['Customer_Acquisition_Cost']
        data['Price_per_Unit'] = data['Revenue'] / data['Units_Sold']
        data['Marketing_Efficiency'] = data['Revenue'] / data['Marketing_Spend']
        
        return data
    
    def calculate_executive_kpis(self, data):
        """Calculate comprehensive executive-level KPIs"""
        recent_data = data.tail(90)
        prev_data = data.tail(180).head(90) if len(data) > 180 else data.head(max(1, len(data)//2))
        
        kpis = {
            # Revenue Analytics
            'total_revenue': recent_data['Revenue'].sum(),
            'revenue_growth_rate': ((recent_data['Revenue'].mean() - prev_data['Revenue'].mean()) / prev_data['Revenue'].mean()) * 100 if len(prev_data) > 0 and prev_data['Revenue'].mean() > 0 else 0,
            'revenue_volatility': recent_data['Revenue'].std() / recent_data['Revenue'].mean() * 100 if recent_data['Revenue'].mean() > 0 else 0,
            'quarterly_run_rate': recent_data['Revenue'].sum() * 4,
            
            # Profitability Analytics
            'gross_margin_current': recent_data['Gross_Margin_Percent'].mean(),
            'net_margin_current': recent_data['Net_Margin_Percent'].mean(),
            'gross_margin_trend': recent_data['Gross_Margin_Percent'].mean() - prev_data['Gross_Margin_Percent'].mean() if len(prev_data) > 0 else 0,
            'margin_compression_risk': max(0, prev_data['Gross_Margin_Percent'].mean() - recent_data['Gross_Margin_Percent'].mean()) if len(prev_data) > 0 else 0,
            
            # Operational Excellence
            'sales_productivity': recent_data['Revenue_per_Employee'].mean(),
            'marketing_roi': (recent_data['Revenue'].sum() - recent_data['Marketing_Spend'].sum()) / recent_data['Marketing_Spend'].sum() * 100 if recent_data['Marketing_Spend'].sum() > 0 else 0,
            'marketing_efficiency': recent_data['Marketing_Efficiency'].mean(),
            
            # Customer Analytics
            'customer_health_score': min(100, recent_data['CAC_LTV_Ratio'].mean() * 20),
            'cac_ltv_ratio': recent_data['CAC_LTV_Ratio'].mean(),
            'churn_risk_prediction': max(0, min(100, 100 - recent_data['CAC_LTV_Ratio'].mean() * 15)),
            
            # Market Position
            'market_share_current': recent_data['Market_Share'].mean(),
            'competitive_advantage_index': min(100, (recent_data['Price_per_Unit'].mean() / recent_data['Competitor_Price'].mean()) * 100) if recent_data['Competitor_Price'].mean() > 0 else 50,
            'price_competitiveness': (recent_data['Price_per_Unit'].mean() / recent_data['Competitor_Price'].mean()) * 100 if recent_data['Competitor_Price'].mean() > 0 else 100,
            
            # Financial Health
            'financial_stability_score': 100 - min(50, recent_data['Revenue'].std() / recent_data['Revenue'].mean() * 100) if recent_data['Revenue'].mean() > 0 else 50,
            'growth_sustainability_index': min(100, max(0, 50 + recent_data['Revenue'].pct_change().mean() * 100)),
        }
        
        return kpis
    
    def generate_alerts(self, data, kpis):
        """Generate intelligent alerts based on KPIs and data"""
        alerts = []
        
        try:
            # Revenue alerts
            revenue_growth = kpis.get('revenue_growth_rate', 0)
            if revenue_growth < -10:
                alerts.append({
                    'type': 'critical',
                    'category': 'Revenue',
                    'message': f"🚨 CRITICAL: Revenue declining by {revenue_growth:.1f}% - Immediate action required",
                    'recommendation': 'Implement emergency revenue recovery plan'
                })
            elif revenue_growth < -5:
                alerts.append({
                    'type': 'warning',
                    'category': 'Revenue',
                    'message': f"⚠️ WARNING: Revenue declining by {revenue_growth:.1f}%",
                    'recommendation': 'Review sales strategy and market conditions'
                })
            
            # Margin alerts
            margin_risk = kpis.get('margin_compression_risk', 0)
            if margin_risk > 5:
                alerts.append({
                    'type': 'critical',
                    'category': 'Profitability',
                    'message': f"🔴 CRITICAL: Severe margin compression ({margin_risk:.1f}%)",
                    'recommendation': 'Urgent cost structure review required'
                })
            
            # Customer health alerts
            churn_risk = kpis.get('churn_risk_prediction', 0)
            if churn_risk > 40:
                alerts.append({
                    'type': 'warning',
                    'category': 'Customer',
                    'message': f"🔥 HIGH CHURN RISK: {churn_risk:.1f}%",
                    'recommendation': 'Launch customer retention initiatives'
                })
            
            # Positive alerts
            if revenue_growth > 20:
                alerts.append({
                    'type': 'success',
                    'category': 'Growth',
                    'message': f"🚀 EXCEPTIONAL GROWTH: Revenue up {revenue_growth:.1f}%",
                    'recommendation': 'Scale successful strategies'
                })
                
        except Exception as e:
            # If there's any error in alert generation, just return empty list
            alerts = []
        
        return alerts
    
    def advanced_forecast(self, data, metric='Revenue', periods=30):
        """Advanced forecasting using multiple methods"""
        if PROPHET_AVAILABLE:
            return self.prophet_forecast(data, metric, periods)
        else:
            return self.simple_forecast(data, metric, periods)
    
    def prophet_forecast(self, data, metric='Revenue', periods=30):
        """Prophet-based forecasting"""
        try:
            # Prepare data for Prophet
            df_prophet = data[['Date', metric]].rename(columns={'Date': 'ds', metric: 'y'})
            df_prophet = df_prophet.dropna()
            
            if len(df_prophet) < 10:
                return self.simple_forecast(data, metric, periods)
            
            # Create Prophet model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.1
            )
            
            model.fit(df_prophet)
            
            # Make future dataframe
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            return {
                'forecast': forecast.tail(periods),
                'model': model,
                'historical_fit': forecast.head(len(df_prophet))
            }
        except Exception as e:
            st.warning(f"Prophet forecasting failed: {str(e)}. Using simple forecasting.")
            return self.simple_forecast(data, metric, periods)
    
    def simple_forecast(self, data, metric='Revenue', periods=30):
        """Simple forecasting as fallback"""
        recent_data = data[metric].tail(90).values
        
        if len(recent_data) < 7:
            # Not enough data for forecasting
            avg_value = recent_data.mean() if len(recent_data) > 0 else 100000
            forecasts = [avg_value] * periods
        else:
            # Simple trend + seasonal
            trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0] if len(recent_data) > 1 else 0
            seasonal_pattern = []
            
            for i in range(7):  # Weekly pattern
                week_data = recent_data[i::7]
                if len(week_data) > 0:
                    seasonal_pattern.append(np.mean(week_data) - np.mean(recent_data))
                else:
                    seasonal_pattern.append(0)
            
            forecasts = []
            last_value = recent_data[-1] if len(recent_data) > 0 else 100000
            
            for i in range(periods):
                seasonal_component = seasonal_pattern[i % 7] if len(seasonal_pattern) > 0 else 0
                forecast_value = last_value + (trend * (i + 1)) + seasonal_component
                forecasts.append(max(0, forecast_value))  # Ensure non-negative
        
        future_dates = pd.date_range(start=data['Date'].max() + timedelta(days=1), periods=periods)
        
        return {
            'forecast': pd.DataFrame({
                'ds': future_dates,
                'yhat': forecasts,
                'yhat_lower': [f * 0.9 for f in forecasts],
                'yhat_upper': [f * 1.1 for f in forecasts]
            }),
            'model': 'Simple',
            'historical_fit': None
        }
    
    def scenario_analysis(self, base_data, scenarios):
        """What-if scenario analysis"""
        results = {}
        
        for scenario_name, params in scenarios.items():
            scenario_data = base_data.copy()
            
            # Apply scenario changes
            if 'price_change' in params and params['price_change'] != 0:
                scenario_data['Price_per_Unit'] *= (1 + params['price_change'] / 100)
                scenario_data['Revenue'] = scenario_data['Price_per_Unit'] * scenario_data['Units_Sold']
                scenario_data['Gross_Margin'] = scenario_data['Revenue'] - scenario_data['Cost_of_Goods']
                scenario_data['Net_Margin'] = scenario_data['Gross_Margin'] - scenario_data['Marketing_Spend']
                scenario_data['Gross_Margin_Percent'] = (scenario_data['Gross_Margin'] / scenario_data['Revenue']) * 100
                scenario_data['Net_Margin_Percent'] = (scenario_data['Net_Margin'] / scenario_data['Revenue']) * 100
            
            if 'market_shock' in params and params['market_shock'] != 0:
                shock_factor = 1 + params['market_shock'] / 100
                scenario_data['Revenue'] *= shock_factor
                scenario_data['Units_Sold'] *= shock_factor
                scenario_data['Gross_Margin'] = scenario_data['Revenue'] - scenario_data['Cost_of_Goods']
                scenario_data['Net_Margin'] = scenario_data['Gross_Margin'] - scenario_data['Marketing_Spend']
                scenario_data['Gross_Margin_Percent'] = (scenario_data['Gross_Margin'] / scenario_data['Revenue']) * 100
                scenario_data['Net_Margin_Percent'] = (scenario_data['Net_Margin'] / scenario_data['Revenue']) * 100
            
            if 'cost_change' in params and params['cost_change'] != 0:
                scenario_data['Cost_of_Goods'] *= (1 + params['cost_change'] / 100)
                scenario_data['Gross_Margin'] = scenario_data['Revenue'] - scenario_data['Cost_of_Goods']
                scenario_data['Net_Margin'] = scenario_data['Gross_Margin'] - scenario_data['Marketing_Spend']
                scenario_data['Gross_Margin_Percent'] = (scenario_data['Gross_Margin'] / scenario_data['Revenue']) * 100
                scenario_data['Net_Margin_Percent'] = (scenario_data['Net_Margin'] / scenario_data['Revenue']) * 100
            
            results[scenario_name] = scenario_data
        
        return results

# Streamlit App Implementation
def main():
    st.markdown('<h1 class="main-header">🚀 Advanced AI-Driven FP&A Platform</h1>', unsafe_allow_html=True)
    
    # Initialize system
    @st.cache_resource
    def get_fpa_system():
        return AdvancedFPASystem()
    
    system = get_fpa_system()
    
    # Sidebar Configuration
    st.sidebar.title("🎛️ Advanced Control Panel")
    
    # Data Source Selection
    st.sidebar.subheader("📊 Data Source")
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Sample Data", "Superstore Dataset", "Upload CSV", "Google Sheets"]
    )
    
    # Load data based on source
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your dataset", 
            type=['csv'],
            help="Upload a CSV file with sales/financial data"
        )
        
    elif data_source == "Superstore Dataset":
        st.sidebar.info("""
        📋 **Superstore Dataset:**
        - Professional retail analytics data
        - Real business patterns and trends
        - Multiple product categories & regions
        
        **To use your own Superstore data:**
        1. Download from Kaggle
        2. Upload via "Upload CSV" option
        """)
    
    # User Configuration
    st.sidebar.subheader("👤 User Settings")
    user_role = st.sidebar.selectbox(
        "Your Role",
        ["C-Suite Executive", "CFO/Finance", "VP Sales", "VP Operations", "VP Marketing", "Data Analyst"]
    )
    
    # Analysis Period
    st.sidebar.subheader("📅 Analysis Period")
    analysis_days = st.sidebar.slider("Days to Analyze", 30, 730, 180)
    
    # Load data
    @st.cache_data
    def load_data():
        if data_source == "Upload CSV" and uploaded_file is not None:
            return system.load_superstore_dataset(uploaded_file)
        elif data_source == "Superstore Dataset":
            return system.load_superstore_dataset()
        else:
            return system.generate_enhanced_sample_data()
    
    data = load_data()
    
    if data is not None and len(data) > 0:
        # Filter data based on date range
        end_date = data['Date'].max()
        start_date = end_date - timedelta(days=analysis_days)
        filtered_data = data[data['Date'] >= start_date].copy()
        
        # Calculate comprehensive KPIs
        kpis = system.calculate_executive_kpis(filtered_data)
        
        # Generate intelligent alerts
        alerts = system.generate_alerts(filtered_data, kpis)
        
        # Main Dashboard Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Executive Dashboard", 
            "🔮 AI Forecasting", 
            "🎯 Scenarios", 
            "⚠️ Alerts",
            "📈 Analytics"
        ])
        
        with tab1:
            render_executive_dashboard(filtered_data, kpis, user_role)
        
        with tab2:
            render_forecasting_lab(system, data, filtered_data)
        
        with tab3:
            render_scenario_lab(system, filtered_data)
        
        with tab4:
            render_alerts_dashboard(alerts, kpis)
        
        with tab5:
            render_deep_analytics(filtered_data, kpis)
    
    else:
        st.error("❌ No data available. Please check your data source configuration.")

def render_executive_dashboard(data, kpis, user_role):
    """Render executive dashboard"""
    st.header("📊 Executive Command Center")
    
    # Data source indicator
    data_info = f"📈 Analyzing {len(data)} records from {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}"
    st.info(data_info)
    
    # Key Performance Indicators
    st.subheader("🎯 Strategic KPIs")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(
            f'''<div class="metric-card">
                <h3>${kpis["total_revenue"]/1000000:.1f}M</h3>
                <p>Total Revenue</p>
                <small>{kpis["revenue_growth_rate"]:+.1f}% vs Prior Period</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'''<div class="metric-card">
                <h3>{kpis["net_margin_current"]:.1f}%</h3>
                <p>Net Margin</p>
                <small>{kpis["gross_margin_trend"]:+.1f}pp vs Prior</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f'''<div class="metric-card">
                <h3>{kpis["customer_health_score"]:.0f}/100</h3>
                <p>Customer Health</p>
                <small>Retention & Value Score</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f'''<div class="metric-card">
                <h3>{kpis["competitive_advantage_index"]:.0f}/100</h3>
                <p>Competitive Position</p>
                <small>Market Advantage Index</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    with col5:
        st.markdown(
            f'''<div class="metric-card">
                <h3>{kpis["growth_sustainability_index"]:.0f}/100</h3>
                <p>Growth Quality</p>
                <small>Sustainability Score</small>
            </div>''',
            unsafe_allow_html=True
        )
    
    # Revenue and Profitability Analysis
    st.subheader("💰 Revenue & Profitability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue trend
        monthly_data = data.groupby(data['Date'].dt.to_period('M')).agg({
            'Revenue': 'sum',
            'Net_Margin': 'sum'
        }).reset_index()
        monthly_data['Date'] = monthly_data['Date'].dt.to_timestamp()
        
        fig_revenue = go.Figure()
        fig_revenue.add_trace(go.Scatter(
            x=monthly_data['Date'],
            y=monthly_data['Revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#667eea', width=3)
        ))
        
        fig_revenue.update_layout(
            title='Monthly Revenue Trend',
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            height=400
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        # Profitability waterfall
        total_revenue = data['Revenue'].sum()
        total_cogs = data['Cost_of_Goods'].sum()
        total_marketing = data['Marketing_Spend'].sum()
        net_profit = data['Net_Margin'].sum()
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="Profitability",
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Revenue", "COGS", "Marketing", "Net Profit"],
            textposition="outside",
            text=[f"${total_revenue/1000000:.1f}M", 
                  f"-${total_cogs/1000000:.1f}M",
                  f"-${total_marketing/1000000:.1f}M", 
                  f"${net_profit/1000000:.1f}M"],
            y=[total_revenue, -total_cogs, -total_marketing, net_profit],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig_waterfall.update_layout(
            title="Profitability Waterfall",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Business Insights
    st.subheader("🧠 Business Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Product category performance
        if 'Product_Category' in data.columns:
            category_performance = data.groupby('Product_Category').agg({
                'Revenue': 'sum',
                'Net_Margin_Percent': 'mean',
                'Units_Sold': 'sum'
            }).reset_index()
            
            fig_category = px.scatter(
                category_performance,
                x='Revenue',
                y='Net_Margin_Percent',
                size='Units_Sold',
                color='Product_Category',
                title='Product Category Performance',
                labels={'Net_Margin_Percent': 'Net Margin %', 'Revenue': 'Total Revenue'}
            )
            st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        # Regional performance
        if 'Region' in data.columns:
            regional_data = data.groupby('Region')['Revenue'].sum().sort_values(ascending=True)
            
            fig_region = go.Figure(go.Bar(
                x=regional_data.values,
                y=regional_data.index,
                orientation='h',
                marker_color='#764ba2'
            ))
            
            fig_region.update_layout(
                title='Revenue by Region',
                xaxis_title='Revenue ($)',
                yaxis_title='Region',
                height=400
            )
            st.plotly_chart(fig_region, use_container_width=True)

def render_forecasting_lab(system, data, filtered_data):
    """Render forecasting interface"""
    st.header("🔮 AI Forecasting Laboratory")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🎛️ Forecast Configuration")
        
        forecast_metric = st.selectbox(
            "Select Metric",
            ["Revenue", "Net_Margin", "Units_Sold", "Gross_Margin"]
        )
        
        forecast_days = st.slider("Forecast Horizon (days)", 7, 90, 30)
        
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Training AI models and generating forecasts..."):
                forecast_result = system.advanced_forecast(data, forecast_metric, forecast_days)
                st.session_state['forecast_result'] = forecast_result
                st.session_state['forecast_metric'] = forecast_metric
    
    with col1:
        if 'forecast_result' in st.session_state:
            forecast_result = st.session_state['forecast_result']
            forecast_metric = st.session_state['forecast_metric']
            
            # Main forecast chart
            fig_forecast = go.Figure()
            
            # Historical data
            recent_data = data.tail(180)
            fig_forecast.add_trace(go.Scatter(
                x=recent_data['Date'],
                y=recent_data[forecast_metric],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            forecast_df = forecast_result['forecast']
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Confidence intervals
            if 'yhat_upper' in forecast_df.columns:
                fig_forecast.add_trace(go.Scatter(
                    x=list(forecast_df['ds']) + list(forecast_df['ds'][::-1]),
                    y=list(forecast_df['yhat_upper']) + list(forecast_df['yhat_lower'][::-1]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Band',
                    showlegend=True
                ))
            
            fig_forecast.update_layout(
                title=f'{forecast_metric} Forecast - Next {forecast_days} Days',
                xaxis_title='Date',
                yaxis_title=forecast_metric,
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Forecast summary
            st.subheader("📊 Forecast Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_forecast = forecast_df['yhat'].mean()
                st.metric("Average Daily", f"{avg_forecast:,.0f}")
            
            with col2:
                total_forecast = forecast_df['yhat'].sum()
                st.metric("Total Forecast", f"{total_forecast:,.0f}")
            
            with col3:
                current_avg = recent_data[forecast_metric].tail(30).mean()
                change = ((avg_forecast - current_avg) / current_avg) * 100 if current_avg > 0 else 0
                st.metric("Change vs Current", f"{change:+.1f}%")

def render_scenario_lab(system, filtered_data):
    """Render scenario planning interface"""
    st.header("🎯 Strategic Scenario Laboratory")
    
    st.markdown("Model the impact of strategic decisions on your business performance.")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🔧 Scenario Builder")
        
        scenario_name = st.text_input("Scenario Name", "Strategic Initiative")
        
        with st.expander("💰 Pricing Strategy"):
            price_change = st.slider("Price Change (%)", -50, 100, 0)
        
        with st.expander("🌍 Market Conditions"):
            market_shock = st.slider("Market Impact (%)", -50, 50, 0)
        
        with st.expander("📊 Cost Structure"):
            cost_change = st.slider("Cost Change (%)", -30, 50, 0)
        
        if st.button("🚀 Run Scenario Analysis", type="primary"):
            scenarios = {
                scenario_name: {
                    'price_change': price_change,
                    'market_shock': market_shock,
                    'cost_change': cost_change
                },
                'Baseline': {}
            }
            
            scenario_results = system.scenario_analysis(filtered_data, scenarios)
            st.session_state['scenario_results'] = scenario_results
            st.session_state['scenario_name'] = scenario_name
    
    with col1:
        if 'scenario_results' in st.session_state:
            scenario_results = st.session_state['scenario_results']
            scenario_name = st.session_state['scenario_name']
            
            # Scenario comparison
            comparison_data = []
            for name, result in scenario_results.items():
                comparison_data.append({
                    'Scenario': name,
                    'Total Revenue': result['Revenue'].sum(),
                    'Total Margin': result['Net_Margin'].sum(),
                    'Avg Margin %': result['Net_Margin_Percent'].mean(),
                    'Revenue Impact': ((result['Revenue'].sum() - scenario_results['Baseline']['Revenue'].sum()) / scenario_results['Baseline']['Revenue'].sum()) * 100 if 'Baseline' in scenario_results and scenario_results['Baseline']['Revenue'].sum() > 0 else 0
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Impact visualization
            fig_impact = go.Figure()
            
            for i, row in comparison_df.iterrows():
                if row['Scenario'] != 'Baseline':
                    fig_impact.add_trace(go.Bar(
                        name=row['Scenario'],
                        x=['Revenue Impact'],
                        y=[row['Revenue Impact']],
                        text=[f"{row['Revenue Impact']:+.1f}%"],
                        textposition='auto'
                    ))
            
            fig_impact.update_layout(
                title=f'Scenario Impact Analysis: {scenario_name}',
                yaxis_title='Impact (%)',
                height=400
            )
            
            st.plotly_chart(fig_impact, use_container_width=True)
            
            # Detailed comparison
            st.subheader("📊 Scenario Comparison")
            st.dataframe(comparison_df.style.format({
                'Total Revenue': '${:,.0f}',
                'Total Margin': '${:,.0f}',
                'Avg Margin %': '{:.1f}%',
                'Revenue Impact': '{:+.1f}%'
            }), use_container_width=True)

def render_alerts_dashboard(alerts, kpis):
    """Render alerts dashboard"""
    st.header("⚠️ Intelligent Alert System")
    
    # Alert summary
    critical_alerts = [a for a in alerts if a.get('type') == 'critical']
    warning_alerts = [a for a in alerts if a.get('type') == 'warning']
    success_alerts = [a for a in alerts if a.get('type') == 'success']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🚨 Critical Alerts", len(critical_alerts))
    with col2:
        st.metric("⚠️ Warning Alerts", len(warning_alerts))
    with col3:
        st.metric("✅ Success Alerts", len(success_alerts))
    
    # Display alerts
    if alerts:
        for alert in alerts:
            alert_type = alert.get('type', 'warning')
            alert_class = f"alert-{alert_type}"
            if alert_type == 'success':
                alert_class = 'alert-success'
            elif alert_type == 'critical':
                alert_class = 'alert-critical'
            else:
                alert_class = 'alert-warning'
            
            st.markdown(
                f'''<div class="{alert_class}">
                    <h4>{alert.get("message", "Alert")}</h4>
                    <p><strong>Recommendation:</strong> {alert.get("recommendation", "Review and take appropriate action")}</p>
                </div>''',
                unsafe_allow_html=True
            )
    else:
        st.success("🎉 All systems are performing well! No alerts detected.")
    
    # Risk dashboard
    st.subheader("🎯 Risk Assessment")
    
    risk_metrics = {
        'Revenue Risk': max(0, -kpis.get('revenue_growth_rate', 0)),
        'Margin Risk': kpis.get('margin_compression_risk', 0),
        'Customer Risk': kpis.get('churn_risk_prediction', 0),
        'Financial Risk': 100 - kpis.get('financial_stability_score', 100)
    }
    
    fig_risk = go.Figure()
    
    risk_names = list(risk_metrics.keys())
    risk_values = list(risk_metrics.values())
    colors = ['green' if v < 20 else 'yellow' if v < 40 else 'red' for v in risk_values]
    
    fig_risk.add_trace(go.Bar(
        x=risk_names,
        y=risk_values,
        marker_color=colors,
        text=[f'{v:.1f}%' for v in risk_values],
        textposition='auto'
    ))
    
    fig_risk.update_layout(
        title='Risk Assessment Matrix',
        yaxis_title='Risk Level (%)',
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig_risk, use_container_width=True)

def render_deep_analytics(data, kpis):
    """Render deep analytics dashboard"""
    st.header("📈 Deep Analytics & Insights")
    
    # Performance by segments
    st.subheader("🔍 Segment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Customer_Segment' in data.columns:
            segment_performance = data.groupby('Customer_Segment').agg({
                'Revenue': 'sum',
                'Net_Margin_Percent': 'mean',
                'Units_Sold': 'sum'
            }).reset_index()
            
            fig_segment = px.bar(
                segment_performance,
                x='Customer_Segment',
                y='Revenue',
                color='Net_Margin_Percent',
                title='Revenue by Customer Segment'
            )
            st.plotly_chart(fig_segment, use_container_width=True)
    
    with col2:
        if 'Sales_Channel' in data.columns:
            channel_performance = data.groupby('Sales_Channel')['Revenue'].sum()
            
            fig_channel = px.pie(
                values=channel_performance.values,
                names=channel_performance.index,
                title='Revenue Distribution by Sales Channel'
            )
            st.plotly_chart(fig_channel, use_container_width=True)
    
    # Time series analysis
    st.subheader("📊 Time Series Analysis")
    
    # Daily revenue with trend
    daily_data = data.groupby('Date')['Revenue'].sum().reset_index()
    
    fig_trend = go.Figure()
    
    fig_trend.add_trace(go.Scatter(
        x=daily_data['Date'],
        y=daily_data['Revenue'],
        mode='lines',
        name='Daily Revenue',
        line=dict(color='lightblue', width=1)
    ))
    
    # Add moving average
    daily_data['MA_7'] = daily_data['Revenue'].rolling(window=7).mean()
    daily_data['MA_30'] = daily_data['Revenue'].rolling(window=30).mean()
    
    fig_trend.add_trace(go.Scatter(
        x=daily_data['Date'],
        y=daily_data['MA_7'],
        mode='lines',
        name='7-Day Average',
        line=dict(color='orange', width=2)
    ))
    
    fig_trend.add_trace(go.Scatter(
        x=daily_data['Date'],
        y=daily_data['MA_30'],
        mode='lines',
        name='30-Day Average',
        line=dict(color='red', width=3)
    ))
    
    fig_trend.update_layout(
        title='Revenue Trend Analysis with Moving Averages',
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        height=500
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Key metrics summary
    st.subheader("📋 Key Metrics Summary")
    
    metrics_summary = {
        'Metric': ['Total Revenue', 'Average Daily Revenue', 'Revenue Growth Rate', 'Net Margin %', 'Customer Health Score'],
        'Value': [
            f"${kpis['total_revenue']/1000000:.1f}M",
            f"${kpis['total_revenue']/len(data):,.0f}",
            f"{kpis['revenue_growth_rate']:+.1f}%",
            f"{kpis['net_margin_current']:.1f}%",
            f"{kpis['customer_health_score']:.0f}/100"
        ],
        'Status': [
            '✅ Strong' if kpis['revenue_growth_rate'] > 5 else '⚠️ Monitor' if kpis['revenue_growth_rate'] > 0 else '🔴 Critical',
            '✅ Healthy',
            '✅ Strong' if kpis['revenue_growth_rate'] > 5 else '⚠️ Monitor' if kpis['revenue_growth_rate'] > 0 else '🔴 Critical',
            '✅ Strong' if kpis['net_margin_current'] > 15 else '⚠️ Monitor' if kpis['net_margin_current'] > 5 else '🔴 Critical',
            '✅ Strong' if kpis['customer_health_score'] > 70 else '⚠️ Monitor' if kpis['customer_health_score'] > 50 else '🔴 Critical'
        ]
    }
    
    st.dataframe(pd.DataFrame(metrics_summary), use_container_width=True)

if __name__ == "__main__":
    main()
