import streamlit as st
import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import io
import plotly.graph_objects as go


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Shopify Forecast App",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CACHED DATA LOADER ---
@st.cache_data
def load_sample():
    return pd.read_csv('shopify_sample1.csv', parse_dates=['Date'])

# --- FORECAST FUNCTION ---
def forecast_sales(sku_df, days_ahead=90):
    """Forecast daily sales for next N days"""
    # Resample to daily (fill missing dates with 0)
    sku_df = sku_df.set_index('Date').resample('D').sum()
    
    if len(sku_df) < 30:  # Need at least 30 days of data
        return None
    
    # Use simpler model if less than 2 years
    n_days = len(sku_df)
    if n_days >= 730:
        model = ExponentialSmoothing(
            sku_df['Quantity'],
            trend='add',
            seasonal='add',
            seasonal_periods=365
        )
    else:
        model = ExponentialSmoothing(
            sku_df['Quantity'],
            trend='add',
            seasonal=None
        )
    
    fitted = model.fit()
    forecast = fitted.forecast(steps=days_ahead)
    
    # Generate future dates (daily)
    future_dates = pd.date_range(sku_df.index[-1], periods=days_ahead+1, freq='D')[1:]
    
    # Return clean DataFrame
    return pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast.round(1)
    })




st.info("Using sample data. Pay $49, and upload your CSV to see real forecasts.")
df = load_sample()


# --- KPI ROW ---
st.title("📦 Shopify Sales Forecast")
total_skus = df['SKU'].nunique()
total_revenue = df['Revenue'].sum()
avg_daily_sales = df.groupby('Date')['Quantity'].sum().mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total SKUs", total_skus)
col2.metric("Total Revenue", f"${total_revenue:,}")
col3.metric("Avg Daily Sales", f"{avg_daily_sales:.0f}")

# --- SKU SELECTOR ---
sku = st.selectbox("Select SKU", df['SKU'].unique())
sku_data = df[df['SKU'] == sku].copy()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "⚠️ Stock Alert", "📊 Trends"])

with tab1:
    forecast_df = forecast_sales(sku_data, days_ahead=90)
    if forecast_df is not None:
        # Show line chart
        fig = px.line(forecast_df, x='Date', y='Forecast', title=f"90-Day Forecast for {sku}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table (first 30 days only)
        st.subheader("Next 30 Days Prediction")
        display_df = forecast_df.head(30).copy()
        # Hide index to avoid duplicate date column
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Need at least 30 days of data for forecast")

with tab2:
    current_stock = sku_data['Current_Stock'].iloc[0]
    weekly_sales = sku_data.set_index('Date')['Quantity'].resample('W').sum().mean()
    weeks_of_stock = current_stock / weekly_sales if weekly_sales > 0 else 999
    
    if weeks_of_stock < 4:
        st.error(f"🚨 REORDER NOW! {weeks_of_stock:.1f} weeks of stock left")
    else:
        st.success(f"✅ Safe stock: {weeks_of_stock:.1f} weeks")
    
    st.metric("Current Stock", f"{current_stock} units")

with tab3:
    fig = px.line(sku_data, x='Date', y='Quantity', title=f"Historical Sales: {sku}")
    st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.subheader(f"📊 Sales Trends for {sku}")
    
    # Prepare daily data
    sku_daily = sku_data.set_index('Date').resample('D').sum()
    
    # --- CHART 1 & 2: Revenue & Quantity Dual Axis ---
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(
            sku_daily,
            x=sku_daily.index,
            y='Revenue',
            title="💰 Revenue Trend (Daily)",
            color_discrete_sequence=['#27ae60']
        )
        fig1.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.line(
            sku_daily,
            x=sku_daily.index,
            y='Quantity',
            title="📦 Units Sold (Daily)",
            color_discrete_sequence=['#3498db']
        )
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    
    # --- CHART 3: Top 5 SKUs ---
    st.subheader("🏆 Top 5 SKUs Across All Products")
    top_skus = df.groupby('SKU').agg({
        'Revenue': 'sum',
        'Quantity': 'sum'
    }).sort_values('Revenue', ascending=False).head(5).reset_index()
    
    fig3 = px.bar(
        top_skus,
        x='SKU',
        y='Revenue',
        title="Total Revenue by SKU",
        color='Quantity',
        color_continuous_scale='Viridis',
        text_auto='$.2s'
    )
    fig3.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig3, use_container_width=True)
    
    # --- CHART 4: FIXED Seasonality Heatmap ---
    st.subheader("📅 Monthly Sales Heatmap")
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    heatmap_data = df.groupby(['SKU', 'YearMonth'])['Quantity'].sum().unstack(fill_value=0)
    
    if sku in heatmap_data.index:
        months = heatmap_data.columns.tolist()
        values = heatmap_data.loc[[sku]].values
        
        fig4 = px.imshow(
            values,
            x=months,
            y=[sku],
            aspect='auto',
            color_continuous_scale='RdYlGn',
            title=f"Monthly Sales Pattern: {sku}"
        )
        fig4.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Not enough monthly data for heatmap")
    
    # --- CHART 5: FIXED Inventory Turnover Gauge ---
    st.subheader("⚡ Inventory Turnover")
    avg_stock = sku_data['Current_Stock'].mean()
    total_sold = sku_data['Quantity'].sum()
    turnover_rate = total_sold / avg_stock if avg_stock > 0 else 0
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig5 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=min(turnover_rate, 10),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Turnover Rate (higher is better)"},
            gauge={
                'axis': {'range': [None, 10]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 2], 'color': "#e74c3c"},
                    {'range': [2, 5], 'color': "#f39c12"},
                    {'range': [5, 10], 'color': "#27ae60"}
                ],
            }
        ))
        fig5.update_layout(height=300, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig5, use_container_width=True)
    
    with col4:
        st.metric("Total Units Sold", f"{total_sold:,}")
        st.metric("Average Stock", f"{avg_stock:,.0f}")
        st.metric("Turnover Rate", f"{turnover_rate:.2f}x")
    
    