"""
CLIO Legal Matters Data Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing legal matters data with 
conversion tracking, lead source analytics, geographic analysis, and more.

Author: Claude Code
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import io
from typing import Optional, Dict, Any, Tuple
import re

# Page configuration
st.set_page_config(
    page_title="CLIO Legal Matters Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional color scheme for legal industry
COLORS = {
    'primary': '#1f2937',    # Navy
    'secondary': '#6b7280',  # Gray
    'accent': '#d97706',     # Gold
    'success': '#059669',    # Green
    'warning': '#dc2626',    # Red
    'background': '#f9fafb', # Light gray
    'text': '#111827'        # Dark gray
}

# Custom CSS for professional styling
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #d97706;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    .stSelectbox > div > div > div {
        background-color: white;
    }
    .stDateInput > div > div > input {
        background-color: white;
    }
    .dashboard-header {
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

def extract_practice_area(matter_type):
    """
    Extract the general practice area from the full matter type string.
    
    Args:
        matter_type: Full matter type string (e.g., "Family Law - Divorce")
    
    Returns:
        General practice area (e.g., "Family Law")
    """
    if pd.isna(matter_type):
        return "Unknown"
    
    # Split by hyphen and take the first part, then clean it
    general_area = str(matter_type).split('-')[0].strip()
    
    # Normalize common variations
    area_mapping = {
        'family law': 'Family Law',
        'criminal': 'Criminal Law',
        'real estate': 'Real Estate Law',
        'civil': 'Civil Law',
        'estate & probate': 'Estate & Probate Law',
        'estate and probate': 'Estate & Probate Law',
        'probate': 'Estate & Probate Law'
    }
    
    # Check if the general area matches any known categories
    general_area_lower = general_area.lower()
    for key, value in area_mapping.items():
        if key in general_area_lower:
            return value
    
    return general_area

def create_geographic_map(df: pd.DataFrame):
    """
    Create an interactive map visualization showing leads and conversions by location.
    """
    # Prepare data for mapping
    if 'Primary Contact State' not in df.columns:
        st.warning("Geographic data not available for mapping")
        return
    
    # Aggregate data by state
    state_data = df.groupby('Primary Contact State').agg({
        'Status': ['count', lambda x: (x == 'Hired').sum()],
        'Total Value': 'sum'
    }).round(0)
    state_data.columns = ['Total Leads', 'Conversions', 'Revenue']
    state_data['Conversion Rate'] = (state_data['Conversions'] / state_data['Total Leads'] * 100).round(1)
    state_data = state_data.reset_index()
    
    # Map state abbreviations to full names for Plotly
    state_codes = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    # Try to match state codes
    state_data['state_name'] = state_data['Primary Contact State'].map(state_codes)
    state_data['state_abbr'] = state_data['Primary Contact State']
    
    # Create choropleth map
    fig_map = px.choropleth(
        state_data,
        locations='state_abbr',
        locationmode='USA-states',
        color='Conversion Rate',
        hover_name='Primary Contact State',
        hover_data={
            'Total Leads': True,
            'Conversions': True,
            'Revenue': ':$,.0f',
            'Conversion Rate': ':.1f%',
            'state_abbr': False
        },
        color_continuous_scale='Viridis',
        labels={'Conversion Rate': 'Conversion Rate (%)'},
        title='Lead Conversion Rate by State'
    )
    
    fig_map.update_geos(scope="usa")
    fig_map.update_layout(
        height=600,
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            lakecolor='rgb(255, 255, 255)',
        )
    )
    
    return fig_map

def create_city_bubble_map(df: pd.DataFrame):
    """
    Create a bubble map showing cities with lead counts and conversion metrics.
    """
    if 'Primary Contact City' not in df.columns or 'Primary Contact State' not in df.columns:
        return None
    
    # Aggregate by city and state
    city_data = df.groupby(['Primary Contact City', 'Primary Contact State']).agg({
        'Status': ['count', lambda x: (x == 'Hired').sum()],
        'Total Value': 'sum'
    }).round(0)
    city_data.columns = ['Total Leads', 'Conversions', 'Revenue']
    city_data['Conversion Rate'] = (city_data['Conversions'] / city_data['Total Leads'] * 100).round(1)
    city_data = city_data.reset_index()
    
    # Filter top cities
    top_cities = city_data.nlargest(20, 'Total Leads')
    
    # Create bubble chart
    fig_bubble = px.scatter(
        top_cities,
        x='Total Leads',
        y='Conversion Rate',
        size='Revenue',
        color='Conversions',
        hover_name='Primary Contact City',
        hover_data={
            'Primary Contact State': True,
            'Revenue': ':$,.0f'
        },
        title='Top 20 Cities: Lead Volume vs Conversion Rate',
        labels={
            'Total Leads': 'Total Lead Count',
            'Conversion Rate': 'Conversion Rate (%)'
        },
        color_continuous_scale='RdYlGn'
    )
    
    fig_bubble.update_layout(height=500)
    
    return fig_bubble

@st.cache_data(ttl=300)
def load_and_process_data(file) -> Optional[pd.DataFrame]:
    """
    Load and process the CLIO legal matters data with comprehensive validation.
    
    Args:
        file: Uploaded file object (CSV or Excel)
        
    Returns:
        Processed DataFrame or None if error
    """
    try:
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)
        
        # Clean column names (remove trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Convert date column
        if 'Created' in df.columns:
            df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
        
        # Convert Yes/No columns to boolean
        yes_no_columns = ['Scheduled', 'No show', 'Call', 'Zoom', 'Qualified Lead']
        for col in yes_no_columns:
            if col in df.columns:
                df[col] = df[col].map({'Yes': True, 'No': False})
        
        # Process retainer columns (new)
        if 'Nonrefundable Retainer' in df.columns:
            df['Nonrefundable Retainer'] = pd.to_numeric(df['Nonrefundable Retainer'], errors='coerce')
        
        if 'Refundable Retainer' in df.columns:
            df['Refundable Retainer'] = pd.to_numeric(df['Refundable Retainer'], errors='coerce')
        
        # Calculate total value for hired matters only
        df['Total Value'] = 0
        if 'Status' in df.columns:
            hired_mask = df['Status'] == 'Hired'
            
            if 'Nonrefundable Retainer' in df.columns and 'Refundable Retainer' in df.columns:
                df.loc[hired_mask, 'Total Value'] = (
                    df.loc[hired_mask, 'Nonrefundable Retainer'].fillna(0) + 
                    df.loc[hired_mask, 'Refundable Retainer'].fillna(0)
                )
            elif 'Nonrefundable Retainer' in df.columns:
                df.loc[hired_mask, 'Total Value'] = df.loc[hired_mask, 'Nonrefundable Retainer'].fillna(0)
            elif 'Refundable Retainer' in df.columns:
                df.loc[hired_mask, 'Total Value'] = df.loc[hired_mask, 'Refundable Retainer'].fillna(0)
            elif 'Value' in df.columns:
                # Fallback to old Value column if retainer columns don't exist
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
                df.loc[hired_mask, 'Total Value'] = df.loc[hired_mask, 'Value'].fillna(0)
        
        # Extract general practice area from Matter Type
        if 'Matter Type' in df.columns:
            df['Practice Area'] = df['Matter Type'].apply(extract_practice_area)
        
        # Clean numeric columns for zip codes
        if 'Primary Contact Zip' in df.columns:
            df['Primary Contact Zip'] = pd.to_numeric(df['Primary Contact Zip'], errors='coerce')
        
        # Add derived columns
        df['Month'] = df['Created'].dt.to_period('M') if 'Created' in df.columns else None
        df['Week'] = df['Created'].dt.to_period('W') if 'Created' in df.columns else None
        df['Day_of_Week'] = df['Created'].dt.day_name() if 'Created' in df.columns else None
        df['Hour'] = df['Created'].dt.hour if 'Created' in df.columns else None
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform data quality checks and return warnings.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data quality metrics and warnings
    """
    quality_report = {
        'total_rows': len(df),
        'missing_data': {},
        'warnings': []
    }
    
    # Check for missing data in key columns
    key_columns = ['Created', 'Status', 'Matter Type', 'Created By', 'Primary Contact']
    for col in key_columns:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            quality_report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            if missing_pct > 5:
                quality_report['warnings'].append(
                    f"⚠️ {col}: {missing_pct:.1f}% missing values ({missing_count} records)"
                )
    
    # Check for future dates
    if 'Created' in df.columns:
        future_dates = df[df['Created'] > datetime.now()]['Created'].count()
        if future_dates > 0:
            quality_report['warnings'].append(
                f"⚠️ Found {future_dates} records with future dates"
            )
    
    # Check for negative values
    if 'Value' in df.columns:
        negative_values = df[df['Value'] < 0]['Value'].count()
        if negative_values > 0:
            quality_report['warnings'].append(
                f"⚠️ Found {negative_values} records with negative values"
            )
    
    return quality_report

def apply_filters(df: pd.DataFrame, 
                 date_range: Tuple[datetime, datetime],
                 status_filter: list,
                 matter_type_filter: list,
                 source_filter: list) -> pd.DataFrame:
    """
    Apply sidebar filters to the DataFrame.
    
    Args:
        df: Input DataFrame
        date_range: Tuple of start and end dates
        status_filter: List of selected statuses
        matter_type_filter: List of selected matter types
        source_filter: List of selected sources
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    
    # Date filter
    if 'Created' in filtered_df.columns:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Created'] >= pd.Timestamp(start_date)) & 
            (filtered_df['Created'] <= pd.Timestamp(end_date))
        ]
    
    # Status filter
    if status_filter and 'Status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
    
    # Matter type filter
    if matter_type_filter and 'Matter Type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Matter Type'].isin(matter_type_filter)]
    
    # Source filter
    if source_filter and 'Primary Contact Source' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Primary Contact Source'].isin(source_filter)]
    
    return filtered_df

def create_kpi_cards(df: pd.DataFrame):
    """Create KPI metric cards for the overview page."""
    col1, col2, col3, col4 = st.columns(4)
    
    total_leads = len(df)
    hired_count = len(df[df['Status'] == 'Hired']) if 'Status' in df.columns else 0
    conversion_rate = (hired_count / total_leads * 100) if total_leads > 0 else 0
    total_value = df['Total Value'].sum() if 'Total Value' in df.columns else 0
    avg_value = df['Total Value'].mean() if 'Total Value' in df.columns and len(df) > 0 else 0
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Leads",
            value=f"{total_leads:,}",
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Conversion Rate",
            value=f"{conversion_rate:.1f}%",
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Value",
            value=f"${total_value:,.0f}",
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Average Value",
            value=f"${avg_value:,.0f}",
            delta=None
        )
        st.markdown('</div>', unsafe_allow_html=True)

def create_kpi_cards_with_context(df: pd.DataFrame):
    """Create enhanced KPI metric cards with contextual information."""
    col1, col2, col3, col4 = st.columns(4)
    
    total_leads = len(df)
    hired_count = len(df[df['Status'] == 'Hired']) if 'Status' in df.columns else 0
    conversion_rate = (hired_count / total_leads * 100) if total_leads > 0 else 0
    total_value = df['Total Value'].sum() if 'Total Value' in df.columns else 0
    avg_value = df['Total Value'].mean() if 'Total Value' in df.columns and len(df) > 0 else 0
    
    # Calculate benchmarks (industry standards for legal practices)
    conversion_benchmark = 25.0  # Industry average conversion rate
    value_benchmark = 5000.0     # Industry average matter value
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Leads",
            value=f"{total_leads:,}",
            help="Total number of leads in selected period"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        delta_conversion = f"+{conversion_rate - conversion_benchmark:.1f}%" if conversion_rate >= conversion_benchmark else f"{conversion_rate - conversion_benchmark:.1f}%"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Conversion Rate",
            value=f"{conversion_rate:.1f}%",
            delta=delta_conversion,
            help=f"vs. Industry benchmark: {conversion_benchmark}%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Value",
            value=f"${total_value:,.0f}",
            help="Total value of hired matters"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        delta_value = f"+${avg_value - value_benchmark:,.0f}" if avg_value >= value_benchmark else f"-${value_benchmark - avg_value:,.0f}"
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Average Value",
            value=f"${avg_value:,.0f}",
            delta=delta_value,
            help=f"vs. Industry benchmark: ${value_benchmark:,.0f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)

def create_kpi_cards_with_retainers(df: pd.DataFrame):
    """Create KPI metric cards including retainer information."""
    col1, col2, col3, col4 = st.columns(4)
    
    total_leads = len(df)
    hired_count = len(df[df['Status'] == 'Hired']) if 'Status' in df.columns else 0
    conversion_rate = (hired_count / total_leads * 100) if total_leads > 0 else 0
    
    # Calculate retainer values for hired matters only
    if 'Status' in df.columns:
        hired_df = df[df['Status'] == 'Hired']
        nonrefundable_total = hired_df['Nonrefundable Retainer'].sum() if 'Nonrefundable Retainer' in hired_df.columns else 0
        refundable_total = hired_df['Refundable Retainer'].sum() if 'Refundable Retainer' in hired_df.columns else 0
        total_revenue = hired_df['Total Value'].sum() if 'Total Value' in hired_df.columns else 0
        avg_revenue = total_revenue / hired_count if hired_count > 0 else 0
    else:
        nonrefundable_total = refundable_total = total_revenue = avg_revenue = 0
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Leads",
            value=f"{total_leads:,}",
            delta=None,
            help="Total number of leads in selected period"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Conversion Rate",
            value=f"{conversion_rate:.1f}%",
            delta=None,
            help="Percentage of leads that converted to hired clients"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta=f"NR: ${nonrefundable_total:,.0f}",
            help=f"Nonrefundable: ${nonrefundable_total:,.0f} | Refundable: ${refundable_total:,.0f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Avg Revenue/Hire",
            value=f"${avg_revenue:,.0f}",
            delta=None,
            help="Average revenue per hired client"
        )
        st.markdown('</div>', unsafe_allow_html=True)

def generate_executive_summary(df: pd.DataFrame):
    """Generate and display an executive summary report."""
    # Calculate key metrics
    total_leads = len(df)
    hired_count = len(df[df['Status'] == 'Hired']) if 'Status' in df.columns else 0
    conversion_rate = (hired_count / total_leads * 100) if total_leads > 0 else 0
    total_value = df['Total Value'].sum() if 'Total Value' in df.columns else 0
    avg_value = df['Total Value'].mean() if 'Total Value' in df.columns and len(df) > 0 else 0
    
    # Top performing matter type
    if 'Matter Type' in df.columns and 'Status' in df.columns:
        top_matter_type = df[df['Status'] == 'Hired']['Matter Type'].value_counts().idxmax() if hired_count > 0 else "N/A"
    else:
        top_matter_type = "N/A"
    
    # Top performing staff member
    if 'Created By' in df.columns and 'Status' in df.columns:
        top_staff = df[df['Status'] == 'Hired']['Created By'].value_counts().idxmax() if hired_count > 0 else "N/A"
    else:
        top_staff = "N/A"
    
    # Create executive summary content
    summary_html = f"""
    <div style="background-color: white; padding: 2rem; border-radius: 0.5rem; border: 1px solid #e5e7eb;">
        <h2 style="color: #1f2937; text-align: center; margin-bottom: 1.5rem;">📄 Executive Summary</h2>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
            <div style="text-align: center;">
                <h3 style="color: #d97706; margin-bottom: 0.5rem;">Performance Overview</h3>
                <p><strong>Total Leads:</strong> {total_leads:,}</p>
                <p><strong>Conversion Rate:</strong> {conversion_rate:.1f}%</p>
                <p><strong>Total Revenue:</strong> ${total_value:,.0f}</p>
                <p><strong>Average Value:</strong> ${avg_value:,.0f}</p>
            </div>
            
            <div style="text-align: center;">
                <h3 style="color: #d97706; margin-bottom: 0.5rem;">Top Performers</h3>
                <p><strong>Best Matter Type:</strong> {top_matter_type}</p>
                <p><strong>Top Staff Member:</strong> {top_staff}</p>
                <p><strong>Matters Hired:</strong> {hired_count}</p>
            </div>
        </div>
        
        <div style="margin-bottom: 1.5rem;">
            <h3 style="color: #d97706; margin-bottom: 0.5rem;">Key Insights</h3>
            <ul style="text-align: left;">
                <li>Conversion rate {'exceeds' if conversion_rate >= 25 else 'is below'} industry benchmark (25%)</li>
                <li>Average matter value {'exceeds' if avg_value >= 5000 else 'is below'} industry average ($5,000)</li>
                <li>{top_matter_type} represents the highest-converting practice area</li>
                <li>{top_staff} leads in successful client acquisition</li>
            </ul>
        </div>
        
        <div style="margin-bottom: 1.5rem;">
            <h3 style="color: #d97706; margin-bottom: 0.5rem;">Recommendations</h3>
            <ul style="text-align: left;">
                <li>Focus marketing efforts on {top_matter_type} to maximize ROI</li>
                <li>Implement {top_staff}'s successful strategies across the team</li>
                <li>{'Maintain current' if conversion_rate >= 25 else 'Improve'} conversion processes</li>
                <li>Monitor and replicate best-performing lead sources</li>
            </ul>
        </div>
        
        <p style="text-align: center; color: #6b7280; font-size: 0.875rem; margin-top: 2rem;">
            Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}
        </p>
    </div>
    """
    
    st.markdown(summary_html, unsafe_allow_html=True)
    
    # Create download content outside of button conditionals
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CLIO Legal Matters - Executive Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #1f2937; }}
        .header {{ text-align: center; color: #1f2937; border-bottom: 3px solid #d97706; padding-bottom: 20px; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 30px; margin-bottom: 30px; }}
        .section {{ text-align: center; }}
        .section h3 {{ color: #d97706; margin-bottom: 15px; }}
        .insights, .recommendations {{ margin-bottom: 30px; }}
        .insights h3, .recommendations h3 {{ color: #d97706; margin-bottom: 15px; }}
        .insights ul, .recommendations ul {{ text-align: left; padding-left: 20px; }}
        .insights li, .recommendations li {{ margin-bottom: 8px; }}
        .footer {{ text-align: center; color: #6b7280; font-size: 14px; margin-top: 40px; border-top: 1px solid #e5e7eb; padding-top: 20px; }}
        @media print {{ body {{ margin: 20px; }} }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚖️ CLIO Legal Matters Dashboard</h1>
        <h2>Executive Summary Report</h2>
    </div>
    
    <div class="grid">
        <div class="section">
            <h3>Performance Overview</h3>
            <p><strong>Total Leads:</strong> {total_leads:,}</p>
            <p><strong>Conversion Rate:</strong> {conversion_rate:.1f}%</p>
            <p><strong>Total Revenue:</strong> ${total_value:,.0f}</p>
            <p><strong>Average Value:</strong> ${avg_value:,.0f}</p>
        </div>
        
        <div class="section">
            <h3>Top Performers</h3>
            <p><strong>Best Matter Type:</strong> {top_matter_type}</p>
            <p><strong>Top Staff Member:</strong> {top_staff}</p>
            <p><strong>Matters Hired:</strong> {hired_count}</p>
        </div>
    </div>
    
    <div class="insights">
        <h3>Key Insights</h3>
        <ul>
            <li>Conversion rate {'exceeds' if conversion_rate >= 25 else 'is below'} industry benchmark (25%)</li>
            <li>Average matter value {'exceeds' if avg_value >= 5000 else 'is below'} industry average ($5,000)</li>
            <li>{top_matter_type} represents the highest-converting practice area</li>
            <li>{top_staff} leads in successful client acquisition</li>
        </ul>
    </div>
    
    <div class="recommendations">
        <h3>Recommendations</h3>
        <ul>
            <li>Focus marketing efforts on {top_matter_type} to maximize ROI</li>
            <li>Implement {top_staff}'s successful strategies across the team</li>
            <li>{'Maintain current' if conversion_rate >= 25 else 'Improve'} conversion processes</li>
            <li>Monitor and replicate best-performing lead sources</li>
        </ul>
    </div>
    
    <div class="footer">
        <p>Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}</p>
        <p>Generated with CLIO Legal Matters Dashboard</p>
    </div>
</body>
</html>
    """
    
    # Create CSV data
    summary_data = pd.DataFrame({
        'Metric': ['Total Leads', 'Conversion Rate (%)', 'Total Revenue ($)', 'Average Value ($)', 
                  'Top Matter Type', 'Top Staff Member', 'Matters Hired'],
        'Value': [total_leads, f"{conversion_rate:.1f}", f"{total_value:,.0f}", f"{avg_value:,.0f}",
                 top_matter_type, top_staff, hired_count]
    })
    
    csv_buffer = io.StringIO()
    summary_data.to_csv(csv_buffer, index=False)
    
    # Direct download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📄 Download HTML Report",
            data=html_content,
            file_name=f"clio_executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            help="Downloads a printable HTML report that you can save as PDF from your browser"
        )
    
    with col2:
        st.download_button(
            label="📈 Download CSV Data",
            data=csv_buffer.getvalue(),
            file_name=f"clio_executive_summary_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            help="Downloads the summary metrics as CSV for further analysis"
        )

def show_guided_tour():
    """Display an interactive guided tour of the dashboard."""
    st.markdown("## 🎯 Dashboard Guided Tour")
    
    tour_steps = {
        "Step 1: Overview Dashboard": """
        **Start Here!** The Overview page gives you the big picture:
        - **KPI Cards**: Key metrics with industry benchmarks
        - **Status Distribution**: See how leads convert
        - **Top Matter Types**: Identify your most common practice areas
        - **Recent Matters**: Latest activity in your pipeline
        """,
        
        "Step 2: Conversion Funnel": """
        **Track Your Sales Process** - Identify where leads drop off:
        - **Visual Funnel**: See conversion at each stage
        - **By Matter Type**: Which practice areas convert best?
        - **By Staff**: Who's your top performer?
        """,
        
        "Step 3: Lead Sources": """
        **ROI Analysis** - Know which marketing works:
        - **Volume by Source**: Where leads come from
        - **ROI Analysis**: Which sources bring paying clients
        - **Performance Heatmap**: Visual source comparison
        """,
        
        "Step 4: Staff Performance": """
        **Team Management** - Compare and improve performance:
        - **Revenue Leaderboard**: Top earners
        - **Conversion Leaders**: Best closers
        - **Performance Scatter Plot**: Leads vs Revenue analysis
        """,
        
        "Pro Tips": """
        **🔥 Power User Features:**
        - **Filters**: Use sidebar to focus on specific time periods, statuses, or sources
        - **Export**: Download filtered data for deeper analysis
        - **Executive Summary**: One-click report for leadership
        - **Data Quality**: Check warnings to ensure accuracy
        """
    }
    
    selected_step = st.selectbox("Choose a tour step:", list(tour_steps.keys()))
    
    st.markdown("### " + selected_step)
    st.markdown(tour_steps[selected_step])
    
    if selected_step != "Pro Tips":
        page_map = {
            "Step 1: Overview Dashboard": "Overview",
            "Step 2: Conversion Funnel": "Conversion Funnel",
            "Step 3: Lead Sources": "Lead Sources",
            "Step 4: Staff Performance": "Staff Performance"
        }
        st.info(f"💡 **Next**: Navigate to the '{page_map[selected_step]}' page in the sidebar to see this in action!")
    
    st.markdown("---")
    st.markdown("**Need help?** Each chart has hover tooltips and help text to guide you.")

def main():
    """Main application function."""
    
    # Header
    st.title("⚖️ CLIO Legal Matters Dashboard")
    st.markdown("Comprehensive analytics for legal matter conversion tracking and performance analysis")
    
    # Guided tour help button
    if st.button("🎯 Take a Guided Tour"):
        show_guided_tour()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your CLIO data file",
        type=['csv', 'xlsx'],
        help="Upload a CSV or Excel file containing your CLIO legal matters data"
    )
    
    # Load default file if no upload
    if not uploaded_file:
        try:
            default_file = "C:/Users/fvare/OneDrive/Desktop/Please Work/August Leads 2025 (1).xlsx"
            df = pd.read_excel(default_file)
            df.columns = df.columns.str.strip()
            if 'Created' in df.columns:
                df['Created'] = pd.to_datetime(df['Created'], errors='coerce')
            yes_no_columns = ['Scheduled', 'No show', 'Call', 'Zoom', 'Qualified Lead']
            for col in yes_no_columns:
                if col in df.columns:
                    df[col] = df[col].map({'Yes': True, 'No': False})
            if 'Value' in df.columns:
                df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df['Month'] = df['Created'].dt.to_period('M') if 'Created' in df.columns else None
            df['Day_of_Week'] = df['Created'].dt.day_name() if 'Created' in df.columns else None
            df['Hour'] = df['Created'].dt.hour if 'Created' in df.columns else None
            st.info("Using default data file: August Leads 2025 (1).xlsx")
        except:
            st.warning("Please upload a data file to begin analysis.")
            return
    else:
        df = load_and_process_data(uploaded_file)
        if df is None:
            return
    
    # Data quality check
    quality_report = validate_data_quality(df)
    if quality_report['warnings']:
        with st.expander("⚠️ Data Quality Warnings", expanded=False):
            for warning in quality_report['warnings']:
                st.warning(warning)
    
    # Sidebar filters
    st.sidebar.header("Dashboard Filters")
    
    # Date range filter
    if 'Created' in df.columns:
        min_date = df['Created'].min().date()
        max_date = df['Created'].max().date()
        default_start = max_date.replace(day=1)  # First day of current month
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if len(date_range) != 2:
            date_range = (default_start, max_date)
    else:
        date_range = (datetime.now().date(), datetime.now().date())
    
    # Status filter
    status_options = df['Status'].unique().tolist() if 'Status' in df.columns else []
    status_filter = st.sidebar.multiselect(
        "Status",
        options=status_options,
        default=status_options
    )
    
    # Matter type filter
    matter_type_options = df['Matter Type'].unique().tolist() if 'Matter Type' in df.columns else []
    matter_type_filter = st.sidebar.multiselect(
        "Matter Type",
        options=matter_type_options,
        default=matter_type_options
    )
    
    # Source filter
    source_options = df['Primary Contact Source'].unique().tolist() if 'Primary Contact Source' in df.columns else []
    source_filter = st.sidebar.multiselect(
        "Lead Source",
        options=source_options,
        default=source_options
    )
    
    # Apply filters
    filtered_df = apply_filters(df, date_range, status_filter, matter_type_filter, source_filter)
    
    # Show filtered data count
    st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df):,} / {len(df):,}")
    
    # Export functionality
    if st.sidebar.button("📥 Export Filtered Data"):
        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"clio_filtered_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Main dashboard content
    if len(filtered_df) == 0:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
        return
    
    # Page navigation
    pages = [
        "Overview",
        "Conversion Funnel", 
        "Lead Sources",
        "Engagement Metrics",
        "Geographic Analysis",
        "Financial Dashboard",
        "Staff Performance",
        "Time Analysis"
    ]
    
    selected_page = st.sidebar.selectbox("Select Dashboard Page", pages)
    
    # Route to selected page
    if selected_page == "Overview":
        show_overview_page(filtered_df)
    elif selected_page == "Conversion Funnel":
        show_conversion_funnel_page(filtered_df)
    elif selected_page == "Lead Sources":
        show_lead_sources_page(filtered_df)
    elif selected_page == "Engagement Metrics":
        show_engagement_metrics_page(filtered_df)
    elif selected_page == "Geographic Analysis":
        show_geographic_analysis_page(filtered_df)
    elif selected_page == "Financial Dashboard":
        show_financial_dashboard_page(filtered_df)
    elif selected_page == "Staff Performance":
        show_staff_performance_page(filtered_df)
    elif selected_page == "Time Analysis":
        show_time_analysis_page(filtered_df)

def show_overview_page(df: pd.DataFrame):
    """Display the overview page with KPIs and status breakdown."""
    st.header("📊 Overview Dashboard")
    
    # Add executive summary button
    if st.button("📄 Generate Executive Summary"):
        generate_executive_summary(df)
    
    # KPI Cards with retainer information
    create_kpi_cards_with_retainers(df)
    
    # Status breakdown and Practice Area distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Status' in df.columns:
            status_counts = df['Status'].value_counts()
            
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Matter Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(
                hovertemplate="<b>%{label}</b><br>" +
                             "Count: %{value}<br>" +
                             "Percentage: %{percent}<br>" +
                             "<extra></extra>"
            )
            fig_pie.update_layout(
                font_size=12,
                title_font_size=16,
                showlegend=True
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if 'Practice Area' in df.columns:
            practice_counts = df['Practice Area'].value_counts()
            fig_bar = px.bar(
                x=practice_counts.values,
                y=practice_counts.index,
                orientation='h',
                title="Leads by Practice Area",
                color=practice_counts.values,
                color_continuous_scale='Blues'
            )
            fig_bar.update_traces(
                hovertemplate="<b>%{y}</b><br>" +
                             "Count: %{x}<br>" +
                             "<extra></extra>"
            )
            fig_bar.update_layout(
                font_size=12,
                title_font_size=16,
                showlegend=False,
                yaxis={'categoryorder':'total ascending'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Retainer Analysis Section
    st.subheader("💰 Retainer Analysis")
    if 'Nonrefundable Retainer' in df.columns or 'Refundable Retainer' in df.columns:
        hired_df = df[df['Status'] == 'Hired'] if 'Status' in df.columns else df
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Retainer type comparison
            retainer_data = {
                'Type': ['Nonrefundable', 'Refundable'],
                'Total': [
                    hired_df['Nonrefundable Retainer'].sum() if 'Nonrefundable Retainer' in hired_df.columns else 0,
                    hired_df['Refundable Retainer'].sum() if 'Refundable Retainer' in hired_df.columns else 0
                ]
            }
            
            fig_retainer = px.bar(
                retainer_data,
                x='Type',
                y='Total',
                title='Retainer Revenue Breakdown',
                color='Type',
                color_discrete_map={'Nonrefundable': COLORS['success'], 'Refundable': COLORS['accent']}
            )
            fig_retainer.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                             "Total: $%{y:,.0f}<br>" +
                             "<extra></extra>"
            )
            st.plotly_chart(fig_retainer, use_container_width=True)
        
        with col2:
            # Average retainer by practice area
            if 'Practice Area' in hired_df.columns:
                retainer_by_area = hired_df.groupby('Practice Area').agg({
                    'Nonrefundable Retainer': 'mean',
                    'Refundable Retainer': 'mean',
                    'Total Value': 'mean'
                }).round(0)
                
                retainer_by_area = retainer_by_area.sort_values('Total Value', ascending=False).head(5)
                
                # Format for display
                display_retainer = retainer_by_area.copy()
                for col in display_retainer.columns:
                    display_retainer[col] = display_retainer[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
                
                st.subheader("Top 5 Practice Areas by Avg Revenue")
                st.dataframe(display_retainer, use_container_width=True)
    
    # Recent activity table with better formatting
    st.subheader("Recent Matters")
    if 'Created' in df.columns:
        columns_to_show = ['Created', 'Practice Area', 'Status', 'Primary Contact', 'Total Value']
        available_columns = [col for col in columns_to_show if col in df.columns]
        
        recent_df = df.nlargest(10, 'Created')[available_columns].copy()
        if 'Total Value' in recent_df.columns:
            recent_df['Total Value'] = recent_df['Total Value'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        if 'Created' in recent_df.columns:
            recent_df['Created'] = recent_df['Created'].dt.strftime('%Y-%m-%d')
        st.dataframe(recent_df, use_container_width=True)

def show_conversion_funnel_page(df: pd.DataFrame):
    """Display conversion funnel analysis."""
    st.header("🎯 Conversion Funnel Analysis")
    
    # Calculate funnel metrics
    total_leads = len(df)
    qualified_leads = df['Qualified Lead'].sum() if 'Qualified Lead' in df.columns else 0
    scheduled = df['Scheduled'].sum() if 'Scheduled' in df.columns else 0
    hired = len(df[df['Status'] == 'Hired']) if 'Status' in df.columns else 0
    
    # Funnel chart
    funnel_data = {
        'Stage': ['Total Leads', 'Qualified Leads', 'Scheduled', 'Hired'],
        'Count': [total_leads, qualified_leads, scheduled, hired],
        'Percentage': [100, (qualified_leads/total_leads)*100 if total_leads > 0 else 0,
                      (scheduled/total_leads)*100 if total_leads > 0 else 0,
                      (hired/total_leads)*100 if total_leads > 0 else 0]
    }
    
    fig_funnel = go.Figure(go.Funnel(
        y=funnel_data['Stage'],
        x=funnel_data['Count'],
        textinfo="value+percent initial",
        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    ))
    fig_funnel.update_layout(
        title="Lead Conversion Funnel",
        font_size=14,
        height=500
    )
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    # Conversion rates by matter type
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Matter Type' in df.columns and 'Status' in df.columns:
            conversion_by_type = df.groupby('Matter Type').agg({
                'Status': ['count', lambda x: (x == 'Hired').sum()]
            }).round(2)
            conversion_by_type.columns = ['Total', 'Hired']
            conversion_by_type['Conversion Rate %'] = (conversion_by_type['Hired'] / conversion_by_type['Total'] * 100).round(1)
            conversion_by_type = conversion_by_type.sort_values('Conversion Rate %', ascending=False)
            
            st.subheader("Conversion Rate by Matter Type")
            st.dataframe(conversion_by_type, use_container_width=True)
    
    with col2:
        if 'Created By' in df.columns and 'Status' in df.columns:
            conversion_by_staff = df.groupby('Created By').agg({
                'Status': ['count', lambda x: (x == 'Hired').sum()]
            }).round(2)
            conversion_by_staff.columns = ['Total', 'Hired']
            conversion_by_staff['Conversion Rate %'] = (conversion_by_staff['Hired'] / conversion_by_staff['Total'] * 100).round(1)
            conversion_by_staff = conversion_by_staff.sort_values('Conversion Rate %', ascending=False)
            
            st.subheader("Conversion Rate by Staff Member")
            st.dataframe(conversion_by_staff, use_container_width=True)

def show_lead_sources_page(df: pd.DataFrame):
    """Display lead source analytics."""
    st.header("📈 Lead Source Analytics")
    
    if 'Primary Contact Source' not in df.columns:
        st.warning("Primary Contact Source data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Leads by source
        source_counts = df['Primary Contact Source'].value_counts()
        fig_sources = px.bar(
            x=source_counts.values,
            y=source_counts.index,
            orientation='h',
            title="Leads by Source",
            color=source_counts.values,
            color_continuous_scale='Viridis'
        )
        # Clean hover template
        fig_sources.update_traces(
            hovertemplate="<b>%{y}</b><br>" +
                         "Lead Count: %{x}<br>" +
                         "<extra></extra>"
        )
        fig_sources.update_layout(
            yaxis={'categoryorder':'total ascending'},
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_sources, use_container_width=True)
    
    with col2:
        # ROI by source
        if 'Value' in df.columns and 'Status' in df.columns:
            roi_by_source = df[df['Status'] == 'Hired'].groupby('Primary Contact Source').agg({
                'Value': ['sum', 'count', 'mean']
            }).round(0)
            roi_by_source.columns = ['Total Value', 'Count', 'Avg Value']
            roi_by_source = roi_by_source.sort_values('Total Value', ascending=False)
            
            # Format values for display
            roi_display = roi_by_source.copy()
            roi_display['Total Value'] = roi_display['Total Value'].apply(lambda x: f"${x:,.0f}")
            roi_display['Avg Value'] = roi_display['Avg Value'].apply(lambda x: f"${x:,.0f}")
            
            st.subheader("ROI by Source (Hired Only)")
            st.dataframe(roi_display, use_container_width=True)
    
    # Source performance heatmap
    if 'Status' in df.columns:
        source_status = pd.crosstab(df['Primary Contact Source'], df['Status'], normalize='index') * 100
        
        fig_heatmap = px.imshow(
            source_status.values,
            x=source_status.columns,
            y=source_status.index,
            color_continuous_scale='RdYlBu_r',
            aspect='auto',
            title="Source Performance Heatmap (% by Status)",
            text_auto=".1f"
        )
        # Clean hover template
        fig_heatmap.update_traces(
            hovertemplate="<b>%{y}</b><br>" +
                         "Status: %{x}<br>" +
                         "Percentage: %{z:.1f}%<br>" +
                         "<extra></extra>"
        )
        fig_heatmap.update_layout(
            height=500,
            xaxis_title="Status",
            yaxis_title="Lead Source"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Export individual chart data
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Export Source Counts"):
            csv_buffer = io.StringIO()
            source_counts.to_frame('Lead Count').to_csv(csv_buffer)
            st.download_button(
                "Download Source Counts CSV",
                csv_buffer.getvalue(),
                f"lead_sources_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with col2:
        if st.button("💰 Export ROI Analysis") and 'Value' in df.columns:
            csv_buffer = io.StringIO()
            roi_by_source.to_csv(csv_buffer)
            st.download_button(
                "Download ROI Analysis CSV",
                csv_buffer.getvalue(),
                f"roi_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )

def show_engagement_metrics_page(df: pd.DataFrame):
    """Display engagement metrics analysis."""
    st.header("📞 Engagement Metrics")
    
    # Engagement activity rates
    col1, col2 = st.columns(2)
    
    engagement_cols = ['Scheduled', 'Call', 'Zoom', 'No show']
    available_cols = [col for col in engagement_cols if col in df.columns]
    
    if available_cols:
        with col1:
            engagement_rates = {}
            for col in available_cols:
                if col in df.columns:
                    rate = df[col].sum() / len(df) * 100
                    engagement_rates[col] = rate
            
            fig_engagement = px.bar(
                x=list(engagement_rates.keys()),
                y=list(engagement_rates.values()),
                title="Engagement Activity Rates (%)",
                color=list(engagement_rates.values()),
                color_continuous_scale='Blues'
            )
            fig_engagement.update_layout(showlegend=False)
            st.plotly_chart(fig_engagement, use_container_width=True)
        
        with col2:
            # No-show analysis
            if 'No show' in df.columns and 'Scheduled' in df.columns:
                scheduled_df = df[df['Scheduled'] == True]
                if len(scheduled_df) > 0:
                    no_show_rate = scheduled_df['No show'].sum() / len(scheduled_df) * 100
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=no_show_rate,
                        title={'text': "No-Show Rate (%)"},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "darkred"},
                               'steps': [{'range': [0, 25], 'color': "lightgray"},
                                        {'range': [25, 50], 'color': "yellow"},
                                        {'range': [50, 100], 'color': "red"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 90}}
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Correlation matrix
    numeric_cols = ['Scheduled', 'Call', 'Zoom', 'Qualified Lead']
    available_numeric = [col for col in numeric_cols if col in df.columns]
    
    if len(available_numeric) >= 2:
        st.subheader("Engagement Activity Correlation")
        correlation_matrix = df[available_numeric].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Engagement Activities Correlation Matrix"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

def show_geographic_analysis_page(df: pd.DataFrame):
    """Display enhanced geographic analysis with interactive maps."""
    st.header("🗺️ Geographic Analysis")
    
    if 'Primary Contact State' not in df.columns:
        st.warning("Geographic data not available")
        return
    
    # Create tabs for different geographic views
    tab1, tab2, tab3 = st.tabs(["State Map", "City Analysis", "Regional Metrics"])
    
    with tab1:
        # Interactive state map
        st.subheader("Lead Conversion Rate by State")
        fig_map = create_geographic_map(df)
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
        
        # State performance table
        state_metrics = df.groupby('Primary Contact State').agg({
            'Status': ['count', lambda x: (x == 'Hired').sum()],
            'Total Value': 'sum'
        }).round(0)
        state_metrics.columns = ['Total Leads', 'Conversions', 'Revenue']
        state_metrics['Conversion Rate %'] = (state_metrics['Conversions'] / state_metrics['Total Leads'] * 100).round(1)
        state_metrics = state_metrics.sort_values('Revenue', ascending=False).head(10)
        
        # Format for display
        display_metrics = state_metrics.copy()
        display_metrics['Revenue'] = display_metrics['Revenue'].apply(lambda x: f"${x:,.0f}")
        
        st.subheader("Top 10 States by Revenue")
        st.dataframe(display_metrics, use_container_width=True)
    
    with tab2:
        # City bubble map
        st.subheader("City Performance Analysis")
        fig_bubble = create_city_bubble_map(df)
        if fig_bubble:
            st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Top cities table
        if 'Primary Contact City' in df.columns:
            city_performance = df.groupby(['Primary Contact City', 'Primary Contact State']).agg({
                'Status': ['count', lambda x: (x == 'Hired').sum()],
                'Total Value': 'sum'
            }).round(0)
            city_performance.columns = ['Total Leads', 'Conversions', 'Revenue']
            city_performance['Conversion Rate %'] = (city_performance['Conversions'] / city_performance['Total Leads'] * 100).round(1)
            city_performance = city_performance.sort_values('Revenue', ascending=False).head(15)
            
            # Format for display
            display_city = city_performance.copy()
            display_city['Revenue'] = display_city['Revenue'].apply(lambda x: f"${x:,.0f}")
            
            st.subheader("Top 15 Cities by Revenue")
            st.dataframe(display_city, use_container_width=True)
    
    with tab3:
        # Regional analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Create regions based on states
            def get_region(state):
                northeast = ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA']
                midwest = ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD']
                south = ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX']
                west = ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
                
                if state in northeast:
                    return 'Northeast'
                elif state in midwest:
                    return 'Midwest'
                elif state in south:
                    return 'South'
                elif state in west:
                    return 'West'
                else:
                    return 'Other'
            
            df['Region'] = df['Primary Contact State'].apply(get_region)
            
            region_metrics = df.groupby('Region').agg({
                'Status': ['count', lambda x: (x == 'Hired').sum()],
                'Total Value': 'sum'
            }).round(0)
            region_metrics.columns = ['Total Leads', 'Conversions', 'Revenue']
            region_metrics['Conversion Rate %'] = (region_metrics['Conversions'] / region_metrics['Total Leads'] * 100).round(1)
            
            fig_region = px.bar(
                region_metrics.reset_index(),
                x='Region',
                y='Revenue',
                title='Revenue by Region',
                color='Conversion Rate %',
                color_continuous_scale='Viridis'
            )
            fig_region.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                             "Revenue: $%{y:,.0f}<br>" +
                             "Conversion Rate: %{color:.1f}%<br>" +
                             "<extra></extra>"
            )
            st.plotly_chart(fig_region, use_container_width=True)
        
        with col2:
            # Heatmap of conversions by state and practice area
            if 'Practice Area' in df.columns:
                state_practice = pd.crosstab(
                    df['Primary Contact State'], 
                    df['Practice Area'], 
                    df['Status'] == 'Hired', 
                    aggfunc='sum'
                )
                
                # Get top 10 states and top 5 practice areas
                top_states = df['Primary Contact State'].value_counts().head(10).index
                top_practices = df['Practice Area'].value_counts().head(5).index
                
                state_practice_filtered = state_practice.loc[
                    state_practice.index.isin(top_states),
                    state_practice.columns.isin(top_practices)
                ]
                
                fig_heatmap = px.imshow(
                    state_practice_filtered,
                    title='Conversions: State vs Practice Area',
                    color_continuous_scale='YlOrRd',
                    aspect='auto',
                    text_auto=True
                )
                fig_heatmap.update_traces(
                    hovertemplate="<b>State:</b> %{y}<br>" +
                                 "<b>Practice Area:</b> %{x}<br>" +
                                 "<b>Conversions:</b> %{z}<br>" +
                                 "<extra></extra>"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

def show_financial_dashboard_page(df: pd.DataFrame):
    """Display financial analysis dashboard with retainer breakdown."""
    st.header("💰 Financial Dashboard")
    
    if 'Total Value' not in df.columns:
        st.warning("Financial data not available")
        return
    
    # Filter for hired matters only for revenue analysis
    hired_df = df[df['Status'] == 'Hired'] if 'Status' in df.columns else df[df['Total Value'].notna()]
    
    # Retainer breakdown over time
    if 'Created' in hired_df.columns and len(hired_df) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Cumulative revenue over time
            daily_revenue = hired_df.groupby(hired_df['Created'].dt.date).agg({
                'Total Value': 'sum',
                'Nonrefundable Retainer': 'sum',
                'Refundable Retainer': 'sum'
            }).fillna(0)
            
            daily_revenue['Cumulative Total'] = daily_revenue['Total Value'].cumsum()
            daily_revenue['Cumulative NR'] = daily_revenue['Nonrefundable Retainer'].cumsum()
            daily_revenue['Cumulative R'] = daily_revenue['Refundable Retainer'].cumsum()
            
            fig_cumulative = go.Figure()
            fig_cumulative.add_trace(go.Scatter(
                x=daily_revenue.index,
                y=daily_revenue['Cumulative Total'],
                name='Total Revenue',
                line=dict(color=COLORS['primary'], width=3)
            ))
            if 'Nonrefundable Retainer' in hired_df.columns:
                fig_cumulative.add_trace(go.Scatter(
                    x=daily_revenue.index,
                    y=daily_revenue['Cumulative NR'],
                    name='Nonrefundable',
                    line=dict(color=COLORS['success'], width=2)
                ))
            if 'Refundable Retainer' in hired_df.columns:
                fig_cumulative.add_trace(go.Scatter(
                    x=daily_revenue.index,
                    y=daily_revenue['Cumulative R'],
                    name='Refundable',
                    line=dict(color=COLORS['accent'], width=2)
                ))
            
            fig_cumulative.update_layout(
                title='Cumulative Revenue Over Time',
                xaxis_title='Date',
                yaxis_title='Cumulative Revenue ($)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_cumulative, use_container_width=True)
        
        with col2:
            # Revenue distribution by practice area
            if 'Practice Area' in hired_df.columns:
                practice_revenue = hired_df.groupby('Practice Area').agg({
                    'Total Value': 'sum',
                    'Nonrefundable Retainer': 'mean',
                    'Refundable Retainer': 'mean'
                }).round(0)
                practice_revenue = practice_revenue.sort_values('Total Value', ascending=False).head(5)
                
                fig_practice = px.bar(
                    practice_revenue.reset_index(),
                    x='Practice Area',
                    y='Total Value',
                    title='Total Revenue by Practice Area',
                    color='Total Value',
                    color_continuous_scale='Viridis'
                )
                fig_practice.update_traces(
                    hovertemplate="<b>%{x}</b><br>" +
                                 "Total Revenue: $%{y:,.0f}<br>" +
                                 "<extra></extra>"
                )
                fig_practice.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_practice, use_container_width=True)
    
    # Detailed financial metrics table
    st.subheader("Financial Performance by Practice Area")
    if 'Practice Area' in hired_df.columns:
        financial_summary = hired_df.groupby('Practice Area').agg({
            'Total Value': ['sum', 'mean', 'count'],
            'Nonrefundable Retainer': ['sum', 'mean'],
            'Refundable Retainer': ['sum', 'mean']
        }).round(0)
        
        financial_summary.columns = [
            'Total Revenue', 'Avg Revenue', 'Cases',
            'Total NR', 'Avg NR',
            'Total R', 'Avg R'
        ]
        
        financial_summary = financial_summary.sort_values('Total Revenue', ascending=False)
        
        # Format for display
        display_financial = financial_summary.copy()
        currency_cols = ['Total Revenue', 'Avg Revenue', 'Total NR', 'Avg NR', 'Total R', 'Avg R']
        for col in currency_cols:
            if col in display_financial.columns:
                display_financial[col] = display_financial[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "$0")
        
        st.dataframe(display_financial, use_container_width=True)
    
    # Monthly revenue trends
    if 'Month' in hired_df.columns and len(hired_df) > 0:
        monthly_revenue = hired_df.groupby('Month').agg({
            'Total Value': 'sum',
            'Nonrefundable Retainer': 'sum', 
            'Refundable Retainer': 'sum'
        }).fillna(0)
        
        fig_monthly = px.bar(
            monthly_revenue.reset_index(),
            x=monthly_revenue.index.astype(str),
            y=['Total Value', 'Nonrefundable Retainer', 'Refundable Retainer'],
            title='Monthly Revenue by Retainer Type',
            color_discrete_map={
                'Total Value': COLORS['primary'],
                'Nonrefundable Retainer': COLORS['success'],
                'Refundable Retainer': COLORS['accent']
            }
        )
        fig_monthly.update_layout(
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            barmode='group'
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

def show_staff_performance_page(df: pd.DataFrame):
    """Display staff performance analysis."""
    st.header("👥 Staff Performance")
    
    if 'Created By' not in df.columns:
        st.warning("Staff data not available")
        return
    
    # Staff performance metrics
    staff_performance = df.groupby('Created By').agg({
        'Status': ['count', lambda x: (x == 'Hired').sum()],
        'Value': lambda x: x[df.loc[x.index, 'Status'] == 'Hired'].sum() if 'Status' in df.columns else x.sum(),
        'Scheduled': 'sum' if 'Scheduled' in df.columns else lambda x: 0
    }).round(0)
    
    if 'Status' in df.columns:
        staff_performance.columns = ['Total Leads', 'Hired', 'Revenue Generated', 'Scheduled']
        staff_performance['Conversion Rate %'] = (staff_performance['Hired'] / staff_performance['Total Leads'] * 100).round(1)
    else:
        staff_performance.columns = ['Total Leads', 'Revenue Generated', 'Scheduled']
    
    # Leaderboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performers by Revenue")
        top_revenue = staff_performance.nlargest(5, 'Revenue Generated')[['Revenue Generated', 'Total Leads']]
        st.dataframe(top_revenue, use_container_width=True)
    
    with col2:
        if 'Conversion Rate %' in staff_performance.columns:
            st.subheader("🎯 Top Performers by Conversion Rate")
            top_conversion = staff_performance.nlargest(5, 'Conversion Rate %')[['Conversion Rate %', 'Total Leads']]
            st.dataframe(top_conversion, use_container_width=True)
    
    # Staff performance chart
    if len(staff_performance) > 0:
        fig_staff = px.scatter(
            staff_performance.reset_index(),
            x='Total Leads',
            y='Revenue Generated',
            size='Conversion Rate %' if 'Conversion Rate %' in staff_performance.columns else 'Revenue Generated',
            hover_name='Created By',
            title="Staff Performance: Leads vs Revenue",
            color='Conversion Rate %' if 'Conversion Rate %' in staff_performance.columns else 'Revenue Generated',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_staff, use_container_width=True)
    
    # Full staff performance table
    st.subheader("Complete Staff Performance")
    st.dataframe(staff_performance.sort_values('Revenue Generated', ascending=False), use_container_width=True)

def show_time_analysis_page(df: pd.DataFrame):
    """Display time-based analysis."""
    st.header("⏰ Time-Based Analysis")
    
    if 'Created' not in df.columns:
        st.warning("Date data not available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Leads by day of week
        if 'Day_of_Week' in df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = df['Day_of_Week'].value_counts().reindex(day_order, fill_value=0)
            
            fig_days = px.bar(
                x=day_counts.index,
                y=day_counts.values,
                title="Leads by Day of Week",
                color=day_counts.values,
                color_continuous_scale='Blues'
            )
            fig_days.update_layout(showlegend=False)
            st.plotly_chart(fig_days, use_container_width=True)
    
    with col2:
        # Leads by hour
        if 'Hour' in df.columns:
            hour_counts = df['Hour'].value_counts().sort_index()
            
            fig_hours = px.bar(
                x=hour_counts.index,
                y=hour_counts.values,
                title="Leads by Hour of Day",
                color=hour_counts.values,
                color_continuous_scale='Greens'
            )
            fig_hours.update_layout(
                xaxis_title="Hour",
                showlegend=False
            )
            st.plotly_chart(fig_hours, use_container_width=True)
    
    # Monthly trends
    if 'Month' in df.columns:
        monthly_data = df.groupby('Month').agg({
            'Status': 'count',
            'Value': lambda x: x[df.loc[x.index, 'Status'] == 'Hired'].sum() if 'Status' in df.columns else x.sum()
        }).round(0)
        monthly_data.columns = ['Lead Count', 'Revenue']
        
        fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_monthly.add_trace(
            go.Scatter(x=monthly_data.index.astype(str), y=monthly_data['Lead Count'], name="Lead Count"),
            secondary_y=False
        )
        
        fig_monthly.add_trace(
            go.Scatter(x=monthly_data.index.astype(str), y=monthly_data['Revenue'], name="Revenue"),
            secondary_y=True
        )
        
        fig_monthly.update_xaxes(title_text="Month")
        fig_monthly.update_yaxes(title_text="Lead Count", secondary_y=False)
        fig_monthly.update_yaxes(title_text="Revenue ($)", secondary_y=True)
        fig_monthly.update_layout(title_text="Monthly Trends: Leads vs Revenue")
        
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Time-based performance table
    if 'Day_of_Week' in df.columns and 'Status' in df.columns:
        time_performance = df.groupby('Day_of_Week').agg({
            'Status': ['count', lambda x: (x == 'Hired').sum()],
            'Value': lambda x: x[df.loc[x.index, 'Status'] == 'Hired'].sum()
        }).round(0)
        time_performance.columns = ['Total Leads', 'Hired', 'Revenue']
        time_performance['Conversion Rate %'] = (time_performance['Hired'] / time_performance['Total Leads'] * 100).round(1)
        time_performance = time_performance.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        st.subheader("Performance by Day of Week")
        st.dataframe(time_performance, use_container_width=True)

if __name__ == "__main__":
    main()