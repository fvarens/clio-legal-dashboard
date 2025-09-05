"""
Enhanced CLIO Legal Matters Dashboard with Lead Input System

This module adds a comprehensive lead input and management system to the existing
CLIO dashboard, allowing intake staff to manually add leads and sales staff to
update outcomes.

Author: Claude Code
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import uuid
import plotly.express as px
import plotly.graph_objects as go

# Constants for the lead input system
INTAKE_STAFF = [
    "Fabio Varens Ojeda",
    "Llewelyn Abrahams", 
    "Briana Markland",
    "Meylin Lainez",
    "Camila Mosquera"
]

MATTER_TYPES = [
    "Family Law",
    "Criminal Law", 
    "Civil Law",
    "Real Estate Law",
    "Estate & Probate Law"
]

LEAD_SOURCES = {
    "ARAG": {"has_contact": False},
    "Associated with Contact Lead": {"has_contact": True, "contact_type": "Contact Lead"},
    "Avvo": {"has_contact": False},
    "Avvo Text": {"has_contact": False},
    "Bing": {"has_contact": False},
    "Chat GPT": {"has_contact": False},
    "Event": {"has_contact": False},
    "Facebook": {"has_contact": False},
    "Google": {"has_contact": False},
    "Google LSA": {"has_contact": False},
    "Instagram": {"has_contact": False},
    "Justia": {"has_contact": False},
    "Lawyers.com": {"has_contact": False},
    "Leah": {"has_contact": False},
    "LinkedIn": {"has_contact": False},
    "Martindale Nolo": {"has_contact": False},
    "Ngage": {"has_contact": False},
    "Previous Client": {"has_contact": False},
    "Referral - Business Associate": {"has_contact": True, "contact_type": "Business Associate"},
    "Referral - Client": {"has_contact": True, "contact_type": "Client Name"},
    "Referral - Attorney": {"has_contact": True, "contact_type": "Attorney Name"},
    "Referral - Friend": {"has_contact": True, "contact_type": "Friend Name"},
    "Schecter Firm": {"has_contact": False},
    "TikTok": {"has_contact": False},
    "Twitter": {"has_contact": False},
    "Walk-In": {"has_contact": False},
    "Website": {"has_contact": False},
    "Yelp": {"has_contact": False},
    "YouTube": {"has_contact": False}
}

UNQUALIFIED_REASONS = [
    "No finances",
    "Pro-bono",
    "Indigent", 
    "Competition",
    "Not a case we handle",
    "Mentally Unwell",
    "Bad Potential Client"
]

# File paths for storing lead data
LEADS_DATA_FILE = "manual_leads_data.csv"
LEAD_UPDATES_FILE = "lead_updates.json"

# Color scheme matching main dashboard
COLORS = {
    'primary': '#1f2937',
    'secondary': '#6b7280', 
    'accent': '#d97706',
    'success': '#059669',
    'warning': '#dc2626',
    'background': '#f9fafb',
    'text': '#111827'
}

def initialize_lead_storage():
    """Initialize storage files if they don't exist."""
    if not os.path.exists(LEADS_DATA_FILE):
        # Create empty DataFrame with all necessary columns
        columns = [
            'Lead_ID', 'Created', 'Created By', 'Matter Type', 'Primary Contact Source',
            'Contact_Name', 'Primary Contact', 'Primary Contact Email', 'Primary Contact Phone', 
            'Initial_Result', 'Meeting_Type', 'Show_Status', 'Outcome', 'Qualification_Status',
            'Marketing_Qualified', 'Sales_Qualified', 'Unqualified_Reason',
            'Status', 'Last_Updated', 'Updated_By', 'Nonrefundable Retainer',
            'Refundable Retainer', 'Total Value', 'Follow_Up_Date', 'Notes',
            'Scheduled', 'Call', 'Zoom', 'No show', 'Qualified Lead'
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(LEADS_DATA_FILE, index=False)
    
    if not os.path.exists(LEAD_UPDATES_FILE):
        with open(LEAD_UPDATES_FILE, 'w') as f:
            json.dump({}, f)

def load_manual_leads():
    """Load manually entered leads from storage."""
    if os.path.exists(LEADS_DATA_FILE):
        df = pd.read_csv(LEADS_DATA_FILE)
        # Convert date columns
        date_columns = ['Created', 'Last_Updated', 'Follow_Up_Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df
    return pd.DataFrame()

def save_lead(lead_data):
    """Save a new lead to the storage."""
    df = load_manual_leads()
    new_lead = pd.DataFrame([lead_data])
    df = pd.concat([df, new_lead], ignore_index=True)
    df.to_csv(LEADS_DATA_FILE, index=False)
    return True

def update_lead(lead_id, updates):
    """Update an existing lead."""
    df = load_manual_leads()
    if lead_id in df['Lead_ID'].values:
        idx = df[df['Lead_ID'] == lead_id].index[0]
        for key, value in updates.items():
            df.at[idx, key] = value
        df.at[idx, 'Last_Updated'] = datetime.now()
        df.to_csv(LEADS_DATA_FILE, index=False)
        
        # Log the update
        try:
            with open(LEAD_UPDATES_FILE, 'r') as f:
                update_log = json.load(f)
        except:
            update_log = {}
        
        if lead_id not in update_log:
            update_log[lead_id] = []
        
        update_log[lead_id].append({
            'timestamp': datetime.now().isoformat(),
            'updates': updates,
            'updated_by': updates.get('Updated_By', 'Unknown')
        })
        
        with open(LEAD_UPDATES_FILE, 'w') as f:
            json.dump(update_log, f)
        
        return True
    return False

def show_lead_input_page():
    """Display the lead input form page."""
    st.header("➕ New Lead Input")
    st.markdown("---")
    
    # Initialize storage
    initialize_lead_storage()
    
    # Initialize session state for dynamic fields
    if 'lead_source' not in st.session_state:
        st.session_state.lead_source = list(LEAD_SOURCES.keys())[0]
    if 'initial_result' not in st.session_state:
        st.session_state.initial_result = "Scheduled"
    if 'show_qualified_options' not in st.session_state:
        st.session_state.show_qualified_options = False
    
    st.markdown("### 👤 Lead Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        intake_name = st.selectbox(
            "Intake Staff Name *",
            options=INTAKE_STAFF,
            help="Select the intake staff member",
            key="intake_staff"
        )
    
    with col2:
        matter_type = st.selectbox(
            "Matter Type *",
            options=MATTER_TYPES,
            help="Select the type of legal matter",
            key="matter_type"
        )
    
    with col3:
        lead_source = st.selectbox(
            "Lead Source *",
            options=sorted(LEAD_SOURCES.keys()),
            help="Select the source of this lead",
            key="lead_source"
        )
    
    # Conditional contact name field (outside form)
    contact_name = ""
    if LEAD_SOURCES[lead_source]["has_contact"]:
        contact_type = LEAD_SOURCES[lead_source]["contact_type"]
        contact_name = st.text_input(
            f"{contact_type} Name *",
            placeholder=f"Enter {contact_type.lower()} name",
            help=f"Please provide the {contact_type.lower()} who referred this lead",
            key="contact_name"
        )
    
    st.markdown("### 📞 Contact Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        client_name = st.text_input(
            "Client Name *",
            placeholder="Enter client's full name",
            key="client_name"
        )
    
    with col2:
        client_email = st.text_input(
            "Client Email",
            placeholder="email@example.com",
            key="client_email"
        )
    
    with col3:
        client_phone = st.text_input(
            "Client Phone",
            placeholder="(xxx) xxx-xxxx",
            key="client_phone"
        )
    
    st.markdown("### 📊 Initial Outcome")
    
    col1, col2 = st.columns(2)
    
    with col1:
        initial_result = st.selectbox(
            "Initial Result *",
            options=["Scheduled", "Unqualified", "Follow Up Required", "No Response"],
            help="What was the initial outcome of this lead?",
            key="initial_result"
        )
    
    meeting_type = None
    with col2:
        if initial_result == "Scheduled":
            meeting_type = st.selectbox(
                "Meeting Type",
                options=["Zoom", "Call"],
                help="How will the meeting be conducted?",
                key="meeting_type"
            )
            
            st.info("💡 Sales team will update show status and outcome later")
    
    # Qualification section
    marketing_qualified = False
    sales_qualified = False
    unqualified_reason = None
    
    if initial_result == "Unqualified":
        st.markdown("### ❌ Unqualified Reason")
        unqualified_reason = st.selectbox(
            "Reason for Disqualification",
            options=UNQUALIFIED_REASONS,
            key="unqualified_reason"
        )
    else:
        st.markdown("### ✅ Qualification Status")
        
        is_qualified = st.checkbox("Lead is Qualified", key="is_qualified")
        
        if is_qualified:
            col1, col2 = st.columns(2)
            with col1:
                marketing_qualified = st.checkbox("Marketing Qualified", key="marketing_qualified")
            with col2:
                sales_qualified = st.checkbox("Sales Qualified", key="sales_qualified")
    
    # Notes section
    st.markdown("### 📝 Additional Notes")
    notes = st.text_area(
        "Notes (Optional)",
        placeholder="Add any additional information about this lead...",
        height=100,
        key="notes"
    )
    
    # Submit button
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if st.button("💾 Save Lead", use_container_width=True, type="primary"):
            # Validation
            errors = []
            
            if not client_name:
                errors.append("Please enter the client's name")
            
            if LEAD_SOURCES[lead_source]["has_contact"] and not contact_name:
                errors.append(f"Please enter the {LEAD_SOURCES[lead_source]['contact_type']} name")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Create lead data
                lead_data = {
                    'Lead_ID': str(uuid.uuid4()),
                    'Created': datetime.now(),
                    'Created By': intake_name,
                    'Matter Type': matter_type,
                    'Primary Contact Source': lead_source,
                    'Contact_Name': contact_name,
                    'Primary Contact': client_name,
                    'Primary Contact Email': client_email,
                    'Primary Contact Phone': client_phone,
                    'Initial_Result': initial_result,
                    'Meeting_Type': meeting_type,
                    'Show_Status': None,
                    'Outcome': None,
                    'Qualification_Status': 'Qualified' if (marketing_qualified or sales_qualified) else ('Unqualified' if initial_result == 'Unqualified' else 'Pending'),
                    'Marketing_Qualified': marketing_qualified,
                    'Sales_Qualified': sales_qualified,
                    'Unqualified_Reason': unqualified_reason,
                    'Status': 'Pending' if initial_result == 'Scheduled' else initial_result,
                    'Last_Updated': datetime.now(),
                    'Updated_By': intake_name,
                    'Nonrefundable Retainer': 0,
                    'Refundable Retainer': 0,
                    'Total Value': 0,
                    'Follow_Up_Date': None,
                    'Notes': notes,
                    'Scheduled': True if initial_result == 'Scheduled' else False,
                    'Call': True if meeting_type == 'Call' else False,
                    'Zoom': True if meeting_type == 'Zoom' else False,
                    'No show': False,
                    'Qualified Lead': True if (marketing_qualified or sales_qualified) else False
                }
                
                # Save lead
                if save_lead(lead_data):
                    st.success(f"✅ Lead successfully saved! Lead ID: {lead_data['Lead_ID'][:8]}...")
                    st.balloons()
                    
                    # Clear form by resetting session state
                    for key in ['client_name', 'client_email', 'client_phone', 'notes', 
                               'contact_name', 'is_qualified', 'marketing_qualified', 
                               'sales_qualified']:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.rerun()
                else:
                    st.error("Failed to save lead. Please try again.")

def show_lead_update_page():
    """Display the lead update page for sales team."""
    st.header("🔄 Update Lead Status")
    st.markdown("---")
    
    # Load existing leads
    df = load_manual_leads()
    
    if df.empty:
        st.warning("No leads available to update")
        return
    
    # Filter for pending scheduled leads
    pending_leads = df[
        (df['Initial_Result'] == 'Scheduled') & 
        (df['Show_Status'].isna() | (df['Show_Status'] == ''))
    ].copy()
    
    if pending_leads.empty:
        st.info("No scheduled leads pending update")
        
        # Show recent updates for reference
        recent_updates = df[df['Show_Status'].notna()].sort_values('Last_Updated', ascending=False).head(5)
        if not recent_updates.empty:
            st.subheader("Recent Updates")
            display_cols = ['Primary Contact', 'Matter Type', 'Show_Status', 'Outcome', 'Status', 'Last_Updated']
            available_cols = [col for col in display_cols if col in recent_updates.columns]
            st.dataframe(recent_updates[available_cols], use_container_width=True)
    else:
        # Create lead selector
        lead_options = {}
        for _, lead in pending_leads.iterrows():
            display_name = f"{lead['Primary Contact']} - {lead['Matter Type']} ({lead['Created'].strftime('%Y-%m-%d')})"
            lead_options[display_name] = lead['Lead_ID']
        
        selected_display = st.selectbox(
            "Select Lead to Update",
            options=list(lead_options.keys())
        )
        
        if selected_display:
            selected_lead_id = lead_options[selected_display]
            selected_lead = df[df['Lead_ID'] == selected_lead_id].iloc[0]
            
            # Display lead information
            st.markdown("### Lead Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Client:** {selected_lead['Primary Contact']}")
                st.write(f"**Matter Type:** {selected_lead['Matter Type']}")
            
            with col2:
                st.write(f"**Source:** {selected_lead['Primary Contact Source']}")
                st.write(f"**Meeting Type:** {selected_lead.get('Meeting_Type', 'Not specified')}")
            
            with col3:
                st.write(f"**Created By:** {selected_lead['Created By']}")
                st.write(f"**Created:** {selected_lead['Created'].strftime('%Y-%m-%d %H:%M')}")
            
            st.markdown("---")
            
            # Update form
            with st.form("update_lead_form"):
                st.markdown("### 📊 Meeting Outcome")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    show_status = st.selectbox(
                        "Show Status *",
                        options=["Show", "No Show"]
                    )
                
                with col2:
                    if show_status == "Show":
                        outcome = st.selectbox(
                            "Outcome *",
                            options=["Hired", "Did not hire", "Unqualified", "Following up"]
                        )
                    else:  # No Show
                        outcome = st.selectbox(
                            "Outcome *",
                            options=["Following up", "Did not hire", "Rescheduled"]
                        )
                
                # Financial information if hired
                nonrefundable = 0
                refundable = 0
                
                if outcome == "Hired":
                    st.markdown("### 💰 Retainer Information")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        nonrefundable = st.number_input(
                            "Nonrefundable Retainer ($)",
                            min_value=0,
                            step=100,
                            help="Enter the nonrefundable retainer amount"
                        )
                    
                    with col2:
                        refundable = st.number_input(
                            "Refundable Retainer ($)",
                            min_value=0,
                            step=100,
                            help="Enter the refundable retainer amount"
                        )
                
                # Follow-up information
                follow_up_date = None
                if outcome in ["Following up", "Rescheduled"]:
                    st.markdown("### 📅 Follow-up")
                    follow_up_date = st.date_input(
                        "Follow-up Date",
                        min_value=datetime.now().date()
                    )
                
                # Additional notes
                st.markdown("### 📝 Update Notes")
                update_notes = st.text_area(
                    "Notes",
                    placeholder="Add any notes about this interaction..."
                )
                
                # Updater name
                updater_name = st.selectbox(
                    "Updated By *",
                    options=INTAKE_STAFF
                )
                
                # Submit button
                col1, col2, col3 = st.columns([2,1,2])
                with col2:
                    update_submitted = st.form_submit_button(
                        "💾 Update Lead",
                        use_container_width=True,
                        type="primary"
                    )
                
                if update_submitted:
                    # Determine final status
                    if outcome == "Hired":
                        final_status = "Hired"
                    elif outcome in ["Did not hire", "Unqualified"]:
                        final_status = "Not Hired"
                    elif outcome in ["Following up", "Rescheduled"]:
                        final_status = "Follow Up Required"
                    else:
                        final_status = outcome
                    
                    # Prepare updates
                    updates = {
                        'Show_Status': show_status,
                        'Outcome': outcome,
                        'Status': final_status,
                        'Updated_By': updater_name,
                        'Nonrefundable Retainer': nonrefundable,
                        'Refundable Retainer': refundable,
                        'Total Value': nonrefundable + refundable,
                        'No show': show_status == "No Show",
                        'Qualified Lead': outcome not in ["Unqualified", "Did not hire"],
                        'Notes': (selected_lead.get('Notes', '') + f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {update_notes}") if update_notes else selected_lead.get('Notes', '')
                    }
                    
                    if follow_up_date:
                        updates['Follow_Up_Date'] = follow_up_date
                    
                    # Update the lead
                    if update_lead(selected_lead_id, updates):
                        st.success("✅ Lead successfully updated!")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("Failed to update lead. Please try again.")

def show_lead_management_dashboard():
    """Display comprehensive lead management dashboard."""
    st.header("📊 Lead Management Dashboard")
    st.markdown("---")
    
    # Load manual leads
    df = load_manual_leads()
    
    if df.empty:
        st.info("No manual leads have been entered yet. Go to 'Lead Input' to add leads.")
        return
    
    # Dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "👥 By Staff", 
        "📊 By Source", 
        "⏰ Pending Actions",
        "📋 Full Lead List"
    ])
    
    with tab1:
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_leads = len(df)
            st.metric("Total Manual Leads", total_leads)
        
        with col2:
            hired_leads = len(df[df['Status'] == 'Hired'])
            conversion_rate = (hired_leads / total_leads * 100) if total_leads > 0 else 0
            st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
        
        with col3:
            total_value = df['Total Value'].sum()
            st.metric("Total Revenue", f"${total_value:,.0f}")
        
        with col4:
            pending_leads = len(df[df['Status'].isin(['Pending', 'Follow Up Required'])])
            st.metric("Pending Leads", pending_leads)
        
        # Lead status distribution
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = df['Status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Lead Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_status, use_container_width=True)
        
        with col2:
            # Leads by matter type
            matter_counts = df['Matter Type'].value_counts()
            fig_matter = px.bar(
                x=matter_counts.values,
                y=matter_counts.index,
                orientation='h',
                title="Leads by Matter Type",
                color=matter_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_matter, use_container_width=True)
        
        # Timeline chart
        if not df.empty:
            daily_leads = df.groupby(df['Created'].dt.date).size().reset_index(name='Count')
            fig_timeline = px.line(
                daily_leads,
                x='Created',
                y='Count',
                title='Daily Lead Volume',
                markers=True
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
    
    with tab2:
        # Staff performance
        staff_metrics = df.groupby('Created By').agg({
            'Lead_ID': 'count',
            'Status': lambda x: (x == 'Hired').sum(),
            'Total Value': 'sum'
        }).reset_index()
        staff_metrics.columns = ['Staff', 'Total Leads', 'Conversions', 'Revenue']
        staff_metrics['Conversion Rate %'] = (staff_metrics['Conversions'] / staff_metrics['Total Leads'] * 100).round(1)
        
        # Display metrics
        for _, staff in staff_metrics.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(staff['Staff'], f"{staff['Total Leads']} leads")
            with col2:
                st.metric("Conversions", staff['Conversions'])
            with col3:
                st.metric("Conv. Rate", f"{staff['Conversion Rate %']:.1f}%")
            with col4:
                st.metric("Revenue", f"${staff['Revenue']:,.0f}")
            st.markdown("---")
    
    with tab3:
        # Source analysis
        source_metrics = df.groupby('Primary Contact Source').agg({
            'Lead_ID': 'count',
            'Status': lambda x: (x == 'Hired').sum(),
            'Total Value': 'sum'
        }).reset_index()
        source_metrics.columns = ['Source', 'Leads', 'Conversions', 'Revenue']
        source_metrics = source_metrics.sort_values('Revenue', ascending=False)
        
        # Top sources chart
        fig_sources = px.bar(
            source_metrics.head(10),
            x='Revenue',
            y='Source',
            orientation='h',
            title='Top 10 Lead Sources by Revenue',
            color='Conversions',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_sources, use_container_width=True)
        
        # Source performance table
        st.dataframe(
            source_metrics.style.format({
                'Revenue': '${:,.0f}'
            }),
            use_container_width=True
        )
    
    with tab4:
        # Pending actions
        st.subheader("🔔 Leads Requiring Action")
        
        # Scheduled meetings pending outcome
        pending_meetings = df[
            (df['Initial_Result'] == 'Scheduled') & 
            (df['Show_Status'].isna() | (df['Show_Status'] == ''))
        ]
        
        if not pending_meetings.empty:
            st.markdown("### Scheduled Meetings Pending Update")
            for _, lead in pending_meetings.iterrows():
                col1, col2, col3, col4 = st.columns([2,2,2,1])
                with col1:
                    st.write(f"**{lead['Primary Contact']}**")
                with col2:
                    st.write(f"{lead['Matter Type']}")
                with col3:
                    st.write(f"Created: {lead['Created'].strftime('%Y-%m-%d')}")
                with col4:
                    st.write("⏰ Pending")
        
        # Follow-ups due
        if 'Follow_Up_Date' in df.columns:
            follow_ups = df[
                df['Follow_Up_Date'].notna() & 
                (df['Follow_Up_Date'] <= pd.Timestamp.now())
            ]
            
            if not follow_ups.empty:
                st.markdown("### Follow-ups Due Today")
                for _, lead in follow_ups.iterrows():
                    st.write(f"- **{lead['Primary Contact']}** ({lead['Matter Type']}) - Due: {lead['Follow_Up_Date'].strftime('%Y-%m-%d')}")
    
    with tab5:
        # Full lead list
        st.subheader("Complete Lead List")
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                options=df['Status'].unique().tolist(),
                default=df['Status'].unique().tolist()
            )
        
        with col2:
            staff_filter = st.multiselect(
                "Filter by Staff",
                options=df['Created By'].unique().tolist(),
                default=df['Created By'].unique().tolist()
            )
        
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(df['Created'].min().date(), df['Created'].max().date())
            )
        
        # Apply filters
        if len(date_range) == 2:
            filtered_df = df[
                (df['Status'].isin(status_filter)) &
                (df['Created By'].isin(staff_filter)) &
                (df['Created'].dt.date >= date_range[0]) &
                (df['Created'].dt.date <= date_range[1])
            ]
        else:
            filtered_df = df[
                (df['Status'].isin(status_filter)) &
                (df['Created By'].isin(staff_filter))
            ]
        
        # Display filtered data
        display_columns = ['Created', 'Primary Contact', 'Matter Type', 
                          'Primary Contact Source', 'Status', 'Total Value', 
                          'Created By']
        available_display_columns = [col for col in display_columns if col in filtered_df.columns]
        display_df = filtered_df[available_display_columns].copy()
        
        if 'Created' in display_df.columns:
            display_df['Created'] = display_df['Created'].dt.strftime('%Y-%m-%d %H:%M')
        if 'Total Value' in display_df.columns:
            display_df['Total Value'] = display_df['Total Value'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Export button
        if st.button("📥 Export Lead Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"manual_leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# CSS Styling for the lead input interface
LEAD_INPUT_CSS = """
<style>
    /* Form styling */
    .stForm {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        background-color: white;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Select box styling */
    .stSelectbox > div > div > select {
        background-color: white;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
        cursor: pointer;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        cursor: pointer;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        animation: slideIn 0.5s;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Card styling for metrics */
    .lead-metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        transition: transform 0.3s;
        margin: 0.5rem 0;
    }
    
    .lead-metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 25px rgba(0,0,0,0.15);
    }
    
    /* Header styling for lead pages */
    .lead-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
</style>
"""