# CLIO Legal Matters Dashboard

A comprehensive Streamlit dashboard for analyzing legal matters data with conversion tracking, lead source analytics, geographic analysis, and performance metrics.

## Features

### 📊 Overview Dashboard
- Key Performance Indicators (KPIs)
- Matter status distribution
- Top matter types
- Recent matters table

### 🎯 Conversion Funnel Analysis
- Visual funnel chart showing lead progression
- Conversion rates by matter type
- Staff member conversion performance

### 📈 Lead Source Analytics
- Lead volume by source
- ROI analysis by source
- Source performance heatmap

### 📞 Engagement Metrics
- Activity rates (scheduled, calls, zoom meetings)
- No-show analysis with gauge chart
- Engagement correlation matrix

### 🗺️ Geographic Analysis
- Top states and cities by lead count
- Geographic performance comparison
- State-level conversion analysis

### 💰 Financial Dashboard
- Cumulative revenue over time
- Matter value distribution
- Revenue by matter type
- Financial performance metrics

### 👥 Staff Performance
- Staff leaderboards by revenue and conversion
- Performance scatter plot analysis
- Complete staff metrics table

### ⏰ Time-Based Analysis
- Lead patterns by day of week and hour
- Monthly trends comparison
- Time-based performance metrics

## Installation

1. Ensure you have Python 3.8+ installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the dashboard:**
   ```bash
   streamlit run clio_dashboard.py
   ```

2. **Access the dashboard:**
   - Open your web browser and go to `http://localhost:8501`

3. **Upload your data:**
   - Use the file uploader to upload your CLIO data (CSV or Excel format)
   - The dashboard will automatically load the default file if available

4. **Navigate and filter:**
   - Use the sidebar to navigate between dashboard pages
   - Apply filters for date range, status, matter type, and lead source
   - Export filtered data using the download button

## Data Requirements

Your CSV/Excel file should contain these columns:
- `Created` (timestamp)
- `Matter Type`
- `Status`
- `Created By`
- `Primary Contact`
- `Primary Contact Phone Number`
- `Scheduled` (Yes/No)
- `No show` (Yes/No)
- `Call` (Yes/No)
- `Zoom` (Yes/No)
- `Qualified Lead` (Yes/No)
- `Notes`
- `Primary Contact Type`
- `Primary Contact Source`
- `Matter Source 1`
- `Primary Contact City`
- `Primary Contact State`
- `Primary Contact Zip`
- `County`
- `Description`
- `Value` (monetary)

## Features

### Data Processing
- Automatic data validation and cleaning
- Missing data detection and warnings
- Date parsing and derived time columns
- Boolean conversion for Yes/No fields

### Interactive Filtering
- Date range selection
- Multi-select filters for status, matter type, and source
- Real-time data filtering across all visualizations

### Export Functionality
- Download filtered data as CSV
- Timestamped file names
- Preserves all applied filters

### Professional Styling
- Legal industry color scheme (navy, gray, gold)
- Mobile-responsive design
- Clean, professional interface
- Metric cards with visual indicators

### Data Quality Monitoring
- Automatic data quality checks
- Warning system for data issues
- Missing data percentage tracking
- Future date detection

## Technical Details

- **Framework:** Streamlit
- **Visualization:** Plotly Express/Graph Objects
- **Data Processing:** Pandas + NumPy
- **Caching:** Streamlit cache for performance
- **File Support:** CSV and Excel (.xlsx)

## Color Scheme

- **Primary:** Navy (#1f2937)
- **Secondary:** Gray (#6b7280)
- **Accent:** Gold (#d97706)
- **Success:** Green (#059669)
- **Warning:** Red (#dc2626)

## Performance Optimizations

- Data caching with TTL (5 minutes)
- Lazy loading of visualizations
- Efficient DataFrame operations
- Memory-conscious aggregations

## Troubleshooting

### Common Issues

1. **File Upload Errors:**
   - Ensure file is CSV or Excel format
   - Check column names match expected format
   - Verify data types are correct

2. **Performance Issues:**
   - Large files may take time to process
   - Try filtering date range to reduce data size
   - Clear cache if needed

3. **Missing Visualizations:**
   - Check if required columns exist in your data
   - Review data quality warnings
   - Ensure data contains records for selected filters

### Data Quality Checks

The dashboard automatically checks for:
- Missing values in key columns (>5% triggers warning)
- Future dates in the Created column
- Negative values in the Value column
- Inconsistent data formats

## License

This dashboard is created for analyzing CLIO legal matter data and is provided as-is for educational and business analysis purposes.

## Support

For issues or feature requests, please check:
1. Data format requirements
2. Column name matching
3. Filter settings
4. Browser compatibility

The dashboard works best with recent versions of Chrome, Firefox, Safari, and Edge.