import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="DATA ANALYZER",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #ff33dd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stSelectbox label {
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📊 DATA ANALYZER & VISUALIZER</h1>', unsafe_allow_html=True)

# Sidebar for file upload and settings
st.sidebar.header("📁 Data Upload Here")

def load_data():
    """Load data from uploaded file"""
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
        help="Upload CSV, Excel, JSON, or Parquet files"
    )
    
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.parquet'):
                df = pd.read_parquet(uploaded_file)
            
            st.sidebar.success(f"✅ Selected file loaded successfully !")
            st.sidebar.success(f"✅ Selected File fetched successfully !")
            st.sidebar.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            return None
    return None

def generate_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    sample_data = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.normal(1000, 200, 100).round(2),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'Product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 100),
        'Customers': np.random.randint(50, 200, 100),
        'Revenue': np.random.normal(5000, 1000, 100).round(2),
        'Satisfaction': np.random.uniform(3.0, 5.0, 100).round(1)
    })
    return sample_data

def data_overview(df):
    """Display data overview and basic statistics"""
    st.header("📋 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", f"{df.shape[1]:,}")
    with col3:
        st.metric("Numeric Columns", f"{df.select_dtypes(include=[np.number]).shape[1]:,}")
    with col4:
        st.metric("Text Columns", f"{df.select_dtypes(include=['object']).shape[1]:,}")
    
    # Data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data types and info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.subheader("📈 Basic Statistics")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for statistical analysis")

def create_visualizations(df):
    """Create various visualizations"""
    st.header("📊 Data Visualizations")
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", 
         "Heatmap", "Pie Chart", "Distribution Plot", "Time Series"]
    )
    
    if viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X-axis", numeric_cols)
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols)
        with col3:
            color_by = st.selectbox("Color by", ["None"] + categorical_cols)
        
        fig = px.scatter(
            df, x=x_axis, y=y_axis,
            color=color_by if color_by != "None" else None,
            title=f"{x_axis} vs {y_axis}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Line Chart":
        if date_cols:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", date_cols + numeric_cols)
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols)
            
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No date columns found for time series visualization")
    
    elif viz_type == "Bar Chart":
        if categorical_cols and numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Category", categorical_cols)
            with col2:
                y_axis = st.selectbox("Value", numeric_cols)
            
            agg_func = st.selectbox("Aggregation", ["sum", "mean", "count", "median"])
            
            if agg_func == "count":
                grouped_df = df.groupby(x_axis).size().reset_index(name='count')
                fig = px.bar(grouped_df, x=x_axis, y='count', title=f"Count by {x_axis}")
            else:
                grouped_df = df.groupby(x_axis)[y_axis].agg(agg_func).reset_index()
                fig = px.bar(grouped_df, x=x_axis, y=y_axis, title=f"{agg_func.title()} of {y_axis} by {x_axis}")
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Histogram":
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                column = st.selectbox("Column", numeric_cols)
            with col2:
                bins = st.slider("Number of bins", 10, 50, 30)
            
            fig = px.histogram(df, x=column, nbins=bins, title=f"Distribution of {column}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                y_axis = st.selectbox("Numeric Column", numeric_cols)
            with col2:
                x_axis = st.selectbox("Group by", ["None"] + categorical_cols)
            
            if x_axis == "None":
                fig = px.box(df, y=y_axis, title=f"Box Plot of {y_axis}")
            else:
                fig = px.box(df, x=x_axis, y=y_axis, title=f"Box Plot of {y_axis} by {x_axis}")
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Heatmap":
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Pie Chart":
        if categorical_cols:
            column = st.selectbox("Column", categorical_cols)
            value_counts = df[column].value_counts().head(10)
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution of {column}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Distribution Plot":
        if numeric_cols:
            column = st.selectbox("Column", numeric_cols)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[column], name="Histogram", opacity=0.7))
            
            # Add KDE curve
            from scipy import stats
            kde = stats.gaussian_kde(df[column].dropna())
            x_range = np.linspace(df[column].min(), df[column].max(), 100)
            y_kde = kde(x_range)
            
            # Scale KDE to match histogram
            hist_counts, _ = np.histogram(df[column].dropna(), bins=30)
            scale_factor = max(hist_counts) / max(y_kde)
            
            fig.add_trace(go.Scatter(x=x_range, y=y_kde * scale_factor, mode='lines', name="KDE"))
            fig.update_layout(title=f"Distribution of {column}")
            st.plotly_chart(fig, use_container_width=True)

def data_filtering(df):
    """Add data filtering capabilities"""
    st.header("🔍 Data Filtering")
    
    # Column selection for filtering
    filter_col = st.selectbox("Select column to filter", df.columns)
    
    if df[filter_col].dtype in ['object']:
        # Categorical filtering
        unique_values = df[filter_col].unique()
        selected_values = st.multiselect(f"Select {filter_col} values", unique_values, default=unique_values)
        filtered_df = df[df[filter_col].isin(selected_values)]
    
    elif df[filter_col].dtype in ['int64', 'float64']:
        # Numeric filtering
        min_val = float(df[filter_col].min())
        max_val = float(df[filter_col].max())
        selected_range = st.slider(f"Select {filter_col} range", min_val, max_val, (min_val, max_val))
        filtered_df = df[(df[filter_col] >= selected_range[0]) & (df[filter_col] <= selected_range[1])]
    
    else:
        filtered_df = df
    
    st.info(f"Filtered data: {filtered_df.shape[0]} rows (from {df.shape[0]} original rows)")
    
    return filtered_df

def export_data(df):
    """Export filtered data"""
    st.header("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV file",
                data=csv,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📥 Download Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Data', index=False)
            excel_data = output.getvalue()
            st.download_button(
                label="Download Excel file",
                data=excel_data,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def main():
    """Main application function"""
    
    # Data loading
    df = load_data()
    
    # Use sample data if no file uploaded
    if df is None:
        if st.sidebar.button("🎯 Use Sample Data"):
            df = generate_sample_data()
            st.sidebar.success("Sample data loaded!")
        else:
            st.info("👈 Please upload a data file to get started !")
            return
    
    # Main dashboard
    if df is not None:
        # Data overview
        data_overview(df)
        
        st.markdown("---")
        
        # Data filtering
        filtered_df = data_filtering(df)
        
        st.markdown("---")
        
        # Visualizations
        create_visualizations(filtered_df)
        
        st.markdown("---")
        
        # Export options
        export_data(filtered_df)
        
        # Footer
        st.markdown("---")
       
       
       
        st.markdown("*                           📊 © 2024 - 2026 ALL RIGHTS RESERVED  BY RAJKUMAR SHAHU      ")
        st.markdown("---")

if __name__ == "__main__":
    main()