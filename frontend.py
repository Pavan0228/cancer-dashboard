"""
Frontend UI Components for Cancer Death Prediction Dashboard
Handles all Streamlit UI elements and visualizations
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class DashboardUI:
    """Main dashboard UI manager"""
    
    def __init__(self):
        self.apply_custom_css()
    
    def apply_custom_css(self):
        """Apply custom CSS styling"""
        st.markdown("""
        <style>
            .main-header {
                font-size: 3rem;
                font-weight: bold;
                color: #1f77b4;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-container {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
            }
            .stSelectbox > div > div > select {
                background-color: #ffffff;
            }
            .prediction-mode-info {
                background-color: black;
                padding: 1rem;
                border-radius: 10px;
                border-left: 5px solid #1f77b4;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main dashboard header"""
        st.markdown('<h1 class="main-header">🏥 Cancer Death Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    def render_sidebar_controls(self, countries, cancer_types, year_range, prediction_modes, model_types):
        """Render sidebar control elements"""
        st.sidebar.header("🎯 Dashboard Controls")
        
        # Country selection
        selected_country = st.sidebar.selectbox(
            "📍 Select Country",
            countries,
            index=countries.index("United States") if "United States" in countries else 0
        )
        
        # Cancer type selection
        st.sidebar.subheader("🎗️ Cancer Types")
        selected_cancers = st.sidebar.multiselect(
            "Select cancer types for analysis",
            cancer_types,
            default=cancer_types[:5] if len(cancer_types) >= 5 else cancer_types
        )
        
        # Year range selection
        min_year, max_year = year_range
        
        st.sidebar.subheader("📅 Year Range")
        train_start = st.sidebar.slider("Training Start Year", min_year, max_year-5, min_year)
        train_end = st.sidebar.slider("Training End Year", train_start+5, max_year, max_year)
        
        pred_start = st.sidebar.number_input("Prediction Start Year", value=max_year+1, min_value=max_year+1)
        pred_end = st.sidebar.number_input("Prediction End Year", value=max_year+10, min_value=int(pred_start))
        
        # Model configuration
        st.sidebar.subheader("🤖 Model Configuration")
        
        prediction_mode = st.sidebar.selectbox(
            "Prediction Mode",
            options=list(prediction_modes.keys()),
            format_func=lambda x: prediction_modes[x]
        )
        
        model_type = st.sidebar.selectbox(
            "Model Type",
            options=list(model_types.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        return {
            'selected_country': selected_country,
            'selected_cancers': selected_cancers,
            'train_start': train_start,
            'train_end': train_end,
            'pred_start': pred_start,
            'pred_end': pred_end,
            'prediction_mode': prediction_mode,
            'model_type': model_type
        }
    
    def render_prediction_mode_info(self, mode, modes_dict):
        """Render information about the selected prediction mode"""
        mode_descriptions = {
            'trend_based': "Uses historical time trends to predict future values. Simple but effective for stable trends.",
            'feature_rich': "Incorporates additional features like moving averages and polynomial terms for more complex patterns.",
            'ensemble': "Combines multiple models for more robust predictions.",
            'growth_rate': "Focuses on growth rate patterns and acceleration in cancer deaths.",
            'comparative': "Designed for comparing predictions across different regions or demographics."
        }
        
        st.markdown(f"""
        <div class="prediction-mode-info">
            <h4>🔍 {modes_dict[mode]}</h4>
            <p>{mode_descriptions.get(mode, 'Advanced prediction mode with specialized algorithms.')}</p>
        </div>
        """, unsafe_allow_html=True)

class VisualizationManager:
    """Handle all chart and visualization creation"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set1
    
    def create_historical_trends_chart(self, data, selected_cancers, country):
        """Create historical trends line chart"""
        fig = go.Figure()
        
        for i, cancer in enumerate(selected_cancers):
            color = self.color_palette[i % len(self.color_palette)]
            fig.add_trace(go.Scatter(
                x=data['Year'],
                y=data[cancer],
                mode='lines+markers',
                name=cancer,
                line=dict(width=2, color=color),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"Historical Cancer Deaths in {country}",
            xaxis_title="Year",
            yaxis_title="Number of Deaths",
            hovermode='x unified',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_prediction_comparison_chart(self, historical_data, prediction_data, selected_cancers, country, split_year):
        """Create chart comparing historical and predicted data"""
        fig = go.Figure()
        
        for i, cancer in enumerate(selected_cancers):
            color = self.color_palette[i % len(self.color_palette)]
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical_data['Year'],
                y=historical_data[cancer],
                mode='lines+markers',
                name=f'{cancer} (Historical)',
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
            
            # Predicted data
            fig.add_trace(go.Scatter(
                x=prediction_data['Year'],
                y=prediction_data[cancer],
                mode='lines+markers',
                name=f'{cancer} (Predicted)',
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(size=6, symbol='diamond')
            ))
        
        # Add vertical line to separate historical from predicted
        fig.add_vline(
            x=split_year + 0.5,
            line_dash="dot",
            line_color="red",
            annotation_text="Prediction Start"
        )
        
        fig.update_layout(
            title=f"Cancer Death Predictions: {country}",
            xaxis_title="Year",
            yaxis_title="Number of Deaths",
            hovermode='x unified',
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_model_performance_chart(self, X_test, y_test, y_pred, selected_cancers):
        """Create model performance visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Predicted', 'Residuals Plot', 'Error Distribution', 'Performance by Cancer Type'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Actual vs Predicted scatter plot
        for i, cancer in enumerate(selected_cancers[:3]):  # Show up to 3 cancer types
            color = self.color_palette[i % len(self.color_palette)]
            fig.add_trace(
                go.Scatter(
                    x=y_test[:, i] if len(y_test.shape) > 1 else y_test,
                    y=y_pred[:, i] if len(y_pred.shape) > 1 else y_pred,
                    mode='markers',
                    name=f'{cancer}',
                    marker=dict(color=color, size=8, opacity=0.7)
                ),
                row=1, col=1
            )
        
        # Perfect prediction line
        if len(y_test) > 0:
            min_val = min(y_test.min() if hasattr(y_test, 'min') else min(y_test),
                         y_pred.min() if hasattr(y_pred, 'min') else min(y_pred))
            max_val = max(y_test.max() if hasattr(y_test, 'max') else max(y_test),
                         y_pred.max() if hasattr(y_pred, 'max') else max(y_pred))
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='gray')
                ),
                row=1, col=1
            )
        
        # Residuals plot
        for i, cancer in enumerate(selected_cancers[:3]):
            residuals = (y_test[:, i] - y_pred[:, i]) if len(y_test.shape) > 1 else (y_test - y_pred)
            fig.add_trace(
                go.Scatter(
                    x=y_pred[:, i] if len(y_pred.shape) > 1 else y_pred,
                    y=residuals,
                    mode='markers',
                    name=f'Residuals - {cancer}',
                    marker=dict(color=self.color_palette[i], size=6, opacity=0.7),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Values", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        
        return fig
    
    def create_feature_importance_chart(self, importance_data):
        """Create feature importance bar chart"""
        # Check what columns are available and adapt accordingly
        if 'Feature Importance' in importance_data.columns:
            y_col = 'Feature Importance'
            color_col = 'Feature Importance'
        elif 'Importance' in importance_data.columns:
            y_col = 'Importance'
            color_col = 'Importance'
        else:
            # Fallback - use the first numeric column
            numeric_cols = importance_data.select_dtypes(include=[np.number]).columns
            y_col = numeric_cols[0] if len(numeric_cols) > 0 else importance_data.columns[-1]
            color_col = y_col
        
        # Determine x-axis column
        if 'Cancer Type' in importance_data.columns:
            x_col = 'Cancer Type'
        elif 'Feature' in importance_data.columns:
            x_col = 'Feature'
        else:
            x_col = importance_data.columns[0]
        
        fig = px.bar(
            importance_data,
            x=x_col,
            y=y_col,
            title="Model Feature Importance",
            color=color_col,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        return fig
    
    def create_growth_rate_chart(self, growth_data):
        """Create growth rate analysis chart"""
        fig = px.bar(
            growth_data,
            x='Year',
            y='Growth Rate (%)',
            color='Cancer Type',
            title="Predicted Year-over-Year Growth Rates",
            barmode='group'
        )
        fig.update_layout(height=400)
        return fig
    
    def create_country_comparison_chart(self, comparison_data, cancer_type):
        """Create country comparison line chart"""
        cancer_data = comparison_data[comparison_data['Cancer Type'] == cancer_type]
        
        fig = px.line(
            cancer_data,
            x='Year',
            y='Predicted Deaths',
            color='Country',
            title=f"Predicted {cancer_type} Deaths by Country",
            markers=True
        )
        fig.update_layout(height=400)
        return fig
    
    def create_heatmap(self, pivot_data, title="Heatmap"):
        """Create heatmap visualization"""
        fig = px.imshow(
            pivot_data.values,
            labels=dict(x="Cancer Type", y="Country", color="Avg Predicted Deaths"),
            x=pivot_data.columns,
            y=pivot_data.index,
            color_continuous_scale='Reds',
            title=title
        )
        fig.update_layout(height=500)
        return fig

class MetricsDisplay:
    """Handle metrics and KPI displays"""
    
    def __init__(self):
        pass
    
    def display_model_metrics(self, metrics):
        """Display model performance metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Absolute Error", f"{metrics['mae']:.2f}")
        with col2:
            st.metric("R² Score", f"{metrics['r2']:.4f}")
        with col3:
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
        with col4:
            st.metric("CV Score", f"{metrics['cv_score']:.4f}")
    
    def display_dataset_overview(self, total_countries, year_range, cancer_types, total_records):
        """Display dataset overview metrics"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Countries", total_countries)
        with col2:
            st.metric("Year Range", f"{year_range[0]}-{year_range[1]}")
        with col3:
            st.metric("Cancer Types", cancer_types)
        with col4:
            st.metric("Total Records", total_records)
    
    def display_prediction_insights(self, historical_data, prediction_data, selected_cancers, train_end):
        """Display key prediction insights"""
        st.subheader("🎯 Key Prediction Insights")
        
        cols = st.columns(min(3, len(selected_cancers)))
        
        for i, cancer in enumerate(selected_cancers[:3]):
            with cols[i]:
                if len(historical_data) > 0 and cancer in historical_data.columns:
                    last_historical = historical_data[cancer].iloc[-1]
                    first_prediction = prediction_data[cancer].iloc[0]
                    growth_rate = ((first_prediction - last_historical) / last_historical * 100) if last_historical > 0 else 0
                    
                    st.metric(
                        f"📊 {cancer[:15]}...",
                        f"{int(first_prediction):,}",
                        f"{growth_rate:+.1f}% vs {train_end}"
                    )

class DataExporter:
    """Handle data export functionality"""
    
    def __init__(self):
        pass
    
    def export_predictions_csv(self, prediction_data):
        """Export predictions to CSV format"""
        csv = prediction_data.to_csv(index=False)
        return csv
    
    def export_model_performance(self, metrics, test_results):
        """Export model performance data"""
        performance_data = {
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        }
        performance_df = pd.DataFrame(performance_data)
        return performance_df.to_csv(index=False)