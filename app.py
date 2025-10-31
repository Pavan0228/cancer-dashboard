"""
Cancer Death Prediction Dashboard - Main Application
A Streamlit app for analyzing and predicting global cancer death trends using advanced ML models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from backend import CancerDataProcessor, PredictionModelManager, AnalyticsEngine
from frontend import DashboardUI, VisualizationManager, MetricsDisplay, DataExporter
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page (must be first Streamlit command)
st.set_page_config(
    page_title="🏥 Cancer Death Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize backend components
@st.cache_resource
def initialize_backend():
    """Initialize backend components with caching"""
    data_processor = CancerDataProcessor()
    model_manager = PredictionModelManager()
    analytics_engine = AnalyticsEngine()
    return data_processor, model_manager, analytics_engine

@st.cache_data
def load_and_process_data():
    """Load and cache data"""
    data_processor, _, _ = initialize_backend()
    return data_processor.load_data()

def main():
    # Initialize components
    data_processor, model_manager, analytics_engine = initialize_backend()
    dashboard_ui = DashboardUI()
    viz_manager = VisualizationManager()
    metrics_display = MetricsDisplay()
    data_exporter = DataExporter()
    
    # Render header
    dashboard_ui.render_header()
    
    # Load data
    try:
        df, cancer_types = load_and_process_data()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.info("Please ensure the cancer_deaths.csv file is in the data/ directory.")
        return
    
    if df is None or cancer_types is None:
        st.error("Failed to load data. Please check if the data file exists.")
        return
    
    # Get data info
    countries = data_processor.get_available_countries()
    year_range = data_processor.get_year_range()
    
    # Render sidebar controls
    controls = dashboard_ui.render_sidebar_controls(
        countries=countries,
        cancer_types=cancer_types,
        year_range=year_range,
        prediction_modes=model_manager.PREDICTION_MODES,
        model_types=model_manager.MODEL_TYPES
    )
    
    # Extract control values
    selected_country = controls['selected_country']
    selected_cancers = controls['selected_cancers']
    train_start = controls['train_start']
    train_end = controls['train_end']
    pred_start = controls['pred_start']
    pred_end = controls['pred_end']
    prediction_mode = controls['prediction_mode']
    model_type = controls['model_type']
    
    if not selected_cancers:
        st.warning("Please select at least one cancer type.")
        return
    
    # Display prediction mode info
    dashboard_ui.render_prediction_mode_info(prediction_mode, model_manager.PREDICTION_MODES)
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training & Performance", 
        "🔮 Predictions", 
        "🌍 Country Comparison",
        "📈 Advanced Analytics"
    ])
    
    with tab1:
        st.header("📊 Dataset Overview")
        
        # Display dataset metrics
        metrics_display.display_dataset_overview(
            total_countries=len(countries),
            year_range=year_range,
            cancer_types=len(cancer_types),
            total_records=len(df)
        )
        
        # Get country data
        country_data = data_processor.get_country_data(selected_country)
        
        st.subheader(f"📈 {selected_country} - Historical Trends")
        
        if not country_data.empty:
            # Create and display historical trends chart
            fig = viz_manager.create_historical_trends_chart(
                country_data, selected_cancers, selected_country
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.subheader("📋 Raw Data")
            display_data = country_data[['Year'] + selected_cancers].sort_values('Year', ascending=False)
            st.dataframe(display_data, use_container_width=True)
            
            # Download button for raw data
            csv_data = data_exporter.export_predictions_csv(display_data)
            st.download_button(
                label="📥 Download Raw Data",
                data=csv_data,
                file_name=f"{selected_country}_cancer_data.csv",
                mime="text/csv"
            )
        
        else:
            st.warning(f"No data available for {selected_country}")
    
    with tab2:
        st.header("🤖 Model Training & Performance")
        
        # Model training section
        train_years = list(range(train_start, train_end + 1))
        
        # Show training configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Training Years:** {train_start}-{train_end}")
        with col2:
            st.info(f"**Prediction Mode:** {model_manager.PREDICTION_MODES[prediction_mode]}")
        with col3:
            st.info(f"**Model Type:** {model_type.replace('_', ' ').title()}")
        
        with st.spinner(f"Training {prediction_mode} model with {model_type}..."):
            model, metrics, test_results, error = model_manager.create_model(
                df, cancer_types, selected_cancers, selected_country, 
                train_years, prediction_mode, model_type
            )
        
        if error:
            st.error(error)
            st.info("💡 **Suggestions:**")
            st.write("- Try selecting a different country with more data")
            st.write("- Increase the training year range")
            st.write("- Select fewer cancer types")
            return
        
        if model is None:
            st.error("Failed to create model")
            return
        
        # Display comprehensive metrics
        st.subheader("📊 Model Performance Metrics")
        metrics_display.display_model_metrics(metrics)
        
        # Performance interpretation
        if metrics['r2'] > 0.8:
            st.success("🎯 Excellent model performance! High prediction accuracy.")
        elif metrics['r2'] > 0.6:
            st.warning("⚠️ Good model performance. Consider more training data for better accuracy.")
        else:
            st.error("❌ Poor model performance. Try different parameters or more data.")
        
        # Feature importance analysis
        st.subheader("🎯 Model Feature Analysis")
        
        # Get feature importance for each cancer type
        importance_data = []
        
        # Handle different display modes for feature importance
        if prediction_mode == 'feature_rich':
            # Show feature importance per feature across cancer types
            for i, cancer in enumerate(selected_cancers):
                if hasattr(model.estimators_[i], 'feature_importances_'):
                    importance_values = model.estimators_[i].feature_importances_
                    feature_names = ['Year', 'Year²', 'Year_norm']
                    
                    # Add moving average features if they exist
                    if len(importance_values) > len(feature_names):
                        remaining_features = len(importance_values) - len(feature_names)
                        for j in range(remaining_features):
                            feature_names.append(f'MA_Feature_{j+1}')
                    
                    for j, importance in enumerate(importance_values):
                        if j < len(feature_names):
                            importance_data.append({
                                'Cancer Type': cancer,
                                'Feature': feature_names[j],
                                'Importance': importance
                            })
        else:
            # Simple mode - show importance per cancer type
            for i, cancer in enumerate(selected_cancers):
                if hasattr(model.estimators_[i], 'feature_importances_'):
                    # Take the average importance across all features for this cancer type
                    avg_importance = np.mean(model.estimators_[i].feature_importances_)
                    importance_data.append({
                        'Cancer Type': cancer,
                        'Feature Importance': avg_importance
                    })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            fig_imp = viz_manager.create_feature_importance_chart(importance_df)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
        
        # Model performance visualization
        if test_results:
            X_test, y_test, y_pred = test_results
            
            st.subheader("� Model Performance Visualization")
            
            # Create performance charts
            fig_perf = viz_manager.create_model_performance_chart(
                X_test, y_test, y_pred, selected_cancers
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
            # Export model performance
            performance_csv = data_exporter.export_model_performance(metrics, test_results)
            st.download_button(
                label="📥 Download Model Performance",
                data=performance_csv,
                file_name=f"{selected_country}_model_performance.csv",
                mime="text/csv"
            )
    
    with tab3:
        st.header("🔮 Future Predictions")
        
        if 'model' not in locals() or model is None:
            st.warning("Please train the model first in the 'Model Training & Performance' tab.")
            return
        
        # Get model scaler if available
        model_key = f"{selected_country}_{prediction_mode}_{model_type}"
        scaler = model_manager.scalers.get(model_key, None)
        
        # Get base data for feature-rich predictions
        base_data = data_processor.get_country_data(selected_country, train_start, train_end)
        
        # Generate predictions
        with st.spinner("Generating predictions..."):
            pred_df = model_manager.predict_future(
                model, scaler, int(pred_start), int(pred_end), 
                selected_cancers, prediction_mode, base_data
            )
        
        # Get historical data for comparison
        historical_data = data_processor.get_country_data(selected_country, train_start, train_end)
        
        # Create prediction visualization
        st.subheader(f"📈 Predicted Cancer Deaths in {selected_country} ({pred_start}-{pred_end})")
        
        fig_pred = viz_manager.create_prediction_comparison_chart(
            historical_data, pred_df, selected_cancers, selected_country, train_end
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Key prediction insights
        metrics_display.display_prediction_insights(
            historical_data, pred_df, selected_cancers, train_end
        )
        
        # Prediction scenarios
        st.subheader("🎭 Prediction Scenarios")
        
        scenario = st.selectbox(
            "Select Analysis Scenario",
            ["Standard Prediction", "Best Case (-10%)", "Worst Case (+15%)", "Conservative (+5%)"]
        )
        
        if scenario != "Standard Prediction":
            scenario_df = pred_df.copy()
            if scenario == "Best Case (-10%)":
                for cancer in selected_cancers:
                    scenario_df[cancer] *= 0.9
                st.success("📉 Best case: 10% reduction in predicted deaths")
            elif scenario == "Worst Case (+15%)":
                for cancer in selected_cancers:
                    scenario_df[cancer] *= 1.15
                st.error("📈 Worst case: 15% increase in predicted deaths")
            elif scenario == "Conservative (+5%)":
                for cancer in selected_cancers:
                    scenario_df[cancer] *= 1.05
                st.warning("📊 Conservative: 5% increase in predicted deaths")
            
            # Display scenario predictions
            st.dataframe(scenario_df.round(0).astype(int), use_container_width=True)
        else:
            # Standard prediction table
            st.subheader("📋 Detailed Predictions")
            pred_display = pred_df.round(0).astype(int)
            st.dataframe(pred_display, use_container_width=True)
        
        # Growth rate analysis
        st.subheader("📈 Growth Rate Analysis")
        
        growth_analysis = analytics_engine.calculate_growth_rates(pred_df, selected_cancers)
        
        if not growth_analysis.empty:
            fig_growth = viz_manager.create_growth_rate_chart(growth_analysis)
            st.plotly_chart(fig_growth, use_container_width=True)
            
            # Highlight concerning trends
            high_growth = growth_analysis[growth_analysis['Growth Rate (%)'] > 10]
            if not high_growth.empty:
                st.warning("⚠️ **High Growth Rate Alert:**")
                for _, row in high_growth.iterrows():
                    st.write(f"- **{row['Cancer Type']}** in {row['Year']}: {row['Growth Rate (%)']:.1f}% growth")
        
        # Export predictions
        csv_predictions = data_exporter.export_predictions_csv(pred_df)
        st.download_button(
            label="📥 Download Predictions",
            data=csv_predictions,
            file_name=f"{selected_country}_predictions_{pred_start}-{pred_end}.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.header("🌍 Country Comparison")
        
        # Multi-country selector
        comparison_countries = st.multiselect(
            "Select countries for comparison",
            countries,
            default=[selected_country] + [c for c in ["China", "India", "United States", "Brazil"] if c in countries and c != selected_country][:3]
        )
        
        if len(comparison_countries) < 2:
            st.warning("Please select at least 2 countries for comparison.")
            return
        
        # Comparison type selector
        comparison_type = st.selectbox(
            "Comparison Type",
            ["Predictions", "Historical Trends", "Growth Rates", "Model Performance"]
        )
        
        if comparison_type == "Predictions":
            st.subheader("📊 Cross-Country Prediction Comparison")
            
            comparison_data = []
            model_performances = {}
            
            progress_bar = st.progress(0)
            
            for i, country in enumerate(comparison_countries):
                progress_bar.progress((i + 1) / len(comparison_countries))
                
                # Train model for each country
                country_model, country_metrics, _, country_error = model_manager.create_model(
                    df, cancer_types, selected_cancers, country, 
                    train_years, prediction_mode, model_type
                )
                
                if country_model is not None and not country_error:
                    model_performances[country] = country_metrics
                    
                    # Get scaler and base data for this country
                    country_key = f"{country}_{prediction_mode}_{model_type}"
                    country_scaler = model_manager.scalers.get(country_key, None)
                    country_base_data = data_processor.get_country_data(country, train_start, train_end)
                    
                    # Generate predictions
                    country_pred = model_manager.predict_future(
                        country_model, country_scaler, int(pred_start), int(pred_end),
                        selected_cancers, prediction_mode, country_base_data
                    )
                    
                    for _, row in country_pred.iterrows():
                        for cancer in selected_cancers:
                            comparison_data.append({
                                'Country': country,
                                'Year': row['Year'],
                                'Cancer Type': cancer,
                                'Predicted Deaths': row[cancer]
                            })
            
            progress_bar.empty()
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create comparison charts for each cancer type
                for cancer in selected_cancers:
                    fig_comp = viz_manager.create_country_comparison_chart(
                        comparison_df, cancer
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                # Summary heatmap
                st.subheader("🌡️ Prediction Heatmap (Average Deaths)")
                
                avg_comparison = comparison_df.groupby(['Country', 'Cancer Type'])['Predicted Deaths'].mean().reset_index()
                heatmap_pivot = avg_comparison.pivot(index='Country', columns='Cancer Type', values='Predicted Deaths')
                
                fig_heatmap = viz_manager.create_heatmap(
                    heatmap_pivot,
                    title="Average Predicted Deaths by Country and Cancer Type"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Model performance comparison
                if model_performances:
                    st.subheader("🎯 Model Performance Comparison")
                    perf_data = []
                    for country, metrics in model_performances.items():
                        perf_data.append({
                            'Country': country,
                            'R² Score': metrics['r2'],
                            'MAE': metrics['mae'],
                            'RMSE': metrics['rmse'],
                            'CV Score': metrics['cv_score']
                        })
                    
                    perf_df = pd.DataFrame(perf_data)
                    st.dataframe(perf_df, use_container_width=True)
        
        elif comparison_type == "Historical Trends":
            st.subheader("📈 Historical Trends Comparison")
            
            # Compare historical data across countries
            comparison_stats = analytics_engine.compare_countries(
                df, comparison_countries, selected_cancers, (train_start, train_end)
            )
            
            if not comparison_stats.empty:
                # Display comparison statistics
                st.dataframe(comparison_stats, use_container_width=True)
                
                # Create visualizations for average deaths by country
                for cancer in selected_cancers[:3]:
                    cancer_stats = comparison_stats[comparison_stats['Cancer Type'] == cancer]
                    
                    fig = px.bar(
                        cancer_stats,
                        x='Country',
                        y='Average Deaths',
                        title=f"Average {cancer} Deaths by Country ({train_start}-{train_end})",
                        color='Average Deaths',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif comparison_type == "Growth Rates":
            st.subheader("📊 Growth Rate Comparison")
            
            growth_comparison = []
            
            for country in comparison_countries:
                country_data = data_processor.get_country_data(country, train_start, train_end)
                if len(country_data) > 1:
                    trends = analytics_engine.identify_trends(country_data, selected_cancers)
                    
                    for cancer, trend_info in trends.items():
                        growth_comparison.append({
                            'Country': country,
                            'Cancer Type': cancer,
                            'Trend': trend_info['trend'],
                            'Rate per Year': trend_info['rate_per_year'],
                            'R² Score': trend_info['r2']
                        })
            
            if growth_comparison:
                growth_df = pd.DataFrame(growth_comparison)
                st.dataframe(growth_df, use_container_width=True)
                
                # Visualize trends
                fig = px.bar(
                    growth_df,
                    x='Country',
                    y='Rate per Year',
                    color='Cancer Type',
                    title="Annual Growth Rate by Country and Cancer Type",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("📈 Advanced Analytics")
        
        # Risk analysis
        st.subheader("⚠️ Risk Factor Analysis")
        
        country_data = data_processor.get_country_data(selected_country)
        if len(country_data) > 10:
            risk_analysis = analytics_engine.predict_risk_factors(country_data, selected_cancers)
            
            if risk_analysis:
                risk_data = []
                for cancer, risk_info in risk_analysis.items():
                    risk_data.append({
                        'Cancer Type': cancer,
                        'Risk Level': risk_info['risk_level'],
                        'Change Ratio': risk_info['change_ratio'],
                        'Recent Average': risk_info['recent_average'],
                        'Historical Average': risk_info['historical_average']
                    })
                
                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True)
                
                # Color-code by risk level
                for _, row in risk_df.iterrows():
                    if row['Risk Level'] == 'High Risk':
                        st.error(f"🚨 **{row['Cancer Type']}**: High risk trend detected")
                    elif row['Risk Level'] == 'Moderate Risk':
                        st.warning(f"⚠️ **{row['Cancer Type']}**: Moderate risk trend")
                    elif row['Risk Level'] == 'Improving':
                        st.success(f"✅ **{row['Cancer Type']}**: Improving trend")
        
        # Trend analysis
        st.subheader("📊 Trend Analysis")
        
        trends = analytics_engine.identify_trends(country_data, selected_cancers)
        
        if trends:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Increasing Trends:**")
                increasing = [cancer for cancer, info in trends.items() if info['trend'] == 'Increasing']
                for cancer in increasing:
                    st.write(f"📈 {cancer}: +{trends[cancer]['rate_per_year']:.1f} deaths/year")
            
            with col2:
                st.write("**Decreasing Trends:**")
                decreasing = [cancer for cancer, info in trends.items() if info['trend'] == 'Decreasing']
                for cancer in decreasing:
                    st.write(f"📉 {cancer}: {trends[cancer]['rate_per_year']:.1f} deaths/year")
        
        # Export all analytics
        st.subheader("📥 Export Analytics")
        
        if st.button("Generate Comprehensive Report"):
            # Create comprehensive report
            report_data = {
                'Country': selected_country,
                'Analysis_Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'Prediction_Mode': prediction_mode,
                'Model_Type': model_type,
                'Training_Years': f"{train_start}-{train_end}",
                'Prediction_Years': f"{pred_start}-{pred_end}",
                'Selected_Cancers': ', '.join(selected_cancers)
            }
            
            report_df = pd.DataFrame([report_data])
            report_csv = report_df.to_csv(index=False)
            
            st.download_button(
                label="📄 Download Comprehensive Report",
                data=report_csv,
                file_name=f"cancer_analysis_report_{selected_country}.csv",
                mime="text/csv"
            )
            
            st.success("Report generated successfully!")

if __name__ == "__main__":
    main()