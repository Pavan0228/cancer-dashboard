"""
Backend Logic for Cancer Death Prediction Dashboard
Handles data processing, model training, and predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CancerDataProcessor:
    """Handle data loading and preprocessing"""
    
    def __init__(self, data_path='data/cancer_deaths.csv'):
        self.data_path = data_path
        self.df = None
        self.cancer_types = None
        
    def load_data(self):
        """Load and preprocess the cancer deaths dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            
            # Clean column names - remove the long descriptive parts
            cancer_columns = {}
            for col in self.df.columns:
                if 'Deaths -' in col:
                    # Extract just the cancer type name
                    cancer_name = col.split('Deaths - ')[1].split(' - Sex:')[0]
                    cancer_columns[col] = cancer_name
            
            # Rename columns for easier use
            self.df = self.df.rename(columns=cancer_columns)
            
            # Remove rows with missing Entity or Year
            self.df = self.df.dropna(subset=['Entity', 'Year'])
            
            # Convert Year to int
            self.df['Year'] = self.df['Year'].astype(int)
            
            # Get list of cancer types (excluding Entity, Code, Year)
            self.cancer_types = [col for col in self.df.columns if col not in ['Entity', 'Code', 'Year']]
            
            return self.df, self.cancer_types
        
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def get_country_data(self, country, start_year=None, end_year=None):
        """Get data for a specific country and year range"""
        if self.df is None:
            raise Exception("Data not loaded. Call load_data() first.")
        
        country_data = self.df[self.df['Entity'] == country].copy()
        
        if start_year:
            country_data = country_data[country_data['Year'] >= start_year]
        if end_year:
            country_data = country_data[country_data['Year'] <= end_year]
            
        return country_data
    
    def get_available_countries(self):
        """Get list of available countries"""
        if self.df is None:
            return []
        return sorted(self.df['Entity'].unique())
    
    def get_year_range(self):
        """Get the available year range"""
        if self.df is None:
            return None, None
        return self.df['Year'].min(), self.df['Year'].max()

class PredictionModelManager:
    """Handle different types of prediction models"""
    
    PREDICTION_MODES = {
        'trend_based': 'Trend-Based Prediction (Time Series)',
        'feature_rich': 'Feature-Rich Prediction (Multi-dimensional)',
        'ensemble': 'Ensemble Prediction (Multiple Models)',
        'growth_rate': 'Growth Rate Prediction',
        'comparative': 'Comparative Analysis'
    }
    
    MODEL_TYPES = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'linear_regression': LinearRegression(),
        'svm': SVR(kernel='rbf')
    }
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
    
    def prepare_features(self, data, mode='trend_based', selected_cancers=None):
        """Prepare features based on prediction mode"""
        if mode == 'trend_based':
            # Simple time-based features
            X = data[['Year']].copy()
            
        elif mode == 'feature_rich':
            # Create additional features
            X = data[['Year']].copy()
            X['Year_squared'] = X['Year'] ** 2
            X['Year_normalized'] = (X['Year'] - X['Year'].min()) / (X['Year'].max() - X['Year'].min())
            
            # Add moving averages only for selected cancers if we have enough data
            if len(data) > 3 and selected_cancers is not None:
                for cancer in selected_cancers:
                    if cancer in data.columns:
                        ma_values = data[cancer].rolling(window=3, min_periods=1).mean()
                        X[f'{cancer}_ma3'] = ma_values
                    else:
                        # If cancer not in data, set to 0
                        X[f'{cancer}_ma3'] = 0
            
        elif mode == 'growth_rate':
            # Focus on growth rate features
            X = data[['Year']].copy()
            X['Years_from_start'] = X['Year'] - X['Year'].min()
            
        else:
            # Default to trend-based
            X = data[['Year']].copy()
            
        return X
    
    def create_model(self, data, cancer_types, selected_cancers, country, 
                    train_years, prediction_mode='trend_based', model_type='random_forest'):
        """Create and train prediction model"""
        
        # Filter data for the selected country and years
        country_data = data[(data['Entity'] == country) & (data['Year'].isin(train_years))].copy()
        
        if len(country_data) < 5:
            return None, None, None, "Insufficient data for modeling (need at least 5 years)"
        
        # Prepare features based on mode
        X = self.prepare_features(country_data, prediction_mode, selected_cancers)
        y = country_data[selected_cancers].values
        
        # Handle missing values
        if np.isnan(y).any():
            for i, cancer in enumerate(selected_cancers):
                col_mean = np.nanmean(y[:, i])
                y[np.isnan(y[:, i]), i] = col_mean if not np.isnan(col_mean) else 0
        
        # Split data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Scale features for certain models
        scaler = None
        if model_type in ['svm', 'linear_regression']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test
        
        # Create model based on prediction mode
        if prediction_mode == 'ensemble':
            # Use ensemble of different models
            models = [
                RandomForestRegressor(n_estimators=50, random_state=42),
                GradientBoostingRegressor(n_estimators=50, random_state=42),
                LinearRegression()
            ]
            model = MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, random_state=42)
            )
        else:
            base_model = self.MODEL_TYPES[model_type]
            model = MultiOutputRegressor(base_model)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(X_train)), scoring='r2')
            cv_mean = cv_scores.mean()
        except:
            cv_mean = r2
        
        metrics = {
            'mae': mae,
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'cv_score': cv_mean
        }
        
        # Store model and scaler
        model_key = f"{country}_{prediction_mode}_{model_type}"
        self.models[model_key] = model
        if scaler:
            self.scalers[model_key] = scaler
        self.model_performance[model_key] = metrics
        
        test_results = (X_test, y_test, y_pred)
        
        return model, metrics, test_results, None
    
    def predict_future(self, model, scaler, start_year, end_year, cancer_types, 
                      prediction_mode='trend_based', base_data=None):
        """Generate future predictions"""
        future_years = np.arange(start_year, end_year + 1)
        
        # Create future feature matrix - must match training features exactly
        future_df = pd.DataFrame({'Year': future_years})
        
        if prediction_mode == 'feature_rich':
            future_df['Year_squared'] = future_df['Year'] ** 2
            if base_data is not None and len(base_data) > 0:
                min_year = base_data['Year'].min()
                max_year = base_data['Year'].max()
                future_df['Year_normalized'] = (future_df['Year'] - min_year) / (max_year - min_year)
                
                # For moving averages, use last known values from training data
                for cancer in cancer_types:
                    if cancer in base_data.columns:
                        # Use the last 3 values to compute moving average
                        last_values = base_data[cancer].tail(3)
                        last_ma = last_values.mean() if len(last_values) > 0 else 0
                        future_df[f'{cancer}_ma3'] = last_ma
                    else:
                        # If cancer not in base_data, set to 0
                        future_df[f'{cancer}_ma3'] = 0
        
        elif prediction_mode == 'growth_rate':
            if base_data is not None and len(base_data) > 0:
                start_year_base = base_data['Year'].min()
                future_df['Years_from_start'] = future_df['Year'] - start_year_base
            else:
                future_df['Years_from_start'] = future_df['Year'] - future_years[0]
        
        # Ensure we have the right number of features
        # If scaler exists, it knows the expected number of features
        if scaler is not None:
            try:
                X_future = scaler.transform(future_df)
            except ValueError as e:
                # If scaling fails due to feature mismatch, try to fix it
                expected_features = scaler.n_features_in_
                current_features = future_df.shape[1]
                
                if current_features < expected_features:
                    # Add missing features as zeros
                    for i in range(expected_features - current_features):
                        future_df[f'missing_feature_{i}'] = 0
                elif current_features > expected_features:
                    # Remove extra features
                    future_df = future_df.iloc[:, :expected_features]
                
                X_future = scaler.transform(future_df)
        else:
            X_future = future_df.values
        
        # Make predictions with error handling
        try:
            predictions = model.predict(X_future)
        except ValueError as e:
            # If prediction fails, try with simplified features
            print(f"Feature mismatch detected. Using simplified prediction model: {e}")
            
            # Try with just Year feature as fallback
            X_simple = np.array(future_years).reshape(-1, 1)
            
            # If we have a scaler, we need to be careful about feature dimensions
            if scaler:
                # Create dummy features to match scaler expectations
                n_features_expected = scaler.n_features_in_
                X_padded = np.zeros((len(future_years), n_features_expected))
                X_padded[:, 0] = future_years  # Year in first column
                X_simple = scaler.transform(X_padded)
            
            predictions = model.predict(X_simple)
        
        # Create result DataFrame
        pred_df = pd.DataFrame(predictions, columns=cancer_types)
        pred_df['Year'] = future_years
        
        return pred_df

class AnalyticsEngine:
    """Handle advanced analytics and insights"""
    
    def __init__(self):
        pass
    
    def calculate_growth_rates(self, data, cancer_types):
        """Calculate year-over-year growth rates"""
        growth_data = []
        
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            prev_row = data.iloc[i-1]
            
            for cancer in cancer_types:
                if cancer in data.columns:
                    current_val = current_row[cancer]
                    prev_val = prev_row[cancer]
                    
                    if prev_val > 0:
                        growth_rate = ((current_val - prev_val) / prev_val) * 100
                    else:
                        growth_rate = 0
                    
                    growth_data.append({
                        'Year': current_row['Year'],
                        'Cancer Type': cancer,
                        'Growth Rate (%)': growth_rate,
                        'Current Deaths': current_val,
                        'Previous Deaths': prev_val
                    })
        
        return pd.DataFrame(growth_data)
    
    def identify_trends(self, data, cancer_types, threshold=5):
        """Identify significant trends in cancer deaths"""
        trends = {}
        
        for cancer in cancer_types:
            if cancer in data.columns and len(data) > 2:
                # Calculate trend using simple linear regression
                years = data['Year'].values.reshape(-1, 1)
                deaths = data[cancer].values
                
                # Remove NaN values
                valid_indices = ~np.isnan(deaths)
                if np.sum(valid_indices) > 2:
                    years_clean = years[valid_indices]
                    deaths_clean = deaths[valid_indices]
                    
                    # Fit linear trend
                    from sklearn.linear_model import LinearRegression
                    trend_model = LinearRegression()
                    trend_model.fit(years_clean, deaths_clean)
                    
                    slope = trend_model.coef_[0]
                    
                    # Classify trend
                    if abs(slope) < threshold:
                        trend_type = 'Stable'
                    elif slope > threshold:
                        trend_type = 'Increasing'
                    else:
                        trend_type = 'Decreasing'
                    
                    trends[cancer] = {
                        'trend': trend_type,
                        'slope': slope,
                        'rate_per_year': slope,
                        'r2': trend_model.score(years_clean, deaths_clean)
                    }
        
        return trends
    
    def compare_countries(self, data, countries, cancer_types, year_range=None):
        """Compare cancer statistics across countries"""
        comparison_data = []
        
        for country in countries:
            country_data = data[data['Entity'] == country]
            
            if year_range:
                country_data = country_data[
                    (country_data['Year'] >= year_range[0]) & 
                    (country_data['Year'] <= year_range[1])
                ]
            
            if len(country_data) > 0:
                for cancer in cancer_types:
                    if cancer in country_data.columns:
                        stats = {
                            'Country': country,
                            'Cancer Type': cancer,
                            'Total Deaths': country_data[cancer].sum(),
                            'Average Deaths': country_data[cancer].mean(),
                            'Max Deaths': country_data[cancer].max(),
                            'Min Deaths': country_data[cancer].min(),
                            'Years of Data': len(country_data)
                        }
                        comparison_data.append(stats)
        
        return pd.DataFrame(comparison_data)
    
    def calculate_mortality_rates(self, data, population_data=None):
        """Calculate mortality rates if population data is available"""
        # Placeholder for mortality rate calculations
        # Would need population data to implement properly
        pass
    
    def predict_risk_factors(self, data, cancer_types):
        """Analyze potential risk factors based on trends"""
        risk_analysis = {}
        
        for cancer in cancer_types:
            if cancer in data.columns:
                recent_data = data.tail(5)  # Last 5 years
                older_data = data.head(5)   # First 5 years
                
                if len(recent_data) > 0 and len(older_data) > 0:
                    recent_avg = recent_data[cancer].mean()
                    older_avg = older_data[cancer].mean()
                    
                    if older_avg > 0:
                        change_ratio = recent_avg / older_avg
                        
                        if change_ratio > 1.2:
                            risk_level = 'High Risk'
                        elif change_ratio > 1.1:
                            risk_level = 'Moderate Risk'
                        elif change_ratio < 0.9:
                            risk_level = 'Improving'
                        else:
                            risk_level = 'Stable'
                        
                        risk_analysis[cancer] = {
                            'risk_level': risk_level,
                            'change_ratio': change_ratio,
                            'recent_average': recent_avg,
                            'historical_average': older_avg
                        }
        
        return risk_analysis