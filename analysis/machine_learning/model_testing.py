#!/usr/bin/env python3
"""
Southwest Airlines Model Testing and Visualization Pipeline

This script tests the trained models and creates comprehensive visualizations:
- Model performance analysis
- Feature importance plots
- Prediction vs actual scatter plots
- Residual analysis
- Operational insights visualization
- Interactive plots for Southwest stakeholders
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SouthwestModelTester:
    """
    Comprehensive model testing and visualization for Southwest Airlines
    """
    
    def __init__(self, data_path=None):
        if data_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent.parent
            self.data_path = project_root / "data" / "preprocessed_data" / "southwest_final_preprocessed.csv"
        else:
            self.data_path = Path(data_path)
        
        self.df = None
        self.classification_model = None
        self.regression_model = None
        self.feature_encoders = {}
        self.scaler = StandardScaler()
        self.plots_dir = Path(__file__).parent / "model_testing_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and prepare data for testing"""
        print("📂 LOADING DATA FOR MODEL TESTING")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape}")
        
        # Create features
        self.df['HourOfDay'] = self.df['CRSDepTimeHour']
        self.df['DayOfWeek'] = self.df['DayOfWeek']
        self.df['Season'] = self.df['Season']
        self.df['IsDelayed'] = (self.df['DepDelayMinutes'] > 0).astype(int)
        
        # Delay cause binary features
        delay_causes = ['WeatherDelay', 'CarrierDelay', 'NASDelay', 'LateAircraftDelay']
        for cause in delay_causes:
            if cause in self.df.columns:
                self.df[f'{cause}_Binary'] = (self.df[cause] > 0).astype(int)
        
        # Ensure `Route` exists for downstream analysis/plots that use it
        if 'Route' not in self.df.columns and {'OriginCity','DestCity'}.issubset(self.df.columns):
            self.df['Route'] = self.df['OriginCity'] + ' → ' + self.df['DestCity']

        # Prepare features
        categorical_features = ['OriginCity', 'DestCity', 'Season']
        numerical_features = ['HourOfDay', 'DayOfWeek', 'Distance', 'TaxiOut', 'TaxiIn']
        numerical_features.extend([f'{cause}_Binary' for cause in delay_causes if f'{cause}_Binary' in self.df.columns])
        
        # Encode categorical features
        for feature in categorical_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[f'{feature}_Encoded'] = le.fit_transform(self.df[feature].astype(str))
                self.feature_encoders[feature] = le
                numerical_features.append(f'{feature}_Encoded')
        
        # Create feature matrix
        X = self.df[numerical_features].fillna(0)
        y_class = self.df['IsDelayed']
        y_reg = self.df['DepDelayMinutes']
        
        # Time-based splits
        train_mask = self.df['Year'].isin([2018, 2019])
        test_mask = self.df['Year'].isin([2022, 2023])
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_class_train = y_class[train_mask]
        y_class_test = y_class[test_mask]
        y_reg_train = y_reg[train_mask]
        y_reg_test = y_reg[test_mask]
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test, numerical_features
    
    def train_models(self, X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test):
        """Train the models"""
        print("\n🚀 TRAINING MODELS")
        print("=" * 60)
        
        # Classification model
        print("Training Classification Model (Random Forest)...")
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        self.classification_model.fit(X_train, y_class_train)
        
        # Regression model
        print("Training Regression Model (Random Forest)...")
        delayed_train_mask = y_class_train == 1
        X_train_delayed = X_train[delayed_train_mask]
        y_reg_train_delayed = y_reg_train[delayed_train_mask]
        
        self.regression_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        self.regression_model.fit(X_train_delayed, y_reg_train_delayed)
        
        print("✅ Models trained successfully")
        
        # Make predictions
        y_class_pred = self.classification_model.predict(X_test)
        y_class_proba = self.classification_model.predict_proba(X_test)[:, 1]
        
        delay_predictions = np.zeros(len(X_test))
        delayed_mask = y_class_pred == 1
        if delayed_mask.sum() > 0:
            delay_predictions[delayed_mask] = self.regression_model.predict(X_test[delayed_mask])
        
        return y_class_pred, y_class_proba, delay_predictions
    
    def create_model_performance_plots(self, y_class_test, y_class_pred, y_class_proba, y_reg_test, delay_predictions):
        """Create comprehensive model performance visualizations"""
        print("\n📊 CREATING MODEL PERFORMANCE PLOTS")
        print("=" * 60)
        
        # 1. Classification Performance
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Southwest Airlines Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # Confusion Matrix
        cm = confusion_matrix(y_class_test, y_class_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix\n(Classification Model)')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_class_test, y_class_proba)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_class_test, y_class_proba)
        pr_auc = auc(recall, precision)
        axes[0, 2].plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].legend(loc="lower left")
        axes[0, 2].grid(True, alpha=0.3)
        
        # Regression Performance - Scatter Plot
        axes[1, 0].scatter(y_reg_test, delay_predictions, alpha=0.5, s=1)
        axes[1, 0].plot([0, y_reg_test.max()], [0, y_reg_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Delay (minutes)')
        axes[1, 0].set_ylabel('Predicted Delay (minutes)')
        axes[1, 0].set_title('Predicted vs Actual Delays\n(Regression Model)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_reg_test, delay_predictions)
        mae = mean_absolute_error(y_reg_test, delay_predictions)
        axes[1, 0].text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.1f} min', 
                       transform=axes[1, 0].transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Residual Plot
        residuals = y_reg_test - delay_predictions
        axes[1, 1].scatter(delay_predictions, residuals, alpha=0.5, s=1)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Delay (minutes)')
        axes[1, 1].set_ylabel('Residuals (minutes)')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Error Distribution
        axes[1, 2].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Residuals (minutes)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Error Distribution')
        axes[1, 2].axvline(x=0, color='r', linestyle='--')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Model performance plots created")
    
    def create_feature_importance_plots(self, feature_names):
        """Create feature importance visualizations"""
        print("\n🔍 CREATING FEATURE IMPORTANCE PLOTS")
        print("=" * 60)
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Southwest Airlines Model Feature Importance', fontsize=16, fontweight='bold')
        
        # Classification Feature Importance
        if hasattr(self.classification_model, 'feature_importances_'):
            importances = self.classification_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot top 10 features
            top_features = min(10, len(feature_names))
            axes[0].bar(range(top_features), importances[indices[:top_features]])
            axes[0].set_title('Classification Model - Top 10 Features')
            axes[0].set_xlabel('Features')
            axes[0].set_ylabel('Importance')
            axes[0].set_xticks(range(top_features))
            axes[0].set_xticklabels([feature_names[i] for i in indices[:top_features]], rotation=45, ha='right')
            axes[0].grid(True, alpha=0.3)
            
            # Add values on bars
            for i, v in enumerate(importances[indices[:top_features]]):
                axes[0].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        # Regression Feature Importance
        if hasattr(self.regression_model, 'feature_importances_'):
            importances = self.regression_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot top 10 features
            top_features = min(10, len(feature_names))
            axes[1].bar(range(top_features), importances[indices[:top_features]], color='orange')
            axes[1].set_title('Regression Model - Top 10 Features')
            axes[1].set_xlabel('Features')
            axes[1].set_ylabel('Importance')
            axes[1].set_xticks(range(top_features))
            axes[1].set_xticklabels([feature_names[i] for i in indices[:top_features]], rotation=45, ha='right')
            axes[1].grid(True, alpha=0.3)
            
            # Add values on bars
            for i, v in enumerate(importances[indices[:top_features]]):
                axes[1].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Feature importance plots created")
    
    def create_operational_insights_plots(self, y_class_test, y_class_pred, delay_predictions):
        """Create operational insights visualizations"""
        print("\n🎯 CREATING OPERATIONAL INSIGHTS PLOTS")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Southwest Airlines Operational Insights', fontsize=16, fontweight='bold')
        
        # 1. Delay Prediction by Hour
        hourly_actual = self.df.groupby('CRSDepTimeHour')['DepDelayMinutes'].mean()
        hourly_pred = self.df.groupby('CRSDepTimeHour')['DepDelayMinutes'].mean()  # Simplified for demo
        
        axes[0, 0].plot(hourly_actual.index, hourly_actual.values, marker='o', linewidth=2, label='Actual')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Average Delay (minutes)')
        axes[0, 0].set_title('Average Delay by Hour of Day')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Delay Prediction by Day of Week
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_actual = self.df.groupby('DayOfWeek')['DepDelayMinutes'].mean()
        
        axes[0, 1].bar(range(7), daily_actual.values)
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Delay (minutes)')
        axes[0, 1].set_title('Average Delay by Day of Week')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(day_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Top Problematic Routes
        route_delays = self.df.groupby('Route')['DepDelayMinutes'].agg(['count', 'mean']).round(2)
        route_delays.columns = ['Flight_Count', 'Avg_Delay']
        worst_routes = route_delays[route_delays['Flight_Count'] >= 100].nlargest(10, 'Avg_Delay')
        
        axes[1, 0].barh(range(len(worst_routes)), worst_routes['Avg_Delay'])
        axes[1, 0].set_yticks(range(len(worst_routes)))
        axes[1, 0].set_yticklabels([route[:20] + '...' if len(route) > 20 else route for route in worst_routes.index])
        axes[1, 0].set_xlabel('Average Delay (minutes)')
        axes[1, 0].set_title('Top 10 Problematic Routes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Delay Cause Breakdown
        delay_causes = ['WeatherDelay', 'CarrierDelay', 'NASDelay', 'LateAircraftDelay']
        cause_data = []
        cause_labels = []
        
        for cause in delay_causes:
            if cause in self.df.columns:
                avg_delay = self.df[cause].mean()
                cause_data.append(avg_delay)
                cause_labels.append(cause.replace('Delay', ''))
        
        axes[1, 1].pie(cause_data, labels=cause_labels, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Delay Cause Breakdown\n(Average Delay Contribution)')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'operational_insights.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Operational insights plots created")
    
    def create_prediction_examples_plot(self, X_test, y_class_pred, y_class_proba, delay_predictions, y_reg_test):
        """Create prediction examples visualization"""
        print("\n👤 CREATING PREDICTION EXAMPLES PLOT")
        print("=" * 60)
        
        # Get some example predictions
        n_examples = 20
        example_indices = np.random.choice(len(X_test), n_examples, replace=False)
        
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle('Southwest Airlines Prediction Examples', fontsize=16, fontweight='bold')
        
        # Classification examples
        actual_delayed = y_reg_test.iloc[example_indices] > 0
        predicted_delayed = y_class_pred[example_indices] == 1
        probabilities = y_class_proba[example_indices]
        
        colors = ['green' if actual_delayed.iloc[i] == predicted_delayed[i] else 'red' for i in range(len(example_indices))]
        
        axes[0].bar(range(n_examples), probabilities, color=colors, alpha=0.7)
        axes[0].set_xlabel('Example Flights')
        axes[0].set_ylabel('Delay Probability')
        axes[0].set_title('Classification Predictions (Green=Correct, Red=Incorrect)')
        axes[0].set_xticks(range(n_examples))
        axes[0].set_xticklabels([f'Flight {i+1}' for i in range(n_examples)])
        axes[0].grid(True, alpha=0.3)
        
        # Add threshold line
        axes[0].axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        axes[0].legend()
        
        # Regression examples
        actual_delays = y_reg_test.iloc[example_indices]
        predicted_delays = delay_predictions[example_indices]
        
        x_pos = np.arange(n_examples)
        width = 0.35
        
        axes[1].bar(x_pos - width/2, actual_delays, width, label='Actual', alpha=0.7)
        axes[1].bar(x_pos + width/2, predicted_delays, width, label='Predicted', alpha=0.7)
        axes[1].set_xlabel('Example Flights')
        axes[1].set_ylabel('Delay (minutes)')
        axes[1].set_title('Regression Predictions')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f'Flight {i+1}' for i in range(n_examples)])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'prediction_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Prediction examples plot created")
    
    def run_complete_testing(self):
        """Run complete model testing and visualization pipeline"""
        print("🚀 SOUTHWEST MODEL TESTING PIPELINE")
        print("=" * 70)
        
        # Load and prepare data
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test, feature_names = self.load_and_prepare_data()
        
        # Train models
        y_class_pred, y_class_proba, delay_predictions = self.train_models(
            X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
        )
        
        # Create all visualizations
        self.create_model_performance_plots(y_class_test, y_class_pred, y_class_proba, y_reg_test, delay_predictions)
        self.create_feature_importance_plots(feature_names)
        self.create_operational_insights_plots(y_class_test, y_class_pred, delay_predictions)
        self.create_prediction_examples_plot(X_test, y_class_pred, y_class_proba, delay_predictions, y_reg_test)
        
        # Print final metrics
        print("\n📊 FINAL MODEL METRICS")
        print("=" * 60)
        
        classification_accuracy = (y_class_pred == y_class_test).mean()
        overall_mae = mean_absolute_error(y_reg_test, delay_predictions)
        overall_r2 = r2_score(y_reg_test, delay_predictions)
        
        print(f"Classification Accuracy: {classification_accuracy:.3f}")
        print(f"Overall MAE: {overall_mae:.2f} minutes")
        print(f"Overall R²: {overall_r2:.3f}")
        
        # Regression metrics on delayed flights only
        actually_delayed_mask = y_reg_test > 0
        if actually_delayed_mask.sum() > 0:
            actual_delays = y_reg_test[actually_delayed_mask]
            predicted_delays = delay_predictions[actually_delayed_mask]
            delayed_mae = mean_absolute_error(actual_delays, predicted_delays)
            delayed_r2 = r2_score(actual_delays, predicted_delays)
            
            print(f"Delayed Flights MAE: {delayed_mae:.2f} minutes")
            print(f"Delayed Flights R²: {delayed_r2:.3f}")
        
        print(f"\n🎯 TESTING COMPLETE!")
        print(f"✅ All plots saved to: {self.plots_dir}")
        print(f"✅ Models tested successfully")
        print(f"✅ Ready for Southwest operational use!")

def main():
    """Main function to run model testing"""
    tester = SouthwestModelTester()
    tester.run_complete_testing()

if __name__ == "__main__":
    main()
