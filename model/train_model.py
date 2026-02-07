"""
Employee Attrition Prediction Model Training Script
Train RandomForestClassifier on IBM HR Attrition dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class AttritionModelTrainer:
    """
    Handles the complete ML pipeline for Employee Attrition Prediction
    - Data loading and preprocessing
    - Feature engineering
    - Model training and evaluation
    - Model persistence
    """
    
    def __init__(self, data_path='data/hr_data.csv'):
        """Initialize the trainer with data path"""
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_and_preprocess_data(self):
        """Load CSV and preprocess data"""
        print("📊 Loading HR dataset...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        print(f"✓ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
        
        # Handle missing values
        df = df.dropna()
        print(f"✓ Missing values handled: {df.shape[0]} records remaining")
        
        # Identify categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove target from numerical columns if present
        if 'Attrition' in numerical_cols:
            numerical_cols.remove('Attrition')
        
        # Encode target variable
        print("🔄 Encoding features...")
        if 'Attrition' in df.columns:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Encode categorical features
        for col in categorical_cols:
            if col != 'Attrition':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Normalize numerical features
        self.scaler = StandardScaler()
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        print("✓ Feature encoding and normalization complete")
        
        return df
    
    def train_model(self, df):
        """Train RandomForestClassifier"""
        print("\n🤖 Training Random Forest model...")
        
        # Prepare features and target
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✓ Train set: {X_train.shape[0]} samples")
        print(f"✓ Test set: {X_test.shape[0]} samples")
        
        # Initialize and train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        print("✓ Model training complete")
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📈 Model Performance:")
        print(f"✓ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Stay', 'Leave']))
        
        print("\n🔍 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 10 Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return accuracy
    
    def save_model(self, model_dir='model'):
        """Save trained model and preprocessing objects"""
        print("\n💾 Saving model and preprocessing objects...")
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'attrition_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Scaler saved: {scaler_path}")
        
        # Save label encoders
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_path)
        print(f"✓ Label encoders saved: {encoders_path}")
        
        # Save feature columns
        features_path = os.path.join(model_dir, 'feature_columns.pkl')
        joblib.dump(self.feature_columns, features_path)
        print(f"✓ Feature columns saved: {features_path}")
        
        print("\n✅ Model training and saving complete!")
    
    def run(self):
        """Execute complete training pipeline"""
        try:
            # Load and preprocess data
            df = self.load_and_preprocess_data()
            
            # Train model
            accuracy = self.train_model(df)
            
            # Save model
            self.save_model()
            
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: Dataset not found at {self.data_path}")
            print("Please ensure hr_data.csv exists in the data/ directory")
            return False
            
        except Exception as e:
            print(f"❌ Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    print("="*60)
    print("🚀 Employee Attrition Prediction Model Trainer")
    print("="*60)
    
    # Initialize trainer
    trainer = AttritionModelTrainer()
    
    # Run training pipeline
    success = trainer.run()
    
    if success:
        print("\n" + "="*60)
        print("✅ Training completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Training failed. Please check errors above.")
        print("="*60)

if __name__ == "__main__":
    main()
