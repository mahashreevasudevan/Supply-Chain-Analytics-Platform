import pandas as pd
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
import warnings
import logging
from datetime import datetime, timedelta
import traceback
import streamlit as st
import re
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedIntelligencePlatform:
    """
    Supply Chain and Customer Analytics Platform   
    """
    
    def __init__(self):
        """Initialize the Platform"""
        load_dotenv()
        
        # Connection and data attributes
        self.snowflake_conn = None
        self.connection_error = None
        self.auto_connect_attempted = False
        
        # Data storage
        self.supply_df = None
        self.access_df = None
        self.data_loaded = False
        self.rfm_data = None
        
        # Knowledge graph
        self.knowledge_graph = nx.Graph()
        
        # ML Models storage
        self.ml_models = {}
        self.anomaly_detector = None
        self.churn_predictor = None
        self.forecast_model = None
        self.forecast_data = None
        self.forecast_metrics = None  # NEW: Store forecast accuracy metrics
        self.anomaly_results = None   # NEW: Store anomaly detection results
        self.churn_results = None     # NEW: Store churn prediction results
        
        # Initializing components
        self.setup_mlops()
        self.setup_ai_components()
        
        # Initializing advanced query router
        self.create_advanced_query_router()
        
        logger.info("Platform initialized. Auto-connecting to Snowflake...")
        self.auto_connect_snowflake()
    
    def __del__(self):
        """Cleanup method to close connections"""
        if self.snowflake_conn:
            try:
                self.snowflake_conn.close()
                logger.info("Snowflake connection closed")
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

    # Initialization and Setup
    
    
    def setup_mlops(self):
        """Initializing MLflow for experiment tracking"""
        try:
            if MLFLOW_AVAILABLE:
                mlflow.set_experiment("supply_chain_intelligence")
                self.mlops_enabled = True
                logger.info("MLflow experiment tracking initialized")
            else:
                self.mlops_enabled = False
                logger.warning("MLflow not available")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")
            self.mlops_enabled = False
    
    def setup_ai_components(self):
        """AI/NLP components setup with query parsing"""
        try:
            if NLP_AVAILABLE:
                # Sentiment analysis pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="distilbert-base-uncased-finetuned-sst-2-english"
                )
                
                # Text generation pipeline
                self.text_generator = pipeline(
                    "text-generation", 
                    model="distilgpt2", 
                    max_length=100, 
                    num_return_sequences=1,
                    pad_token_id=50256
                )
                
                # Named Entity Recognition for extracting IDs and entities
                self.ner_pipeline = pipeline(
                    "ner", 
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                
                # Question Answering for contextual queries
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad"
                )
                
                # Zero-shot classification for intent detection
                self.intent_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
                
                self.ai_enabled = True
                logger.info("AI/NLP components initialized successfully")
            else:
                self.ai_enabled = False
                logger.warning("Transformers library not available")
        except Exception as e:
            logger.warning(f"AI components setup failed: {e}")
            self.ai_enabled = False
    
    # Snowflake Connection Methods
    
    
    def validate_environment(self):
        """Environment variables checking"""
        required_vars = {
            'SNOWFLAKE_USER': os.getenv('SNOWFLAKE_USER'),
            'SNOWFLAKE_PASSWORD': os.getenv('SNOWFLAKE_PASSWORD'), 
            'SNOWFLAKE_ACCOUNT': os.getenv('SNOWFLAKE_ACCOUNT'),
            'SNOWFLAKE_WAREHOUSE': os.getenv('SNOWFLAKE_WAREHOUSE')
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            error_msg = f"Missing environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            return False, error_msg
        
        return True, "All environment variables present"
    
    def auto_connect_snowflake(self):
        """Connecting to Snowflake on platform initialization"""
        if self.auto_connect_attempted:
            return self.snowflake_conn is not None
        
        self.auto_connect_attempted = True
        
        try:
            # Validating environment first
            env_valid, env_msg = self.validate_environment()
            if not env_valid:
                self.connection_error = env_msg
                logger.warning(f"Auto-connect skipped: {env_msg}")
                return False
            
            # Importing snowflake connector
            try:
                import snowflake.connector
            except ImportError:
                self.connection_error = "Snowflake connector not installed. Run: pip install snowflake-connector-python"
                logger.error(self.connection_error)
                return False
            
            
            self.snowflake_conn = snowflake.connector.connect(
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
                database=os.getenv('SNOWFLAKE_DATABASE', 'SUPPLY'),
                schema=os.getenv('SNOWFLAKE_SCHEMA', 'ANALYTICS'),
                timeout=30
            )
            
            # Testing connection
            cursor = self.snowflake_conn.cursor()
            try:
                cursor.execute("SELECT CURRENT_VERSION()")
                version = cursor.fetchone()
                logger.info(f"Auto-connected to Snowflake successfully. Version: {version[0] if version else 'Unknown'}")
                self.connection_error = None
                
                # Auto-loading data
                self.auto_load_default_data()
                return True
            finally:
                cursor.close()
            
        except Exception as e:
            error_msg = f"Snowflake auto-connection failed: {str(e)}"
            self.connection_error = error_msg
            logger.warning(error_msg)
            return False

    def connect_snowflake(self):
        """Manual connection method (if required)"""
        try:
            # Validating environment variables first
            env_valid, env_msg = self.validate_environment()
            if not env_valid:
                self.connection_error = env_msg
                return False
            
            # Importing Snowflake connector
            try:
                import snowflake.connector
            except ImportError:
                self.connection_error = "Snowflake connector not installed. Run: pip install snowflake-connector-python"
                logger.error(self.connection_error)
                return False
            
            # Attempting connection
            self.snowflake_conn = snowflake.connector.connect(
                user=os.getenv('SNOWFLAKE_USER'),
                password=os.getenv('SNOWFLAKE_PASSWORD'),
                account=os.getenv('SNOWFLAKE_ACCOUNT'),
                warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
                database=os.getenv('SNOWFLAKE_DATABASE', 'SUPPLY'),
                schema=os.getenv('SNOWFLAKE_SCHEMA', 'ANALYTICS'),
                timeout=30
            )
            
            # Testing connection
            cursor = self.snowflake_conn.cursor()
            try:
                cursor.execute("SELECT CURRENT_VERSION()")
                version = cursor.fetchone()
                logger.info(f"Connected to Snowflake successfully. Version: {version[0] if version else 'Unknown'}")
                self.connection_error = None
                return True
            finally:
                cursor.close()
            
        except Exception as e:
            error_msg = f"Snowflake connection failed: {str(e)}"
            self.connection_error = error_msg
            logger.error(error_msg)
            return False

    def test_connection_context(self):
        """Testing connection and verifying database/schema context"""
        if not self.snowflake_conn:
            logger.error("No Snowflake connection")
            return False
        
        cursor = None
        try:
            cursor = self.snowflake_conn.cursor()
            
            # Checking current context
            cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE()")
            context = cursor.fetchone()
            logger.info(f"Current context - Database: {context[0]}, Schema: {context[1]}, Warehouse: {context[2]}")
            
            # Setting the correct context if needed
            cursor.execute("USE DATABASE SUPPLY")
            cursor.execute("USE SCHEMA ANALYTICS")
            
            # Verifying context change
            cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
            new_context = cursor.fetchone()
            logger.info(f"After USE statements - Database: {new_context[0]}, Schema: {new_context[1]}")
            
            # List tables in current schema
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            table_list = [table[1] for table in tables]
            logger.info(f"Tables in SUPPLY.ANALYTICS: {table_list}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error testing connection context: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    # ML model training
   
    def calculate_forecast_metrics(self, actual_data, forecast_data):
        """Calculating forecast accuracy metrics"""
        try:
            if len(actual_data) == 0 or len(forecast_data) == 0:
                return None
            
            # Aligning dates
            forecast_df = forecast_data.copy()
            actual_df = actual_data.copy()
            
            # Converting to datetime if required
            if actual_df['ORDER_DATE'].dtype == 'object':
                actual_df['ORDER_DATE'] = pd.to_datetime(actual_df['ORDER_DATE'], errors='coerce')
            
            # Grouping by date
            actual_daily = actual_df.groupby('ORDER_DATE')['ORDER_ITEM_QUANTITY'].sum().reset_index()
            actual_daily.columns = ['ds', 'actual']
            
            # Merging with forecast
            merged = pd.merge(forecast_df[['ds', 'yhat']], actual_daily, on='ds', how='inner')
            
            if len(merged) < 2:
                return None
            
            # Calculating metrics
            mae = mean_absolute_error(merged['actual'], merged['yhat'])
            rmse = np.sqrt(mean_squared_error(merged['actual'], merged['yhat']))
            mape = np.mean(np.abs((merged['actual'] - merged['yhat']) / merged['actual'])) * 100
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'data_points': len(merged)
            }
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {e}")
            return None

    def train_prophet_forecast_model(self, supply_df):
        """Prophet forecasting model"""
        if not PROPHET_AVAILABLE:
            logger.error("Prophet not available")
            return False
        
        try:
            # Checking required columns
            if 'ORDER_DATE' not in supply_df.columns or 'ORDER_ITEM_QUANTITY' not in supply_df.columns:
                logger.error("Prophet requires ORDER_DATE and ORDER_ITEM_QUANTITY columns")
                return False
            
            # Preparing data for Prophet
            df_prophet = supply_df[['ORDER_DATE', 'ORDER_ITEM_QUANTITY']].copy()
            
            # Converting date column
            if df_prophet['ORDER_DATE'].dtype == 'object':
                df_prophet['ORDER_DATE'] = pd.to_datetime(df_prophet['ORDER_DATE'], errors='coerce')
            
            # Removing invalid dates
            df_prophet = df_prophet.dropna()
            
            if df_prophet.empty:
                logger.error("No valid data for Prophet after cleaning")
                return False
            
            # Aggregating by date
            daily_data = df_prophet.groupby('ORDER_DATE')['ORDER_ITEM_QUANTITY'].sum().reset_index()
            daily_data.columns = ['ds', 'y']
            
            if len(daily_data) < 10:
                logger.error("Insufficient data points for Prophet (need at least 10)")
                return False
            
            # Splitting data for validation 
            split_date = daily_data['ds'].quantile(0.8)
            train_data = daily_data[daily_data['ds'] <= split_date].copy()
            test_data = daily_data[daily_data['ds'] > split_date].copy()
            
            # Training Prophet model
            model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
            model.fit(train_data)
            
            # Make Forecasts
            future = model.make_future_dataframe(periods=30)
            forecast = model.predict(future)
            
            # Calculating metrics on test data 
            metrics = None
            if len(test_data) > 1:
                test_forecast = model.predict(test_data[['ds']])
                test_forecast['actual'] = test_data['y'].values
                
                mae = mean_absolute_error(test_forecast['actual'], test_forecast['yhat'])
                rmse = np.sqrt(mean_squared_error(test_forecast['actual'], test_forecast['yhat']))
                mape = np.mean(np.abs((test_forecast['actual'] - test_forecast['yhat']) / test_forecast['actual'])) * 100
                
                metrics = {
                    'MAE': mae,
                    'RMSE': rmse,
                    'MAPE': mape,
                    'test_points': len(test_data)
                }
            
           
            self.forecast_model = model
            self.forecast_data = forecast
            self.forecast_metrics = metrics
            
            logger.info("Prophet model trained successfully ")
            return True
            
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return False

    def train_churn_prediction_model(self, supply_df):
        """Churn/risk prediction model"""
        try:
            # Checking required columns
            required_cols = ['CUSTOMER_SEGMENT', 'DELIVERY_STATUS', 'CUSTOMER_CITY', 
                            'SHIPPING_MODE', 'CUSTOMER_STATE', 'LATE_DELIVERY_RISK']
            
            available_cols = [col for col in required_cols if col in supply_df.columns]
            
            if len(available_cols) < 3:
                logger.error(f"Insufficient columns for churn prediction. Need at least 3 from: {required_cols}")
                return False
            
            # Preparing features
            df_ml = supply_df[available_cols].copy()
            
            # Removing rows with missing target
            if 'LATE_DELIVERY_RISK' not in df_ml.columns:
                logger.error("LATE_DELIVERY_RISK column required as target")
                return False
            
            df_ml = df_ml.dropna(subset=['LATE_DELIVERY_RISK'])
            
            if df_ml.empty or len(df_ml) < 50:
                logger.error("Insufficient data for training (need at least 50 records)")
                return False
            
            # Preparing features and target
            feature_cols = [col for col in available_cols if col != 'LATE_DELIVERY_RISK']
            X = df_ml[feature_cols].copy()
            y = df_ml['LATE_DELIVERY_RISK'].copy()
            
            # Encoding categorical variables
            label_encoders = {}
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
            
            # Handling remaining missing values
            X = X.fillna(0)
            
            # Splitting data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scaling features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Training Random Forest model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Generating predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculating metrics
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Generating classification report and confusion matrix
            class_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            
            self.churn_predictor = {
                'model': model,
                'scaler': scaler,
                'label_encoders': label_encoders,
                'features': feature_cols,
                'train_score': train_score,
                'test_score': test_score
            }
            
            self.churn_results = {
                'classification_report': class_report,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred,
                'actual': y_test,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            logger.info(f"Churn prediction model trained - Test accuracy: {test_score:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training churn prediction model: {e}")
            return False

    def train_anomaly_detection_model(self, access_df, supply_df):
        """Anomaly detection model"""
        try:
        
            if access_df is None or access_df.empty:
                logger.info("No access log data, using supply chain data for anomaly detection")
                df_anomaly = supply_df.copy()
            else:
                df_anomaly = access_df.copy()
            
            if df_anomaly is None or df_anomaly.empty:
                logger.error("No data available for anomaly detection")
                return False
        
            # Getting available columns
            available_cols = df_anomaly.columns.tolist()
            logger.info(f"Available columns for anomaly detection: {available_cols}")
        
            # Selecting relevant columns for anomaly detection
            feature_cols = []
        
            # Adding numeric columns
            numeric_cols = df_anomaly.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols.extend(numeric_cols[:5])  # Take first 5 numeric columns
        
            # Adding categorical columns
            categorical_cols = df_anomaly.select_dtypes(include=['object']).columns.tolist()
            feature_cols.extend(categorical_cols[:3])  # Take first 3 categorical columns
        
            if len(feature_cols) < 2:
                logger.error("Insufficient features for anomaly detection")
                return False
        
            # Preparing the feature matrix
            df_features = df_anomaly[feature_cols].copy()
        
            # Handling missing values
            df_features = df_features.dropna()
        
            if df_features.empty or len(df_features) < 100:
                logger.error("Insufficient data for anomaly detection (need at least 100 records)")
                return False
        
            # Encoding categorical variables
            label_encoders = {}
            for col in df_features.columns:
                if df_features[col].dtype == 'object':
                    le = LabelEncoder()
                    df_features[col] = le.fit_transform(df_features[col].astype(str))
                    label_encoders[col] = le
        
            # Scaling the features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_features)
        
            # Training Isolation Forest
            model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
            anomaly_predictions = model.fit_predict(X_scaled)
        
            # Calculating anomaly rate
            anomaly_rate = (anomaly_predictions == -1).mean()
            
            # Creating anomaly results dataframe
            anomaly_df = df_features.copy()
            anomaly_df['anomaly_score'] = model.decision_function(X_scaled)
            anomaly_df['is_anomaly'] = anomaly_predictions == -1
            anomaly_df = anomaly_df.sort_values('anomaly_score')
        
            
            self.anomaly_detector = {
                'model': model,
                'scaler': scaler,
                'label_encoders': label_encoders,
                'features': feature_cols,
                'anomaly_rate': anomaly_rate
            }
            
            self.anomaly_results = {
                'anomaly_data': anomaly_df,
                'anomaly_count': (anomaly_predictions == -1).sum(),
                'total_records': len(df_features)
            }
        
            logger.info(f"Anomaly detection model trained successfully - Anomaly rate: {anomaly_rate:.3f}")
            logger.info(f"Features used: {feature_cols}")
            return True
        
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {e}")
            return False
    
    # Data Management
    
    
    def list_available_tables(self):
        """List available tables in the Snowflake database"""
        if not self.snowflake_conn:
            return []
        
        cursor = None
        try:
            cursor = self.snowflake_conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            table_names = [table[1] for table in tables]
            logger.info(f"Available tables: {table_names}")
            return table_names
            
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def inspect_table_schema(self, table_name):
        """Checking the schema of a specific table"""
        if not self.snowflake_conn:
            return None
        
        cursor = None
        try:
            cursor = self.snowflake_conn.cursor()
            cursor.execute(f"DESCRIBE TABLE {table_name}")
            schema = cursor.fetchall()
            
            schema_df = pd.DataFrame(schema, columns=[
                'name', 'type', 'kind', 'null?', 'default', 
                'primary key', 'unique key', 'check', 'expression', 'comment'
            ])
            return schema_df[['name', 'type', 'null?']]
            
        except Exception as e:
            logger.error(f"Error inspecting table {table_name}: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def fetch_data_from_table(self, table_name, limit=None):
        """Fetching data from a specific table"""
        if not self.snowflake_conn:
            logger.error("No Snowflake connection")
            return None
        
        cursor = None
        try:
            # First check if table exists and has data
            cursor = self.snowflake_conn.cursor()
            
            # Check if table exists
            cursor.execute(f"DESC TABLE {table_name}")
            schema_info = cursor.fetchall()
            logger.info(f"Table {table_name} exists with {len(schema_info)} columns")
            
            # Check row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            logger.info(f"Table {table_name} has {row_count} total rows")
            
            if row_count == 0:
                logger.warning(f"Table {table_name} is empty")
                return pd.DataFrame()
            
            cursor.close()
            cursor = None
            
            
            if limit:
                query = f"SELECT * FROM {table_name} LIMIT {limit}"
            else:
                query = f"SELECT * FROM {table_name}"
            
            logger.info(f"Executing query: {query}")
            
            df = pd.read_sql(query, self.snowflake_conn)
            
            logger.info(f"Pandas read_sql returned dataframe with shape: {df.shape}")
            logger.info(f"Column names: {list(df.columns)}")
            
            if df.empty:
                logger.warning(f"DataFrame is empty despite {row_count} rows in table")
                
                
                cursor = self.snowflake_conn.cursor()
                try:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    if rows:
                        logger.info(f"Cursor method returned {len(rows)} rows, converting to DataFrame")
                        df = pd.DataFrame(rows, columns=columns)
                        logger.info(f"Alternative method created DataFrame with shape: {df.shape}")
                    else:
                        logger.error("Both pandas and cursor methods returned no data")
                finally:
                    cursor.close()
                    cursor = None
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {table_name}: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    def auto_load_default_data(self):
        """Automatically load default tables if they exist"""
        if not self.snowflake_conn:
            return
        
        try:
            
            self.test_connection_context()
            
            default_tables = ['SUPPLY_CHAIN_DATA', 'SUPPLY_CHAIN', 'ORDERS', 'CUSTOMER_ORDERS', 'SALES_DATA']
            access_tables = ['ACCESS_LOGS_DATA', 'ACCESS_LOGS', 'USER_ACCESS', 'WEB_LOGS']
            
            tables = self.list_available_tables()
            
            
            supply_table = None
            for table in default_tables:
                if table in tables:
                    supply_table = table
                    break
            
            
            access_table = None
            for table in access_tables:
                if table in tables:
                    access_table = table
                    break
            
            if supply_table:
                logger.info(f"Auto-loading from supply table: {supply_table}")
                if access_table:
                    logger.info(f"Also loading access table: {access_table}")
                else:
                    logger.info("No access logs table found , proceeding with supply chain data only")
                
                success = self.load_data_interactive(
                    supply_table=supply_table,
                    access_table=access_table
                )
                if success:
                    logger.info("Auto-load completed successfully")
                else:
                    logger.warning("Auto-load failed, but connection is available")
            else:
                logger.info(f"No default tables found for auto-load. Available tables: {tables}")
                
        except Exception as e:
            logger.warning(f"Auto-load failed: {e}")
    
    def load_data_interactive(self, supply_table=None, access_table=None):
        """Load data from tables without limits"""
        supply_loaded = False
        access_loaded = False
        
        try:
            if supply_table:
                logger.info(f"Attempting to load supply table: {supply_table}")
                self.supply_df = self.fetch_data_from_table(supply_table)
                if self.supply_df is not None and not self.supply_df.empty:
                    supply_loaded = True
                    logger.info(f"Successfully loaded {len(self.supply_df)} records from {supply_table}")
                else:
                    logger.warning(f"Failed to load data from {supply_table}")
            
            if access_table:
                logger.info(f"Attempting to load access table: {access_table}")
                self.access_df = self.fetch_data_from_table(access_table)
                if self.access_df is not None and not self.access_df.empty:
                    access_loaded = True
                    logger.info(f"Successfully loaded {len(self.access_df)} records from {access_table}")
                else:
                    logger.warning(f"Failed to load data from {access_table}")
            
            
            if supply_loaded:
                self.data_loaded = True
                logger.info(f"Data loading successful - Supply: {supply_loaded}, Access: {access_loaded}")
                
                # Building knowledge graph 
                try:
                    self.build_knowledge_graph(self.supply_df, self.access_df if access_loaded else None)
                    logger.info("Knowledge graph built successfully")
                except Exception as e:
                    logger.warning(f"Knowledge graph building failed: {e}")
                
                return True
            else:
                self.data_loaded = False
                logger.error("Failed to load supply chain data. This is required for the platform")
                return False
                
        except Exception as e:
            logger.error(f"Error during interactive data loading: {e}")
            self.data_loaded = False
            return False
    
    def get_data_status(self):
        """Get current data loading status"""
        status = {
            'loaded': self.data_loaded,
            'supply_records': len(self.supply_df) if self.supply_df is not None else 0,
            'access_records': len(self.access_df) if self.access_df is not None else 0,
            'connection_error': self.connection_error
        }
        
        if not self.data_loaded and self.supply_df is None:
            status['error'] = "Data not loaded"
        
        return status

    
    # Knowledge Graph
    
    
    def build_knowledge_graph(self, supply_df, access_df=None):
        """Building comprehensive knowledge graph with customers, products, and categories"""
        try:
            
            self.knowledge_graph.clear()
            
            if supply_df is not None and not supply_df.empty:
                logger.info("Building knowledge graph with enhanced node types...")
                
                # Adding customer nodes
                if 'CUSTOMER_ID' in supply_df.columns:
                    for customer_id in supply_df['CUSTOMER_ID'].dropna().unique():
                        self.knowledge_graph.add_node(f"CUSTOMER_{customer_id}", type="customer")
                
                # Adding product nodes  
                if 'PRODUCT_CARD_ID' in supply_df.columns:
                    for product_id in supply_df['PRODUCT_CARD_ID'].dropna().unique():
                        self.knowledge_graph.add_node(f"PRODUCT_{product_id}", type="product")
                
                # Adding category nodes
                if 'CATEGORY_NAME' in supply_df.columns:
                    for category in supply_df['CATEGORY_NAME'].dropna().unique():
                        self.knowledge_graph.add_node(f"CATEGORY_{category}", type="category")
                
                # Adding department nodes
                if 'DEPARTMENT_NAME' in supply_df.columns:
                    for department in supply_df['DEPARTMENT_NAME'].dropna().unique():
                        self.knowledge_graph.add_node(f"DEPARTMENT_{department}", type="department")
                
                # Adding city nodes
                if 'CUSTOMER_CITY' in supply_df.columns:
                    for city in supply_df['CUSTOMER_CITY'].dropna().unique():
                        self.knowledge_graph.add_node(f"CITY_{city}", type="city")
                
                # Adding customer segment nodes
                if 'CUSTOMER_SEGMENT' in supply_df.columns:
                    for segment in supply_df['CUSTOMER_SEGMENT'].dropna().unique():
                        self.knowledge_graph.add_node(f"SEGMENT_{segment}", type="segment")
                
                # Adding edges between different entity types
                for _, row in supply_df.iterrows():
                    try:
                        # Customer-Product relationships
                        if pd.notna(row.get('CUSTOMER_ID')) and pd.notna(row.get('PRODUCT_CARD_ID')):
                            customer_node = f"CUSTOMER_{row['CUSTOMER_ID']}"
                            product_node = f"PRODUCT_{row['PRODUCT_CARD_ID']}"
                            if self.knowledge_graph.has_node(customer_node) and self.knowledge_graph.has_node(product_node):
                                self.knowledge_graph.add_edge(customer_node, product_node, relationship="purchased")
                        
                        # Product-Category relationships
                        if pd.notna(row.get('PRODUCT_CARD_ID')) and pd.notna(row.get('CATEGORY_NAME')):
                            product_node = f"PRODUCT_{row['PRODUCT_CARD_ID']}"
                            category_node = f"CATEGORY_{row['CATEGORY_NAME']}"
                            if self.knowledge_graph.has_node(product_node) and self.knowledge_graph.has_node(category_node):
                                self.knowledge_graph.add_edge(product_node, category_node, relationship="belongs_to")
                        
                        # Customer-City relationships
                        if pd.notna(row.get('CUSTOMER_ID')) and pd.notna(row.get('CUSTOMER_CITY')):
                            customer_node = f"CUSTOMER_{row['CUSTOMER_ID']}"
                            city_node = f"CITY_{row['CUSTOMER_CITY']}"
                            if self.knowledge_graph.has_node(customer_node) and self.knowledge_graph.has_node(city_node):
                                self.knowledge_graph.add_edge(customer_node, city_node, relationship="located_in")
                        
                        # Customer-Segment relationships
                        if pd.notna(row.get('CUSTOMER_ID')) and pd.notna(row.get('CUSTOMER_SEGMENT')):
                            customer_node = f"CUSTOMER_{row['CUSTOMER_ID']}"
                            segment_node = f"SEGMENT_{row['CUSTOMER_SEGMENT']}"
                            if self.knowledge_graph.has_node(customer_node) and self.knowledge_graph.has_node(segment_node):
                                self.knowledge_graph.add_edge(customer_node, segment_node, relationship="belongs_to")
                        
                        # Category-Department relationships
                        if pd.notna(row.get('CATEGORY_NAME')) and pd.notna(row.get('DEPARTMENT_NAME')):
                            category_node = f"CATEGORY_{row['CATEGORY_NAME']}"
                            department_node = f"DEPARTMENT_{row['DEPARTMENT_NAME']}"
                            if self.knowledge_graph.has_node(category_node) and self.knowledge_graph.has_node(department_node):
                                self.knowledge_graph.add_edge(category_node, department_node, relationship="part_of")
                    
                    except Exception as e:
                        
                        continue
            
            logger.info(f"Enhanced knowledge graph built: {self.knowledge_graph.number_of_nodes()} nodes, {self.knowledge_graph.number_of_edges()} edges")
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            # Initialize empty graph on error
            self.knowledge_graph = nx.Graph()
    def create_knowledge_graph_visualization(self):
        """knowledge graph visualization for large datasets"""
        try:
            if self.knowledge_graph.number_of_nodes() == 0:
                return None

            # Getting the original graph
            original_graph = self.knowledge_graph
            
            # For very large graphs, diff average strategy
            total_nodes = original_graph.number_of_nodes()
            
            if total_nodes > 500:  # Reduced threshold for better performance
                
                node_types = {}
                for node, data in original_graph.nodes(data=True):
                    node_type = data.get('type', 'unknown')
                    if node_type not in node_types:
                        node_types[node_type] = []
                    node_types[node_type].append(node)

                
                sampled_nodes = []
                type_limits = {
                    'customer': 50,    # Reduced from customers
                    'product': 30,     # Keep more products  
                    'category': 20,    # Keep most categories
                    'department': 11,  # Keep all departments
                    'city': 25,        # Sample cities
                    'segment': 3       # Keep all segments
                }
                
                for node_type, nodes in node_types.items():
                    limit = type_limits.get(node_type, 20)
                    if len(nodes) > limit:
                        # Use numpy for better random sampling
                        import numpy as np
                        sampled = np.random.choice(nodes, limit, replace=False).tolist()
                        sampled_nodes.extend(sampled)
                    else:
                        sampled_nodes.extend(nodes)

                # Creating subgraph with sampled nodes
                try:
                    graph = original_graph.subgraph(sampled_nodes).copy()
                except Exception as subgraph_error:
                    print(f"Subgraph creation error: {subgraph_error}")
                    # Fallback: create smaller sample
                    all_nodes = list(original_graph.nodes())
                    small_sample = all_nodes[:100]  # Very conservative fallback
                    graph = original_graph.subgraph(small_sample).copy()
            else:
                graph = original_graph

            # Verifying graph
            if graph.number_of_nodes() == 0:
                print("Graph has no nodes after sampling")
                return None

            print(f"Visualization using {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

            # Creating layout with error handling and timeout
            pos = None
            layout_attempts = [
                ('spring', lambda g: nx.spring_layout(g, k=2, iterations=30, seed=42)),
                ('circular', lambda g: nx.circular_layout(g)),
                ('random', lambda g: {node: (np.random.random(), np.random.random()) for node in g.nodes()})
            ]
            
            for layout_name, layout_func in layout_attempts:
                try:
                    print(f"Trying {layout_name} layout...")
                    pos = layout_func(graph)
                    print(f"{layout_name} layout successful")
                    break
                except Exception as layout_error:
                    print(f"{layout_name} layout failed: {layout_error}")
                    continue
            
            if pos is None:
                print("All layout attempts failed")
                return None

            # Preparing node traces by type with better error handling
            node_traces = {}
            node_colors = {
                'customer': '#FF6B6B',
                'product': '#4ECDC4', 
                'category': '#45B7D1',
                'department': '#96CEB4',
                'city': '#FFEAA7',
                'segment': '#DDA0DD',
                'unknown': '#888888'
            }

            # Initializing traces for each node type
            for node, data in graph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                if node_type not in node_traces:
                    node_traces[node_type] = {
                        'x': [], 'y': [], 'text': [], 
                        'name': f"{node_type.title()} ({node_types.get(node_type, [0]).__len__() if 'node_types' in locals() else '?'})",
                        'hovertext': []
                    }

                try:
                    if node in pos:
                        x, y = pos[node]
                        
                        # Validating coordinates
                        if not (np.isfinite(x) and np.isfinite(y)):
                            continue
                            
                        node_traces[node_type]['x'].append(x)
                        node_traces[node_type]['y'].append(y)

                        # Creating display name
                        display_name = str(node).replace('_', ' ')
                        if len(display_name) > 25:
                            display_name = display_name[:22] + "..."

                        node_traces[node_type]['text'].append(display_name)
                        node_traces[node_type]['hovertext'].append(
                            f"<b>Type:</b> {node_type}<br><b>ID:</b> {node}<br><b>Connections:</b> {graph.degree(node)}"
                        )
                except Exception as node_error:
                    print(f"Error processing node {node}: {node_error}")
                    continue

            # Creating edge traces with limits
            edge_x, edge_y = [], []
            edge_count = 0
            max_edges = 300  

            try:
                for edge in graph.edges():
                    if edge_count >= max_edges:
                        break
                        
                    if edge[0] in pos and edge[1] in pos:
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        
                        # Validating edge coordinates
                        if all(np.isfinite([x0, y0, x1, y1])):
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            edge_count += 1
            except Exception as edge_error:
                print(f"Error processing edges: {edge_error}")

            # Creating plotly figure
            fig = go.Figure()

            # Adding edges first
            if edge_x and edge_y and len(edge_x) > 0:
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='rgba(136,136,136,0.2)'),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False,
                    name='Connections'
                ))

            # Adding node traces
            nodes_added = 0
            for node_type, trace_data in node_traces.items():
                if trace_data['x'] and len(trace_data['x']) > 0:
                    try:
                        fig.add_trace(go.Scatter(
                            x=trace_data['x'],
                            y=trace_data['y'],
                            mode='markers',
                            hoverinfo='text',
                            text=trace_data['hovertext'],
                            name=trace_data['name'],
                            marker=dict(
                                size=10,
                                color=node_colors.get(node_type, '#888888'),
                                line=dict(width=1, color='white'),
                                opacity=0.8
                            )
                        ))
                        nodes_added += 1
                    except Exception as trace_error:
                        print(f"Error adding trace for {node_type}: {trace_error}")
                        continue

            if nodes_added == 0:
                print("No node traces were successfully added")
                return None

            # Updating layout
            fig.update_layout(
                title=dict(
                    text=f'Knowledge Graph Visualization<br><sub>Showing {graph.number_of_nodes()} of {total_nodes} nodes ({graph.number_of_edges()} connections)</sub>',
                    x=0.5,
                    font=dict(size=16)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                annotations=[dict(
                    text="Hover over nodes for details • Large datasets are sampled for performance",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.02,
                    xanchor="center", yanchor="bottom",
                    font=dict(color="#666", size=10)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                height=600,
                legend=dict(
                    yanchor="top", y=0.99,
                    xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                ),
                plot_bgcolor='rgba(248,248,248,0.8)'
            )

            print(f"Successfully created visualization with {nodes_added} node types")
            return fig

        except Exception as e:
            print(f"Critical error in knowledge graph visualization: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    
    
    # ANALYTICS METHODS
    
    
    def calculate_rfm_analysis(self, supply_chain_df):
        """RFM analysis"""
        if supply_chain_df is None or supply_chain_df.empty:
            return None
        
        available_cols = supply_chain_df.columns.tolist()
        logger.info(f"Available columns for RFM: {available_cols}")
        
        
        customer_col = 'CUSTOMER_ID' if 'CUSTOMER_ID' in available_cols else None
        order_col = 'ORDER_ID' if 'ORDER_ID' in available_cols else None
        date_col = 'ORDER_DATE' if 'ORDER_DATE' in available_cols else None
        monetary_col = 'SALES' if 'SALES' in available_cols else None
        
        logger.info(f"RFM columns detected - Customer: {customer_col}, Order: {order_col}, Date: {date_col}, Monetary: {monetary_col}")
        
        if not all([customer_col, order_col, date_col, monetary_col]):
            missing = []
            if not customer_col: missing.append('CUSTOMER_ID')
            if not order_col: missing.append('ORDER_ID') 
            if not date_col: missing.append('ORDER_DATE')
            if not monetary_col: missing.append('SALES')
            
            error_msg = f"RFM analysis requires all columns. Missing: {', '.join(missing)}. Available columns: {', '.join(available_cols)}"
            logger.warning(error_msg)
            return None
        
        try:
            
            df_rfm = supply_chain_df.copy()
            
            # Converting date column 
            if df_rfm[date_col].dtype == 'object':
                df_rfm[date_col] = pd.to_datetime(df_rfm[date_col], errors='coerce')
            
            # Removing rows with invalid dates
            df_rfm = df_rfm.dropna(subset=[date_col, customer_col, monetary_col])
            
            if df_rfm.empty:
                logger.warning("No valid date data for RFM analysis")
                return None
            
            # Calculating reference date
            today = df_rfm[date_col].max()
            logger.info(f"Using reference date for RFM: {today}")
            
            # Grouping by customer
            rfm = df_rfm.groupby(customer_col).agg({
                date_col: lambda x: (today - x.max()).days,
                order_col: 'count',
                monetary_col: 'sum'
            }).reset_index()
            
            # Data availability check
            if rfm.empty:
                logger.warning("RFM analysis resulted in empty dataframe")
                return None
            
            rfm.columns = ['CUSTOMER_ID', 'Recency', 'Frequency', 'Monetary']
            
            
            try:
                if len(rfm) >= 5:
                    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
                    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
                    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
                else:
                    
                    rfm['R_Score'] = rfm['Recency'].rank(ascending=False, method='first').apply(lambda x: min(5, max(1, int(x))))
                    rfm['F_Score'] = rfm['Frequency'].rank(ascending=True, method='first').apply(lambda x: min(5, max(1, int(x))))
                    rfm['M_Score'] = rfm['Monetary'].rank(ascending=True, method='first').apply(lambda x: min(5, max(1, int(x))))
            except ValueError as e:
                logger.warning(f"Error in quantile calculation: {e}. Using rank-based scoring instead.")
                
                rfm['R_Score'] = rfm['Recency'].rank(ascending=False, method='first').apply(lambda x: min(5, max(1, int((x-1) / len(rfm) * 5) + 1)))
                rfm['F_Score'] = rfm['Frequency'].rank(ascending=True, method='first').apply(lambda x: min(5, max(1, int((x-1) / len(rfm) * 5) + 1)))
                rfm['M_Score'] = rfm['Monetary'].rank(ascending=True, method='first').apply(lambda x: min(5, max(1, int((x-1) / len(rfm) * 5) + 1)))
            
            def segment_customers(row):
                try:
                    r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
                    if r >= 4 and f >= 4 and m >= 4:
                        return 'Champions'
                    elif r >= 3 and f >= 3 and m >= 3:
                        return 'Loyal Customers'
                    elif r >= 3 and f <= 2:
                        return 'Potential Loyalists'
                    elif r >= 4 and f <= 2:
                        return 'New Customers'
                    elif r <= 2 and f >= 3:
                        return 'At Risk'
                    else:
                        return 'Others'
                except:
                    return 'Others'
            
            rfm['Customer_Segment'] = rfm.apply(segment_customers, axis=1)
            self.rfm_data = rfm
            
            logger.info(f"RFM analysis completed successfully with {len(rfm)} customers")
            return rfm
            
        except Exception as e:
            logger.error(f"Error in RFM analysis: {e}")
            return None

    def analyze_cross_functional_impact(self, supply_df):
        """Analyze cross-functional impact between shipping modes and delivery performance"""
        try:
            if supply_df is None or supply_df.empty:
                return None, None
            
            
            required_cols = ['SHIPPING_MODE', 'LATE_DELIVERY_RISK']
            if not all(col in supply_df.columns for col in required_cols):
                return None, None
            
            # Shipping mode analysis
            delay_impact = supply_df.groupby('SHIPPING_MODE').agg({
                'LATE_DELIVERY_RISK': ['mean', 'count'],
                'SALES': 'sum' if 'SALES' in supply_df.columns else 'count'
            }).round(3)
            
            delay_impact.columns = ['Avg_Late_Delivery_Risk', 'Order_Count', 'Total_Sales']
            delay_impact = delay_impact.reset_index()
            
            # Segment vs shipping analysis
            segment_shipping = None
            if 'CUSTOMER_SEGMENT' in supply_df.columns:
                segment_shipping = supply_df.groupby(['CUSTOMER_SEGMENT', 'SHIPPING_MODE']).agg({
                    'LATE_DELIVERY_RISK': 'mean',
                    'SALES': 'sum' if 'SALES' in supply_df.columns else 'count'
                }).round(3).reset_index()
            
            return delay_impact, segment_shipping
            
        except Exception as e:
            logger.error(f"Error in cross-functional impact analysis: {e}")
            return None, None


    # AI Assistant Methods - Query Routing System
    
    
    def create_advanced_query_router(self):
        """Creating advanced query routing for better question handling"""
        
        # Defining question patterns and their handlers
        self.query_patterns = {
            # Customer Analysis Patterns
            r'customer (\d+) (?:purchase )?history': 'get_customer_purchase_history',
            r'top purchases (?:by )?customer (\d+)': 'get_customer_top_purchases',
            r'customers? (?:who )?spend (?:the )?most': 'get_top_spending_customers',
            r'orders? (?:has )?customer (\d+)': 'get_customer_order_count',
            r'revenue (?:from )?customer (\d+)': 'get_customer_revenue_analysis',
            
            # Product Analysis Patterns
            r'products? (?:generate|with) (?:the )?most revenue': 'get_top_revenue_products',
            r'product (\d+) performance': 'get_product_analysis',
            r'top selling products?': 'get_top_selling_products',
            r'compare product categories': 'compare_product_categories',
            
            # Sales & Revenue Patterns
            r'total revenue': 'get_total_revenue',
            r'revenue trends?': 'get_revenue_trends',
            r'customer segment.* revenue': 'get_segment_revenue',
            r'average order value': 'get_average_order_value',
            
            # Supply Chain Patterns
            r'average shipping time': 'get_avg_shipping_time',
            r'shipping modes?.* delivery risk': 'get_shipping_risk_analysis',
            r'supply chain health': 'generate_supply_chain_health',
            r'late delivery rate': 'get_late_delivery_rate',
            
            # Data Quality Patterns
            r'check data quality': 'generate_enhanced_data_quality_report',
            r'missing values?': 'get_missing_values_report',
            r'data structure': 'generate_schema_summary',
            r'duplicates?': 'get_duplicate_analysis',
            
            # Seasonal and Trends
            r'seasonal patterns?': 'get_seasonal_patterns',
        }

    def enhanced_query_processor(self, query, supply_df, access_df):
        """Processing queries with pattern matching and NLP fallback"""
        
        # Exact matches
        for pattern, method_name in self.query_patterns.items():
            match = re.search(pattern, query.lower())
            if match:
                # Method by name
                method = getattr(self, method_name, None)
                if method:
                    # Extracting parameters 
                    params = match.groups()
                    try:
                        if params:
                            return method(params[0], supply_df)
                        else:
                            return method(supply_df)
                    except TypeError:
                        # Trying without parameters
                        try:
                            return method(supply_df)
                        except:
                            continue
                    except Exception as e:
                        logger.warning(f"Error in pattern handler {method_name}: {e}")
                        continue
        
        
        return self.generate_enhanced_ai_response(query, supply_df, access_df)

    def extract_entities_from_query(self, query):
        """Extracting customer IDs, product IDs, and other entities from natural language"""
        if not self.ai_enabled:
            return {}
        
        try:
            # Using NER to extract entities
            entities = self.ner_pipeline(query)
            
            # Extracting numbers that could be IDs
            numbers = re.findall(r'\b\d+\b', query)
            
            # Extracting potential customer/product identifiers
            customer_patterns = re.findall(r'customer\s*(\w+)', query.lower())
            product_patterns = re.findall(r'product\s*(\w+)', query.lower())
            
            extracted = {
                'customer_ids': customer_patterns + [f"CUST{num.zfill(3)}" for num in numbers if len(num) <= 3],
                'product_ids': product_patterns + [f"PROD{num.zfill(3)}" for num in numbers if len(num) <= 3],
                'raw_numbers': numbers,
                'entities': entities,
                'potential_categories': [entity['word'] for entity in entities if entity['entity_group'] == 'MISC']
            }
            
            return extracted
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {}

    def classify_query_intent(self, query):
        """Classifying the intent of the user query using zero-shot classification"""
        if not self.ai_enabled:
            return "general"
        
        try:
            candidate_labels = [
                "data_retrieval",  # Specific data requests
                "analysis",        # General analysis requests
                "prediction",      # ML/forecasting requests
                "comparison",      # Comparative analysis
                "summary",         # Summary requests
                "quality_check"    # Data quality questions
            ]
            
            result = self.intent_classifier(query, candidate_labels)
            return result['labels'][0] if result['scores'][0] > 0.5 else "general"
        except Exception as e:
            logger.warning(f"Intent classification failed: {e}")
            return "general"
    
    # Customer and product analysis 
    
    
    def get_customer_purchase_history(self, customer_id, supply_df):
        """Getting specific customer's purchase history"""
        if supply_df is None or supply_df.empty:
            return "No supply chain data available."
    
        if 'CUSTOMER_ID' not in supply_df.columns:
            return "Customer ID column not found in the data."
    
        try:
            
            customer_id_str = str(customer_id)
            customer_id_int = None
        
            try:
                customer_id_int = int(customer_id)
            except (ValueError, TypeError):
                pass
        
            
            customer_data = pd.DataFrame()
        
            # Exact match with original type
            customer_data = supply_df[supply_df['CUSTOMER_ID'] == customer_id]
        
            # Exact match with integer conversion
            if customer_data.empty and customer_id_int is not None:
                customer_data = supply_df[supply_df['CUSTOMER_ID'] == customer_id_int]
        
            # Exact match with string conversion
            if customer_data.empty:
                customer_data = supply_df[supply_df['CUSTOMER_ID'].astype(str) == customer_id_str]
        
            # Partial string matching (as fallback)
            if customer_data.empty:
                customer_data = supply_df[supply_df['CUSTOMER_ID'].astype(str).str.contains(customer_id_str, case=False, na=False)]
        
            if customer_data.empty:
                
                available_customers = supply_df['CUSTOMER_ID'].unique()
                sample_customers = list(available_customers[:10])
                return f"Customer {customer_id} not found. Here are some available customer IDs: {sample_customers}"
        
            # Building comprehensive purchase history
            total_orders = len(customer_data)
            total_spent = customer_data['SALES'].sum() if 'SALES' in customer_data.columns else 0
            avg_order = customer_data['SALES'].mean() if 'SALES' in customer_data.columns else 0
        
            history = f"""
**Purchase History for Customer {customer_id}**

**Summary:**
- Total Orders: {total_orders}
- Total Spent: ${total_spent:,.2f}
- Average Order Value: ${avg_order:.2f}

**Recent Orders:**
"""
        
            # Showing recent orders with details
            display_cols = ['ORDER_DATE', 'PRODUCT_CARD_ID', 'CATEGORY_NAME', 'SALES', 'SHIPPING_MODE']
            available_cols = [col for col in display_cols if col in customer_data.columns]
        
            if available_cols:
                recent_orders = customer_data[available_cols].tail(10)
                for _, order in recent_orders.iterrows():
                    history += f"\n- "
                    for col in available_cols:
                        if pd.notna(order[col]):
                            history += f"{col.replace('_', ' ').title()}: {order[col]} | "
                    history = history.rstrip(" | ")
        
            # Adding segment information 
            if 'CUSTOMER_SEGMENT' in customer_data.columns:
                segment = customer_data['CUSTOMER_SEGMENT'].iloc[0]
                history += f"\n\n**Customer Segment:** {segment}"
        
            return history
        
        except Exception as e:
            logger.error(f"Error getting customer history: {e}")
            return f"Error retrieving purchase history for customer {customer_id}: {e}"

    def get_customer_top_purchases(self, customer_id, supply_df):
        """Getting top purchases for a specific customer"""
        try:
            customer_data = supply_df[supply_df['CUSTOMER_ID'].astype(str).str.contains(str(customer_id), case=False, na=False)]
            
            if customer_data.empty:
                return f"No data found for customer {customer_id}"
            
            if 'SALES' not in customer_data.columns:
                return "Sales data not available for purchase analysis"
            
            # Getting top purchases
            top_purchases = customer_data.nlargest(10, 'SALES')
            
            result = f"**Top Purchases for Customer {customer_id}**\n\n"
            
            for i, (_, purchase) in enumerate(top_purchases.iterrows(), 1):
                result += f"{i}. "
                if 'PRODUCT_CARD_ID' in purchase:
                    result += f"Product {purchase['PRODUCT_CARD_ID']}: "
                result += f"${purchase['SALES']:.2f}"
                if 'ORDER_DATE' in purchase:
                    result += f" (Date: {purchase['ORDER_DATE']})"
                result += "\n"
            
            return result
        except Exception as e:
            return f"Error analyzing top purchases for customer {customer_id}: {e}"

    def get_top_spending_customers(self, supply_df):
        """Getting customers who spend the most"""
        try:
            if 'CUSTOMER_ID' not in supply_df.columns or 'SALES' not in supply_df.columns:
                return "Customer spending analysis requires CUSTOMER_ID and SALES columns"
            
            customer_spending = supply_df.groupby('CUSTOMER_ID')['SALES'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
            top_customers = customer_spending.head(10)
            
            result = "**Top 10 Customers by Total Spending**\n\n"
            
            for i, (customer_id, data) in enumerate(top_customers.iterrows(), 1):
                result += f"{i}. Customer {customer_id}:\n"
                result += f"   - Total Spent: ${data['sum']:,.2f}\n"
                result += f"   - Orders: {int(data['count'])}\n"
                result += f"   - Avg Order: ${data['mean']:.2f}\n\n"
            
            return result
        except Exception as e:
            return f"Error analyzing top spending customers: {e}"

    def get_top_revenue_products(self, supply_df):
        """Getting products generating the most revenue"""
        try:
            if 'PRODUCT_CARD_ID' not in supply_df.columns or 'SALES' not in supply_df.columns:
                return "Product revenue analysis requires PRODUCT_CARD_ID and SALES columns"
            
            product_revenue = supply_df.groupby('PRODUCT_CARD_ID')['SALES'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
            top_products = product_revenue.head(10)
            
            result = "**Top 10 Products by Revenue**\n\n"
            
            for i, (product_id, data) in enumerate(top_products.iterrows(), 1):
                result += f"{i}. Product {product_id}:\n"
                result += f"   - Total Revenue: ${data['sum']:,.2f}\n"
                result += f"   - Orders: {int(data['count'])}\n"
                result += f"   - Avg per Order: ${data['mean']:.2f}\n\n"
            
            return result
        except Exception as e:
            return f"Error analyzing top revenue products: {e}"

    def get_avg_shipping_time(self, supply_df):
        """Getting average shipping time analysis"""
        try:
            if 'DAYS_FOR_SHIPPING_REAL' not in supply_df.columns:
                return "Shipping time analysis requires DAYS_FOR_SHIPPING_REAL column"
            
            avg_shipping = supply_df['DAYS_FOR_SHIPPING_REAL'].fillna(0).mean()
            median_shipping = supply_df['DAYS_FOR_SHIPPING_REAL'].fillna(0).median()
            min_shipping = supply_df['DAYS_FOR_SHIPPING_REAL'].fillna(0).min()
            max_shipping = supply_df['DAYS_FOR_SHIPPING_REAL'].fillna(0).max()
            
            result = f"""
**Shipping Time Analysis**

- **Average Shipping Time**: {avg_shipping:.1f} days
- **Median Shipping Time**: {median_shipping:.1f} days
- **Fastest Delivery**: {min_shipping:.0f} days
- **Slowest Delivery**: {max_shipping:.0f} days
"""
            
            # Adding shipping mode breakdown 
            if 'SHIPPING_MODE' in supply_df.columns:
                shipping_by_mode = supply_df.groupby('SHIPPING_MODE')['DAYS_FOR_SHIPPING_REAL'].mean().sort_values()
                result += "\n**Average Shipping Time by Mode:**\n"
                for mode, avg_time in shipping_by_mode.items():
                    result += f"- {mode}: {avg_time:.1f} days\n"
            
            return result
        except Exception as e:
            return f"Error analyzing shipping times: {e}"

    def generate_enhanced_ai_response(self, query, supply_df, access_df):
        """AI response with NLP query parsing and pattern matching first"""
        if supply_df is None or supply_df.empty:
            return "No data is currently loaded. Please check your Snowflake connection and table access."
        
        try:
            # Pattern matching 
            if hasattr(self, 'query_patterns') and hasattr(self, 'enhanced_query_processor'):
                pattern_response = self.enhanced_query_processor(query, supply_df, access_df)
                
                if pattern_response and pattern_response != query:  
                    return pattern_response
            
            # Extracting entities and classifying intent
            entities = self.extract_entities_from_query(query) if hasattr(self, 'extract_entities_from_query') else {}
            intent = self.classify_query_intent(query) if hasattr(self, 'classify_query_intent') else "general"
            
            logger.info(f"Query intent: {intent}, Entities: {entities}")
            query_lower = query.lower()
            
            # Extracting customer ID from query with multiple patterns
            customer_id = None
        
            # Pattern 1: "customer 10374", "customer id 10374"
            customer_patterns = re.findall(r'customer\s*(?:id\s*)?(\d+)', query_lower)
            if customer_patterns:
                customer_id = customer_patterns[0]
        
            # Pattern 2: numbers that could be customer IDs
            if not customer_id:
                numbers = re.findall(r'\b(\d{4,5})\b', query)  # 4-5 digit numbers
                if numbers and any(word in query_lower for word in ['customer', 'purchase', 'buy', 'order']):
                    customer_id = numbers[0]
        
            # Pattern 3: From extracted entities
            if not customer_id and entities and entities.get('customer_ids'):
                customer_id = entities['customer_ids'][0]
        
            #Customer specific queries
            if customer_id and any(word in query_lower for word in ['history', 'purchase', 'order', 'buy', 'top']):
                return self.get_customer_purchase_history(customer_id, supply_df)
            
            # Data retrieval requests
            if intent == "data_retrieval":
                # Customer-specific queries
                if entities.get('customer_ids') and any(word in query.lower() for word in ['history', 'purchase', 'order', 'buy']):
                    customer_id = entities['customer_ids'][0]
                    return self.get_customer_purchase_history(customer_id, supply_df)
                
                # Product-specific queries
                elif entities.get('product_ids') and any(word in query.lower() for word in ['product', 'item', 'sales']):
                    product_id = entities['product_ids'][0]
                    return self.get_product_analysis(product_id, supply_df)
                
                
                elif entities.get('raw_numbers'):
                    number = entities['raw_numbers'][0]
                    if any(word in query.lower() for word in ['customer', 'client']):
                        return self.get_customer_purchase_history(number, supply_df)
                    elif any(word in query.lower() for word in ['product', 'item']):
                        return self.get_product_analysis(number, supply_df)
            
            # Contextual question answering 
            if intent in ["data_retrieval", "analysis"] and "?" in query:
                qa_result = self.answer_contextual_question(query, supply_df)
                if qa_result:
                    return f"**AI Answer:** {qa_result}\n\n" + self.get_relevant_data_context(query, supply_df)
            
            # Category-based routing 
            query_lower = query.lower()
            
            # Revenue related queries 
            if any(word in query_lower for word in ['revenue', 'sales', 'money', 'profit']) or intent == "analysis":
                if entities.get('customer_ids'):
                    customer_id = entities['customer_ids'][0]
                    return self.get_customer_revenue_analysis(customer_id, supply_df)
                elif entities.get('product_ids'):
                    product_id = entities['product_ids'][0]
                    return self.get_product_revenue_analysis(product_id, supply_df)
                else:
                    return self.generate_enhanced_revenue_summary(supply_df)
            
            # Customer related queries 
            elif any(word in query_lower for word in ['customer', 'client', 'buyer']):
                if entities.get('customer_ids') or entities.get('raw_numbers'):
                    customer_id = entities['customer_ids'][0] if entities.get('customer_ids') else entities['raw_numbers'][0]
                    return self.get_customer_purchase_history(customer_id, supply_df)
                else:
                    return self.generate_enhanced_customer_insights(supply_df)
            
            # Product related queries
            elif any(word in query_lower for word in ['product', 'item', 'inventory']):
                if entities.get('product_ids') or entities.get('raw_numbers'):
                    product_id = entities['product_ids'][0] if entities.get('product_ids') else entities['raw_numbers'][0]
                    return self.get_product_analysis(product_id, supply_df)
                else:
                    return self.get_product_overview(supply_df)
            
            # Supply chain related queries
            elif any(word in query_lower for word in ['shipping', 'delivery', 'supply', 'logistics']):
                return self.generate_supply_chain_health(supply_df)
            
            # Comparison queries
            elif intent == "comparison" or any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
                return self.generate_comparison_analysis(query, supply_df)
            
            # Prediction queries
            elif intent == "prediction" or any(word in query_lower for word in ['predict', 'forecast', 'future']):
                return self.generate_ml_predictions_summary(supply_df)
            
            # Data quality queries
            elif any(word in query_lower for word in ['quality', 'missing', 'clean', 'duplicate']):
                return self.generate_enhanced_data_quality_report(supply_df)
            
            # Column/schema queries
            elif any(word in query_lower for word in ['columns', 'fields', 'structure', 'schema']):
                return self.generate_schema_summary(supply_df)
            
            # Top performers queries
            elif any(word in query_lower for word in ['top', 'best', 'highest', 'most']):
                return self.generate_top_performers_summary(supply_df)
            
            # Default with enhanced context
            else:
                contextual_answer = self.answer_contextual_question(query, supply_df)
                if contextual_answer:
                    return f"**AI Answer:** {contextual_answer}\n\n{self.generate_data_overview(supply_df)}"
                else:
                    return f"I can help you analyze your data! Try asking specific questions like 'show customer 10 history' or 'analyze product 5 sales'. Current data includes {len(supply_df)} supply chain records with {len(supply_df.columns)} columns."
            
        except Exception as e:
            logger.error(f"Error generating enhanced AI response: {e}")
            return "Sorry, I encountered an error while analyzing your query. Please try again with a different question."
    
    # Enhanced Visualization
    
    
    def create_forecast_visualization(self):
        """Prophet forecast visualization """
        try:
            if not hasattr(self, 'forecast_model') or self.forecast_model is None or self.forecast_data is None:
                return None
            
            forecast_df = self.forecast_data.copy()
            
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Time Series Forecast with Confidence Intervals', 'Trend and Seasonal Components'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,100,80,0)',
                    name='Confidence Interval',
                    fillcolor='rgba(68, 168, 234, 0.3)'
                ),
                row=1, col=1
            )
            
            # Trend component
            fig.add_trace(
                go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['trend'],
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
            
            # Weekly seasonality if available
            if 'weekly' in forecast_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df['ds'],
                        y=forecast_df['weekly'],
                        mode='lines',
                        name='Weekly Seasonality',
                        line=dict(color='green', width=1)
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title='Prophet Time Series Forecast',
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Values", row=1, col=1)
            fig.update_yaxes(title_text="Components", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating forecast visualization: {e}")
            return None

    def create_churn_model_visualization(self):
        """NEW: Create churn prediction model visualization"""
        try:
            if not self.churn_results:
                return None, None
            
            # Confusion Matrix Heatmap
            conf_matrix = self.churn_results['confusion_matrix']
            
            fig_confusion = go.Figure(data=go.Heatmap(
                z=conf_matrix,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale='Blues',
                text=conf_matrix,
                texttemplate="%{text}",
                textfont={"size": 16},
                showscale=True
            ))
            
            fig_confusion.update_layout(
                title='Confusion Matrix - Churn Prediction Model',
                xaxis_title='Predicted',
                yaxis_title='Actual',
                width=500,
                height=400
            )
            
            # Feature Importance Chart
            if 'feature_importance' in self.churn_results:
                feature_imp = self.churn_results['feature_importance']
                features = list(feature_imp.keys())
                importance = list(feature_imp.values())
                
                fig_importance = go.Figure(data=[
                    go.Bar(x=importance, y=features, orientation='h')
                ])
                
                fig_importance.update_layout(
                    title='Feature Importance - Churn Prediction Model',
                    xaxis_title='Importance Score',
                    yaxis_title='Features',
                    width=500,
                    height=400
                )
                
                return fig_confusion, fig_importance
            
            return fig_confusion, None
            
        except Exception as e:
            logger.error(f"Error creating churn model visualization: {e}")
            return None, None

    def create_anomaly_visualization(self):
        """Anomaly detection visualization"""
        try:
            if not self.anomaly_results or 'anomaly_data' not in self.anomaly_results:
                return None
            
            anomaly_df = self.anomaly_results['anomaly_data']
            
            
            numeric_cols = anomaly_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return None
            
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            
            
            fig = go.Figure()
            
            # Normal Points
            normal_data = anomaly_df[~anomaly_df['is_anomaly']]
            fig.add_trace(go.Scatter(
                x=normal_data[x_col],
                y=normal_data[y_col],
                mode='markers',
                name='Normal',
                marker=dict(color='blue', size=6, opacity=0.6),
                hovertemplate=f'<b>Normal Point</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>'
            ))
            
            # Anomaly points
            anomaly_data = anomaly_df[anomaly_df['is_anomaly']]
            if not anomaly_data.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_data[x_col],
                    y=anomaly_data[y_col],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='red', size=10, symbol='x'),
                    hovertemplate=f'<b>Anomaly</b><br>{x_col}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>'
                ))
            
            fig.update_layout(
                title=f'Anomaly Detection Results: {x_col} vs {y_col}',
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                height=500,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating anomaly visualization: {e}")
            return None

   
    # Additional Analysis 
    
    
    def get_product_analysis(self, product_id, supply_df):
        """Specific product analysis"""
        if supply_df is None or supply_df.empty:
            return "No supply chain data available."
        
        if 'PRODUCT_CARD_ID' not in supply_df.columns:
            return "Product ID column not found in the data."
        
        try:
            # Trying exact match 
            product_data = supply_df[supply_df['PRODUCT_CARD_ID'] == product_id]
            
            # If no exact match, trying partial matching
            if product_data.empty:
                product_data = supply_df[supply_df['PRODUCT_CARD_ID'].astype(str).str.contains(str(product_id), case=False, na=False)]
            
            if product_data.empty:
                available_products = supply_df['PRODUCT_CARD_ID'].unique()[:10]
                return f"Product {product_id} not found. Available products include: {', '.join(map(str, available_products))}"
            
            # Building product analysis
            total_orders = len(product_data)
            total_revenue = product_data['SALES'].sum() if 'SALES' in product_data.columns else 0
            unique_customers = product_data['CUSTOMER_ID'].nunique() if 'CUSTOMER_ID' in product_data.columns else 0
            
            analysis = f"""
**Product Analysis for {product_id}**

**Performance Metrics:**
- Total Orders: {total_orders}
- Total Revenue: ${total_revenue:,.2f}
- Unique Customers: {unique_customers}
- Average Order Value: ${total_revenue/total_orders:.2f}

"""
            
            # Adding category and department info
            if 'CATEGORY_NAME' in product_data.columns:
                category = product_data['CATEGORY_NAME'].iloc[0]
                analysis += f"**Category:** {category}\n"
            
            if 'DEPARTMENT_NAME' in product_data.columns:
                department = product_data['DEPARTMENT_NAME'].iloc[0]
                analysis += f"**Department:** {department}\n"
            
            # Top customers for this product
            if 'CUSTOMER_ID' in product_data.columns and 'SALES' in product_data.columns:
                top_customers = product_data.groupby('CUSTOMER_ID')['SALES'].sum().nlargest(3)
                analysis += f"\n**Top 3 Customers:**\n"
                for i, (customer, sales) in enumerate(top_customers.items(), 1):
                    analysis += f"{i}. Customer {customer}: ${sales:,.2f}\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting product analysis: {e}")
            return f"Error retrieving analysis for product {product_id}"

    def answer_contextual_question(self, question, supply_df):
        """Using question-answering model with data context"""
        if not self.ai_enabled or supply_df is None or supply_df.empty:
            return None
        
        try:
            
            context = f"""
            Supply chain dataset with {len(supply_df)} records and {len(supply_df.columns)} columns.
            """
            
            if 'SALES' in supply_df.columns:
                total_sales = supply_df['SALES'].sum()
                avg_sales = supply_df['SALES'].mean()
                context += f" Total sales: ${total_sales:,.2f}. Average order: ${avg_sales:.2f}."
            
            if 'CUSTOMER_ID' in supply_df.columns:
                unique_customers = supply_df['CUSTOMER_ID'].nunique()
                context += f" {unique_customers} unique customers."
            
            if 'CATEGORY_NAME' in supply_df.columns:
                top_category = supply_df['CATEGORY_NAME'].value_counts().index[0]
                context += f" Top category: {top_category}."
            
            
            result = self.qa_pipeline(question=question, context=context)
            
            if result['score'] > 0.1:  # Confidence threshold
                return result['answer']
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Contextual QA failed: {e}")
            return None

    def get_customer_revenue_analysis(self, customer_id, supply_df):
        """Revenue analysis for specific customer"""
        try:
            customer_data = supply_df[supply_df['CUSTOMER_ID'].astype(str).str.contains(str(customer_id), case=False, na=False)]
            
            if customer_data.empty:
                return f"No data found for customer {customer_id}"
            
            if 'SALES' not in customer_data.columns:
                return "Sales data not available for revenue analysis"
            
            total_revenue = customer_data['SALES'].sum()
            order_count = len(customer_data)
            avg_order = customer_data['SALES'].mean()
            
            return f"""
**Revenue Analysis for Customer {customer_id}**

- **Total Revenue**: ${total_revenue:,.2f}
- **Total Orders**: {order_count}
- **Average Order Value**: ${avg_order:.2f}
- **Revenue Rank**: {self.get_customer_revenue_rank(customer_id, supply_df)}
"""
        except Exception as e:
            return f"Error analyzing customer {customer_id} revenue: {e}"

    def get_customer_revenue_rank(self, customer_id, supply_df):
        """Customer's revenue ranking"""
        try:
            if 'CUSTOMER_ID' not in supply_df.columns or 'SALES' not in supply_df.columns:
                return "N/A"
            
            customer_revenues = supply_df.groupby('CUSTOMER_ID')['SALES'].sum().sort_values(ascending=False)
            rank = (customer_revenues.index == customer_id).argmax() + 1 if customer_id in customer_revenues.index else None
            total_customers = len(customer_revenues)
            
            return f"#{rank} of {total_customers}" if rank else "Not ranked"
        except:
            return "N/A"

    def get_product_overview(self, supply_df):
        """General product overview when no specific product requested"""
        try:
            if 'PRODUCT_CARD_ID' not in supply_df.columns:
                return "Product data not available"
            
            total_products = supply_df['PRODUCT_CARD_ID'].nunique()
            
            overview = f"""
**Product Portfolio Overview**

- **Total Unique Products**: {total_products:,}
- **Total Product Records**: {len(supply_df):,}
"""
            
            if 'SALES' in supply_df.columns:
                product_revenues = supply_df.groupby('PRODUCT_CARD_ID')['SALES'].sum()
                top_products = product_revenues.nlargest(5)
                
                overview += "\n**Top 5 Products by Revenue:**\n"
                for i, (product, revenue) in enumerate(top_products.items(), 1):
                    overview += f"{i}. Product {product}: ${revenue:,.2f}\n"
            
            if 'CATEGORY_NAME' in supply_df.columns:
                categories = supply_df['CATEGORY_NAME'].nunique()
                overview += f"\n**Product Categories**: {categories}"
            
            return overview
        except Exception as e:
            return f"Error generating product overview: {e}"

    def generate_comparison_analysis(self, query, supply_df):
        """Comparison analysis based on query context"""
        try:
            query_lower = query.lower()
            
            # Shipping mode comparison
            if 'shipping' in query_lower and 'SHIPPING_MODE' in supply_df.columns:
                comparison = supply_df.groupby('SHIPPING_MODE').agg({
                    'LATE_DELIVERY_RISK': 'mean',
                    'SALES': 'sum' if 'SALES' in supply_df.columns else 'count',
                    'CUSTOMER_ID': 'count'
                }).round(3)
                
                return f"""
**Shipping Mode Comparison**

{comparison.to_string()}
"""
            
            # Customer segment comparison
            elif 'segment' in query_lower and 'CUSTOMER_SEGMENT' in supply_df.columns:
                comparison = supply_df.groupby('CUSTOMER_SEGMENT').agg({
                    'SALES': ['sum', 'mean', 'count'] if 'SALES' in supply_df.columns else 'count'
                }).round(2)
                
                return f"""
**Customer Segment Comparison**

{comparison.to_string()}
"""
            
            # Category comparison
            elif 'category' in query_lower and 'CATEGORY_NAME' in supply_df.columns:
                comparison = supply_df.groupby('CATEGORY_NAME').agg({
                    'SALES': ['sum', 'mean'] if 'SALES' in supply_df.columns else 'count',
                    'CUSTOMER_ID': 'nunique'
                }).round(2)
                
                return f"""
**Category Performance Comparison**

{comparison.to_string()}
"""
            
            else:
                return "Please specify what you'd like to compare (e.g., shipping modes, customer segments, categories)"
                
        except Exception as e:
            return f"Error generating comparison analysis: {e}"

    def get_relevant_data_context(self, query, supply_df):
        """Getting relevant data context based on query"""
        try:
            context = f"\n**Data Context**: {len(supply_df)} records analyzed"
            
            if 'customer' in query.lower() and 'CUSTOMER_ID' in supply_df.columns:
                customer_count = supply_df['CUSTOMER_ID'].nunique()
                context += f" | {customer_count} unique customers"
            
            if 'product' in query.lower() and 'PRODUCT_CARD_ID' in supply_df.columns:
                product_count = supply_df['PRODUCT_CARD_ID'].nunique()
                context += f" | {product_count} unique products"
            
            return context
        except:
            return ""

    def generate_data_overview(self, supply_df):
        """Generating brief data overview for context"""
        try:
            return f"""
**Quick Data Overview**
- Records: {len(supply_df):,}
- Columns: {len(supply_df.columns)}
- Customers: {supply_df['CUSTOMER_ID'].nunique() if 'CUSTOMER_ID' in supply_df.columns else 'N/A'}
- Products: {supply_df['PRODUCT_CARD_ID'].nunique() if 'PRODUCT_CARD_ID' in supply_df.columns else 'N/A'}
"""
        except:
            return "Data overview unavailable"
    
    # Summary generation
    
    def generate_supply_chain_health(self, supply_df):
        """Supply chain health summary"""
        if supply_df is None or supply_df.empty:
            return "No supply chain data available for analysis."
        
        try:
            health = """
**Supply Chain Health Dashboard**

"""
            
            # Shipping performance 
            if 'DAYS_FOR_SHIPPING_REAL' in supply_df.columns:
                avg_shipping_days = supply_df['DAYS_FOR_SHIPPING_REAL'].fillna(0).mean()
                health += f"**Average Shipping Days**: {avg_shipping_days:.1f}\n\n"
            
            if 'DAYS_FOR_SHIPMENT_SCHEDULED' in supply_df.columns and 'DAYS_FOR_SHIPPING_REAL' in supply_df.columns:
                # Delivery performance
                supply_df_temp = supply_df.copy()
                supply_df_temp['DELAY'] = supply_df_temp['DAYS_FOR_SHIPPING_REAL'] - supply_df_temp['DAYS_FOR_SHIPMENT_SCHEDULED']
                avg_delay = supply_df_temp['DELAY'].fillna(0).mean()
                health += f"**Average Delivery Delay**: {avg_delay:.1f} days\n\n"
            
            # Delivery status distribution
            if 'DELIVERY_STATUS' in supply_df.columns:
                status_dist = supply_df['DELIVERY_STATUS'].value_counts()
                health += "**Delivery Status Distribution**:\n"
                for status, count in status_dist.items():
                    percentage = (count / len(supply_df)) * 100
                    health += f"- **{status}**: {count:,} orders ({percentage:.1f}%)\n"
                health += "\n"
            
            # Late delivery risk analysis
            if 'LATE_DELIVERY_RISK' in supply_df.columns:
                late_delivery_rate = supply_df['LATE_DELIVERY_RISK'].mean()
                health += f"**Late Delivery Risk Rate**: {late_delivery_rate:.1%}\n\n"
            
            # Shipping mode performance
            if 'SHIPPING_MODE' in supply_df.columns:
                mode_dist = supply_df['SHIPPING_MODE'].value_counts()
                health += "**Shipping Mode Distribution**:\n"
                for mode, count in mode_dist.items():
                    percentage = (count / len(supply_df)) * 100
                    health += f"- **{mode}**: {count:,} shipments ({percentage:.1f}%)\n"
                health += "\n"
           
            health += f"**Total Supply Chain Records**: {len(supply_df):,}"
            
            return health
        except Exception as e:
            logger.error(f"Error generating supply chain health: {e}")
            return "Error calculating supply chain health. Please check your data format."
    
    def generate_enhanced_revenue_summary(self, supply_df):
        """Enhanced revenue summary"""
        if supply_df is None or supply_df.empty:
            return "No supply chain data available for revenue analysis."
        
        try:
            
            if 'SALES' not in supply_df.columns:
                return "Revenue analysis requires SALES column from your dataset."
            
            # Null handling
            total_revenue = supply_df['SALES'].fillna(0).sum()
            avg_order_value = supply_df['SALES'].fillna(0).mean()
            max_transaction = supply_df['SALES'].fillna(0).max()
            min_transaction = supply_df['SALES'].fillna(0).min()
            median_value = supply_df['SALES'].fillna(0).median()
            
            
            summary = f"""
**Revenue Analysis Dashboard**

**Key Metrics**
- **Total Revenue**: ${total_revenue:,.2f}
- **Average Transaction**: ${avg_order_value:.2f}
- **Median Transaction**: ${median_value:.2f}
- **Transaction Range**: ${min_transaction:.2f} - ${max_transaction:.2f}

**Data Overview**
- **Records Analyzed**: {len(supply_df):,}
- **Primary Revenue Column**: SALES

"""
            
            # Segment-wise revenue 
            if 'CUSTOMER_SEGMENT' in supply_df.columns:
                segment_revenue = supply_df.groupby('CUSTOMER_SEGMENT')['SALES'].sum().sort_values(ascending=False)
                summary += "**Revenue by Customer Segment**:\n"
                for segment, revenue in segment_revenue.items():
                    summary += f"- **{segment}**: ${revenue:,.2f}\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error generating enhanced revenue summary: {e}")
            return "Error calculating revenue summary. Please check your data format."
    
    def generate_enhanced_customer_insights(self, supply_df):
        """Enhanced customer insights"""
        if supply_df is None or supply_df.empty:
            return "No supply chain data available for customer analysis."
        
        try:
            summary = """
**Customer Intelligence Dashboard**

"""
            
            
            if 'CUSTOMER_ID' in supply_df.columns:
                unique_customers = supply_df['CUSTOMER_ID'].nunique()
                summary += f"**Unique Customers**: {unique_customers:,}\n\n"
            
            if 'CUSTOMER_SEGMENT' in supply_df.columns:
                segment_dist = supply_df['CUSTOMER_SEGMENT'].value_counts()
                summary += "**Customer Segment Distribution**:\n"
                for segment, count in segment_dist.items():
                    percentage = (count / len(supply_df)) * 100
                    summary += f"- **{segment}**: {count:,} orders ({percentage:.1f}%)\n"
                summary += "\n"
            
            if 'CUSTOMER_COUNTRY' in supply_df.columns:
                country_dist = supply_df['CUSTOMER_COUNTRY'].value_counts().head(5)
                summary += "**Top 5 Countries by Order Volume**:\n"
                for country, count in country_dist.items():
                    summary += f"- **{country}**: {count:,} orders\n"
                summary += "\n"
            
            if 'CUSTOMER_CITY' in supply_df.columns:
                city_dist = supply_df['CUSTOMER_CITY'].value_counts().head(5)
                summary += "**Top 5 Cities by Order Volume**:\n"
                for city, count in city_dist.items():
                    summary += f"- **{city}**: {count:,} orders\n"
            
            summary += f"\n**Total Customer Records**: {len(supply_df):,}"
            
            return summary
        except Exception as e:
            logger.error(f"Error generating enhanced customer insights: {e}")
            return "Error calculating customer insights. Please check your data format."
    
    def generate_enhanced_data_quality_report(self, supply_df):
        """Enhanced data quality report"""
        if supply_df is None or supply_df.empty:
            return "No data available for quality analysis."
        
        try:
            
            total_records = len(supply_df)
            total_columns = len(supply_df.columns)
            missing_values = supply_df.isnull().sum().sum()
            duplicate_rows = supply_df.duplicated().sum()
            memory_mb = round(supply_df.memory_usage(deep=True).sum() / 1024**2, 2)
            
            
            numeric_cols = len(supply_df.select_dtypes(include=[np.number]).columns)
            text_cols = len(supply_df.select_dtypes(include=['object']).columns)
            date_cols = len(supply_df.select_dtypes(include=['datetime']).columns)
            
            summary = f"""
**Data Quality Audit Report**

**Dataset Overview**
- **Total Records**: {total_records:,}
- **Total Columns**: {total_columns}
- **Memory Usage**: {memory_mb} MB

**Data Quality Metrics**
- **Missing Values**: {missing_values:,} ({missing_values/(total_records*total_columns)*100:.1f}% of total cells)
- **Duplicate Rows**: {duplicate_rows:,} ({duplicate_rows/total_records*100:.1f}% of records)

**Column Type Distribution**
- **Numeric Columns**: {numeric_cols}
- **Text Columns**: {text_cols}  
- **Date Columns**: {date_cols}

"""
            
            
            missing_by_col = supply_df.isnull().sum()
            high_missing = missing_by_col[missing_by_col > total_records * 0.5]
            
            if len(high_missing) > 0:
                summary += f"**Columns with >50% Missing Data**: {', '.join([col.replace('_', ' ').title() for col in high_missing.index])}"
            else:
                summary += "**Data Completeness**: No columns with excessive missing values"
            
            return summary
        except Exception as e:
            logger.error(f"Error generating enhanced data quality report: {e}")
            return "Error calculating data quality report. Please check your data format."
    
    def generate_schema_summary(self, supply_df):
        """Schema summary"""
        if supply_df is None or supply_df.empty:
            return "No data available for schema analysis."
        
        try:
            
            numeric_cols = supply_df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = supply_df.select_dtypes(include=['object']).columns.tolist()
            date_cols = supply_df.select_dtypes(include=['datetime']).columns.tolist()
            
            summary = f"""
**Data Schema Analysis**

**Total Columns**: {len(supply_df.columns)}

"""
            
            if numeric_cols:
                summary += f"**Numeric Columns** ({len(numeric_cols)}):\n"
                summary += f"```\n{', '.join(numeric_cols[:10])}"
                if len(numeric_cols) > 10:
                    summary += f"\n... and {len(numeric_cols)-10} more"
                summary += "\n```\n\n"
            
            if text_cols:
                summary += f"**Text Columns** ({len(text_cols)}):\n"
                summary += f"```\n{', '.join(text_cols[:10])}"
                if len(text_cols) > 10:
                    summary += f"\n... and {len(text_cols)-10} more"
                summary += "\n```\n\n"
            
            if date_cols:
                summary += f"**Date Columns** ({len(date_cols)}):\n"
                summary += f"```\n{', '.join(date_cols)}\n```"
            
            return summary
        except Exception as e:
            logger.error(f"Error generating schema summary: {e}")
            return "Error generating schema summary. Please check your data format."
    
    def generate_top_performers_summary(self, supply_df):
        """Top performers summary"""
        if supply_df is None or supply_df.empty:
            return "No data available for top performers analysis."
        
        try:
            summary = """
**Top Performers Analysis**

"""
            
            # Top customers by revenue 
            if 'CUSTOMER_ID' in supply_df.columns and 'SALES' in supply_df.columns:
                customer_revenue = supply_df.groupby('CUSTOMER_ID')['SALES'].sum()
                if not customer_revenue.empty:
                    top_customers = customer_revenue.nlargest(5)
                    summary += f"**Top 5 Customers by Sales**:\n"
                    for i, (customer, revenue) in enumerate(top_customers.items(), 1):
                        summary += f"{i}. Customer {customer}: ${revenue:,.2f}\n"
                    summary += "\n"
                else:
                    summary += "No customer revenue data available\n\n"
            
            # Top products by revenue
            if 'PRODUCT_CARD_ID' in supply_df.columns and 'SALES' in supply_df.columns:
                product_revenue = supply_df.groupby('PRODUCT_CARD_ID')['SALES'].sum()
                if not product_revenue.empty:
                    top_products = product_revenue.nlargest(5)
                    summary += f"**Top 5 Products by Sales**:\n"
                    for i, (product, revenue) in enumerate(top_products.items(), 1):
                        summary += f"{i}. Product {product}: ${revenue:,.2f}\n"
                    summary += "\n"
                else:
                    summary += "No product revenue data available\n\n"
            
            # Top categories by revenue
            if 'CATEGORY_NAME' in supply_df.columns and 'SALES' in supply_df.columns:
                category_revenue = supply_df.groupby('CATEGORY_NAME')['SALES'].sum()
                if not category_revenue.empty:
                    top_categories = category_revenue.nlargest(5)
                    summary += f"**Top 5 Categories by Sales**:\n"
                    for i, (category, revenue) in enumerate(top_categories.items(), 1):
                        summary += f"{i}. {category}: ${revenue:,.2f}\n"
            
            return summary
        except Exception as e:
            logger.error(f"Error generating top performers summary: {e}")
            return "Error calculating top performers summary. Please check your data format."
    
    def generate_ml_predictions_summary(self, supply_df):
        """ML predictions summary"""
        if supply_df is None or supply_df.empty:
            return "No data available for ML predictions."
        
        summary = """
**Machine Learning Predictions Dashboard**

"""
        
        model_count = 0
        
        # Prophet Forecasting 
        if hasattr(self, 'forecast_model') and self.forecast_model:
            model_count += 1
            if hasattr(self, 'forecast_data') and self.forecast_data is not None:
                latest_forecast = self.forecast_data.iloc[-1]
                summary += f"**Time Series Forecast**\n"
                summary += f"- **Next Period Prediction**: {latest_forecast['yhat']:.2f}\n"
                summary += f"- **Confidence Range**: {latest_forecast['yhat_lower']:.2f} - {latest_forecast['yhat_upper']:.2f}\n"
                
                
                if hasattr(self, 'forecast_metrics') and self.forecast_metrics:
                    metrics = self.forecast_metrics
                    summary += f"- **MAE**: {metrics['MAE']:.2f}\n"
                    summary += f"- **RMSE**: {metrics['RMSE']:.2f}\n"
                    summary += f"- **MAPE**: {metrics['MAPE']:.1f}%\n"
                
                summary += "\n"
        
        # Churn Prediction 
        if self.churn_predictor:
            model_count += 1
            summary += f"**Churn/Risk Prediction Model**\n"
            summary += f"- **Training Accuracy**: {self.churn_predictor['train_score']:.1%}\n"
            summary += f"- **Test Accuracy**: {self.churn_predictor['test_score']:.1%}\n"
            summary += f"- **Features Used**: {len(self.churn_predictor['features'])}\n\n"
        
        # Anomaly Detection
        if self.anomaly_detector:
            model_count += 1
            summary += f"**Anomaly Detection**\n"
            summary += f"- **Anomaly Rate**: {self.anomaly_detector['anomaly_rate']:.1%}\n"
            summary += f"- **Detection Features**: {len(self.anomaly_detector['features'])}\n"
            
            if self.anomaly_results:
                summary += f"- **Anomalies Found**: {self.anomaly_results['anomaly_count']}\n"
            
            summary += "\n"
        
        if model_count == 0:
            summary += "**No ML models trained yet**\n\nPlease visit the ML/DL Models tab to train forecasting, churn prediction, and anomaly detection models."
        else:
            summary += f"**Active Models**: {model_count}/3 trained and ready"
        
        return summary
    
    # Dashboard
    
    
    def create_streamlit_dashboard(self):
        """Dashboard with visualizations"""
        st.set_page_config(
            page_title="Unified Intelligence Platform",
            #page_icon="📊",
            layout="wide"
        )
        
        st.title("Unified Intelligence Platform")
        st.markdown("Advanced Supply Chain & Customer Analytics with AI/ML")
        
        # Status bar at the top
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.snowflake_conn:
                st.success("Snowflake Connected")
            else:
                st.error("Snowflake Disconnected")
        
        with col2:
            if self.data_loaded:
                st.success(f"Data: {len(self.supply_df):,} records")
            else:
                st.warning("No Data Loaded")
        
        with col3:
            model_count = sum([1 for model in [
                getattr(self, 'forecast_model', None), 
                self.churn_predictor, 
                self.anomaly_detector
            ] if model])
            if model_count > 0:
                st.success(f"ML Models: {model_count}/3")
            else:
                st.info("ML Models: 0/3")
        
        with col4:
            if self.ai_enabled:
                st.success("AI Assistant Ready")
            else:
                st.warning("AI Assistant Limited")
        
        
        if self.connection_error:
            st.error(f"Connection Error: {self.connection_error}")
            return
        
        if not self.data_loaded:
            st.warning("No data available. Please check your Snowflake connection and ensure data is loaded.")
            return
        
        
        self.create_enhanced_unified_dashboard_content()
    
    def create_enhanced_unified_dashboard_content(self):
        """Dashboard with visualizations"""
        supply_df = self.supply_df
        access_df = self.access_df
        
        
        st.header("Key Business Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'SALES' in supply_df.columns:
                total_sales = supply_df['SALES'].fillna(0).sum()
                st.metric("Total Sales", f"${total_sales:,.2f}")
            else:
                st.metric("Total Sales", "N/A")
        
        with col2:
            if 'CUSTOMER_ID' in supply_df.columns:
                total_customers = supply_df['CUSTOMER_ID'].nunique()
                st.metric("Unique Customers", f"{total_customers:,}")
            else:
                st.metric("Unique Customers", "N/A")
        
        with col3:
            if 'DAYS_FOR_SHIPPING_REAL' in supply_df.columns:
                avg_days = supply_df['DAYS_FOR_SHIPPING_REAL'].fillna(0).mean()
                st.metric("Avg Shipping Days", f"{avg_days:.1f}")
            else:
                st.metric("Avg Shipping Days", "N/A")
        
        with col4:
            if 'LATE_DELIVERY_RISK' in supply_df.columns:
                risk_rate = supply_df['LATE_DELIVERY_RISK'].mean()
                st.metric("Late Delivery Risk", f"{risk_rate:.1%}")
            else:
                st.metric("Late Delivery Risk", "N/A")
        
        
        st.header("Enhanced Machine Learning Models")
        
        if not ML_AVAILABLE:
            st.error("Machine Learning libraries not available. Install with: pip install scikit-learn")
        else:
            # Model training buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Train Prophet Forecast"):
                    if PROPHET_AVAILABLE:
                        with st.spinner("Training Prophet model..."):
                            model_result = self.train_prophet_forecast_model(supply_df)
                            if model_result:
                                st.success("Prophet model trained!")
                                st.rerun()
                            else:
                                st.error("Failed - requires ORDER_DATE and ORDER_ITEM_QUANTITY")
                    else:
                        st.error("Prophet not available")
            
            with col2:
                if st.button("Train Churn Prediction"):
                    with st.spinner("Training churn model..."):
                        model_result = self.train_churn_prediction_model(supply_df)
                        if model_result:
                            st.success("Churn model trained!")
                            st.rerun()
                        else:
                            st.error("Failed - check required columns")
            
            with col3:
                if st.button("Train Anomaly Detection"):
                    with st.spinner("Training anomaly model..."):
                        if access_df is not None:
                            model_result = self.train_anomaly_detection_model(access_df, supply_df)
                            if model_result:
                                st.success("Anomaly model trained!")
                                st.rerun()
                            else:
                                st.error("Failed - check access log columns")
                        else:
                            model_result = self.train_anomaly_detection_model(None, supply_df)
                            if model_result:
                                st.success("Anomaly model trained!")
                                st.rerun()
                            else:
                                st.error("Failed - check data columns")
            
            
            st.subheader("Machine Learning Results & Visualizations")
            
            # Time Series Forecasting Results
            if hasattr(self, 'forecast_model') and self.forecast_model and self.forecast_data is not None:
                st.subheader("Time Series Forecast Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Create and display forecast visualization
                    forecast_fig = self.create_forecast_visualization()
                    if forecast_fig:
                        st.plotly_chart(forecast_fig, use_container_width=True)
                    else:
                        st.error("Could not create forecast visualization")
                
                with col2:
                    st.write("**Forecast Metrics**")
                    if hasattr(self, 'forecast_metrics') and self.forecast_metrics:
                        metrics = self.forecast_metrics
                        st.metric("MAE (Mean Absolute Error)", f"{metrics['MAE']:.2f}")
                        st.metric("RMSE (Root Mean Square Error)", f"{metrics['RMSE']:.2f}")
                        st.metric("MAPE (Mean Absolute % Error)", f"{metrics['MAPE']:.1f}%")
                        st.metric("Test Data Points", f"{metrics['test_points']}")
                    else:
                        st.info("Forecast metrics not available")
                    
                    # Latest forecast values
                    latest_forecast = self.forecast_data.iloc[-1]
                    st.write("**Next Period Forecast**")
                    st.write(f"Predicted Value: {latest_forecast['yhat']:.2f}")
                    st.write(f"Lower Bound: {latest_forecast['yhat_lower']:.2f}")
                    st.write(f"Upper Bound: {latest_forecast['yhat_upper']:.2f}")
            
            # Churn Prediction Results
            if self.churn_predictor and self.churn_results:
                st.subheader("Churn Prediction Model Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    
                    confusion_fig, importance_fig = self.create_churn_model_visualization()
                    if confusion_fig:
                        st.plotly_chart(confusion_fig, use_container_width=True)
                    
                    
                    st.write("**Classification Report**")
                    class_report = self.churn_results['classification_report']
                    
                    
                    report_df = pd.DataFrame(class_report).transpose()
                    
                    if 'macro avg' in report_df.index:
                        report_df.loc['macro avg', 'support'] = None
                    if 'weighted avg' in report_df.index:
                        report_df.loc['weighted avg', 'support'] = None
                    
                    st.dataframe(report_df.round(3), use_container_width=True)
                
                with col2:
                    
                    if importance_fig:
                        st.plotly_chart(importance_fig, use_container_width=True)
                    
                    
                    st.write("**Model Performance**")
                    st.metric("Training Accuracy", f"{self.churn_predictor['train_score']:.1%}")
                    st.metric("Test Accuracy", f"{self.churn_predictor['test_score']:.1%}")
                    st.metric("Features Used", len(self.churn_predictor['features']))
                    
                    st.write("**Feature Names:**")
                    for feature in self.churn_predictor['features']:
                        st.write(f"- {feature}")
            
            
            if self.anomaly_detector and self.anomaly_results:
                st.subheader("Anomaly Detection Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    
                    anomaly_fig = self.create_anomaly_visualization()
                    if anomaly_fig:
                        st.plotly_chart(anomaly_fig, use_container_width=True)
                    else:
                        st.warning("Could not create anomaly visualization - insufficient numeric columns")
                
                with col2:
                    st.write("**Anomaly Detection Summary**")
                    st.metric("Total Records Analyzed", self.anomaly_results['total_records'])
                    st.metric("Anomalies Detected", self.anomaly_results['anomaly_count'])
                    st.metric("Anomaly Rate", f"{self.anomaly_detector['anomaly_rate']:.1%}")
                    st.metric("Features Used", len(self.anomaly_detector['features']))
                
                
                if 'anomaly_data' in self.anomaly_results:
                    st.write("**Detected Anomalies (Top 10 by Anomaly Score)**")
                    anomaly_df = self.anomaly_results['anomaly_data']
                    anomalies_only = anomaly_df[anomaly_df['is_anomaly']].head(10)
                    
                    if not anomalies_only.empty:
                        
                        display_cols = [col for col in anomalies_only.columns if col not in ['is_anomaly']][:6]
                        st.dataframe(anomalies_only[display_cols], use_container_width=True)
                    else:
                        st.info("No anomalies detected in the current dataset")
        
        
        
        if self.knowledge_graph.number_of_nodes() > 0:
            st.header("Knowledge Graph Visualization")
            
            
            with st.expander("Debugging Information"):
                st.write(f"Nodes: {self.knowledge_graph.number_of_nodes()}")
                st.write(f"Edges: {self.knowledge_graph.number_of_edges()}")
                
                
                node_types = {}
                for node, data in self.knowledge_graph.nodes(data=True):
                    node_type = data.get('type', 'unknown')
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                st.write("Node types:", node_types)
                
                
                sample_nodes = list(self.knowledge_graph.nodes())[:10]
                st.write("Sample nodes:", sample_nodes)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                
                try:
                    kg_fig = self.create_knowledge_graph_visualization()
                    if kg_fig:
                        st.plotly_chart(kg_fig, use_container_width=True)
                    else:
                        st.error("Could not create knowledge graph visualization - check debug info above")
                except Exception as viz_error:
                    st.error(f"Visualization error: {viz_error}")
            
            with col2:
                st.metric("Graph Nodes", f"{self.knowledge_graph.number_of_nodes():,}")
                st.metric("Graph Edges", f"{self.knowledge_graph.number_of_edges():,}")
                
                
                st.write("**Node Types:**")
                for node_type, count in node_types.items():
                    st.write(f"- {node_type.title()}: {count:,}")
        
    

        
        
        st.header("Business Analytics")
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            if 'ORDER_DATE' in supply_df.columns and 'SALES' in supply_df.columns:
                try:
                    df_viz = supply_df.copy()
                    if df_viz['ORDER_DATE'].dtype == 'object':
                        df_viz['ORDER_DATE'] = pd.to_datetime(df_viz['ORDER_DATE'], errors='coerce')
                    
                    df_viz = df_viz.dropna(subset=['ORDER_DATE'])
                    
                    if not df_viz.empty and len(df_viz) > 1:
                        daily_data = df_viz.groupby(df_viz['ORDER_DATE'].dt.date)['SALES'].sum().reset_index()
                        daily_data.columns = ['Date', 'Sales']
                        
                        if len(daily_data) > 1:
                            fig_trend = px.line(daily_data, x='Date', y='Sales', title="Daily Sales Trend")
                            st.plotly_chart(fig_trend, use_container_width=True)
                        else:
                            st.info("Insufficient date range for trend analysis")
                    else:
                        st.info("No valid date data for time series")
                except Exception as e:
                    st.warning(f"Time series error: {e}")
            else:
                st.info("Time series requires ORDER_DATE and SALES columns")
        
        with col2:
            
            if 'CATEGORY_NAME' in supply_df.columns and 'SALES' in supply_df.columns:
                try:
                    category_data = supply_df.groupby('CATEGORY_NAME')['SALES'].sum().reset_index()
                    if not category_data.empty:
                        fig_cat = px.pie(category_data, values='SALES', names='CATEGORY_NAME', 
                                       title="Sales by Category")
                        st.plotly_chart(fig_cat, use_container_width=True)
                    else:
                        st.info("No category data available")
                except Exception as e:
                    st.warning(f"Category analysis error: {e}")
            else:
                st.info("Category analysis requires CATEGORY_NAME and SALES columns")
        
        
        if 'SHIPPING_MODE' in supply_df.columns and 'LATE_DELIVERY_RISK' in supply_df.columns:
            st.subheader("🚚 Shipping Performance")
            
            shipping_performance = supply_df.groupby('SHIPPING_MODE').agg({
                'LATE_DELIVERY_RISK': ['mean', 'count'],
                'SALES': 'sum' if 'SALES' in supply_df.columns else 'count'
            }).round(3)
            
            shipping_performance.columns = ['Late_Delivery_Rate', 'Total_Orders', 'Total_Sales']
            shipping_performance = shipping_performance.reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_delivery = px.bar(shipping_performance, x='SHIPPING_MODE', y='Late_Delivery_Rate',
                                    title="Late Delivery Rate by Shipping Mode")
                st.plotly_chart(fig_delivery, use_container_width=True)
            
            with col2:
                fig_orders = px.bar(shipping_performance, x='SHIPPING_MODE', y='Total_Orders',
                                  title="Order Volume by Shipping Mode")
                st.plotly_chart(fig_orders, use_container_width=True)
        
        
        st.header("AI Assistant")
        
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Revenue Analysis", use_container_width=True):
                summary = self.generate_enhanced_revenue_summary(supply_df)
                with st.expander("Revenue Analysis Results", expanded=True):
                    st.markdown(summary)
        
        with col2:
            if st.button("Customer Intelligence", use_container_width=True):
                insights = self.generate_enhanced_customer_insights(supply_df)
                with st.expander("Customer Intelligence Results", expanded=True):
                    st.markdown(insights)
        
        with col3:
            if st.button("Data Quality Audit", use_container_width=True):
                quality = self.generate_enhanced_data_quality_report(supply_df)
                with st.expander("Data Quality Results", expanded=True):
                    st.markdown(quality)
        
        with col4:
            if st.button("Supply Chain Health", use_container_width=True):
                health = self.generate_supply_chain_health(supply_df)
                with st.expander("Supply Chain Health Results", expanded=True):
                    st.markdown(health)
        

        st.subheader("Ask Questions About Your Data")
        
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        
        recent_messages = st.session_state.messages[-4:] if len(st.session_state.messages) > 4 else st.session_state.messages
        
        for message in recent_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        
        if prompt := st.chat_input("Ask about revenue, customers, shipping, data quality, top performers..."):
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    if hasattr(self, 'query_patterns') and hasattr(self, 'enhanced_query_processor'):
                        response = self.enhanced_query_processor(prompt, supply_df, access_df)
                    else:
                        response = self.generate_enhanced_ai_response(prompt, supply_df, access_df)
                    st.markdown(response)
            
            
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        
        st.header("🔬 Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RFM Analysis
            st.subheader("Customer RFM Segmentation")
            if st.button("Generate RFM Analysis"):
                with st.spinner("Calculating RFM scores..."):
                    rfm_data = self.calculate_rfm_analysis(supply_df)
                    if rfm_data is not None:
                        st.success("RFM Analysis completed!")
                        
                        # RFM Distribution
                        segment_counts = rfm_data['Customer_Segment'].value_counts()
                        fig_segments = px.pie(values=segment_counts.values, names=segment_counts.index,
                                            title="Customer Segments")
                        st.plotly_chart(fig_segments, use_container_width=True)
                        
                        # RFM Summary table
                        segment_summary = rfm_data.groupby('Customer_Segment').agg({
                            'Recency': 'mean',
                            'Frequency': 'mean', 
                            'Monetary': 'mean',
                            'CUSTOMER_ID': 'count'
                        }).round(2).reset_index()
                        segment_summary.columns = ['Segment', 'Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Count']
                        st.dataframe(segment_summary, use_container_width=True)
                    else:
                        st.error("RFM analysis failed - requires CUSTOMER_ID, ORDER_ID, ORDER_DATE, SALES")
        
        with col2:
            # Cross-functional analysis
            st.subheader("Cross-functional Impact")
            delay_impact, segment_shipping = self.analyze_cross_functional_impact(supply_df)
            
            if delay_impact is not None and not delay_impact.empty:
                st.write("**Shipping Mode Performance**")
                st.dataframe(delay_impact, use_container_width=True)
                
                if segment_shipping is not None and not segment_shipping.empty:
                    st.write("**Segment vs Shipping Performance**")
                    st.dataframe(segment_shipping.head(10), use_container_width=True)
            else:
                st.info("Cross-functional analysis requires SHIPPING_MODE and LATE_DELIVERY_RISK columns")
        
        
        st.header("Data Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Schema Info")
            numeric_cols = len(supply_df.select_dtypes(include=[np.number]).columns)
            text_cols = len(supply_df.select_dtypes(include=['object']).columns)
            date_cols = len(supply_df.select_dtypes(include=['datetime']).columns)
            
            st.write(f"**Total Columns**: {len(supply_df.columns)}")
            st.write(f"**Numeric**: {numeric_cols}")
            st.write(f"**Text**: {text_cols}")
            st.write(f"**Date**: {date_cols}")
        
        with col2:
            st.subheader("Data Quality")
            missing_values = supply_df.isnull().sum().sum()
            duplicate_rows = supply_df.duplicated().sum()
            total_cells = len(supply_df) * len(supply_df.columns)
            
            st.write(f"**Missing Values**: {missing_values:,}")
            st.write(f"**Missing %**: {missing_values/total_cells*100:.1f}%")
            st.write(f"**Duplicates**: {duplicate_rows:,}")
            st.write(f"**Memory**: {supply_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        with col3:
            st.subheader("Top Performers")
            if 'CATEGORY_NAME' in supply_df.columns and 'SALES' in supply_df.columns:
                top_categories = supply_df.groupby('CATEGORY_NAME')['SALES'].sum().nlargest(3)
                st.write("**Top 3 Categories**:")
                for i, (category, sales) in enumerate(top_categories.items(), 1):
                    st.write(f"{i}. {category}: ${sales:,.0f}")
            else:
                st.info("Top performers require CATEGORY_NAME and SALES")
        
        
        st.markdown("---")
        st.markdown("**Platform Status**: All systems operational | **Data Source**: Snowflake | **ML Framework**: Scikit-learn, Prophet")



# STREAMLIT setup 


@st.cache_resource
def get_platform():
    """Platform instance """
    return UnifiedIntelligencePlatform()


def main():
    """Main function to run the Intelligence Platform"""
    try:
        # Getting the cached platform instance
        platform = get_platform()
        
        # Creating and running the dashboard
        platform.create_streamlit_dashboard()
        
    except Exception as e:
        st.error(f"Error initializing platform: {e}")
        logger.error(f"Platform initialization error: {e}")
        
        # Additional debugging information
        st.subheader("Debug Information")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()                                            