import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import optuna
import warnings
warnings.filterwarnings('ignore')

optuna.logging.set_verbosity(optuna.logging.INFO)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeFiVolumePredictor:
    """A comprehensive class for predicting volumes with semantic features and Optuna optimization."""

    def __init__(self, use_semantic_features=True):
        self.use_semantic = use_semantic_features
        self.scalers = {}
        self.models = {}
        self.results = {}
        self.best_params = {}
        self.data_for_opt = {}  # To store data for Optuna

    # --- Feature Engineering (No Changes) ---
    def create_semantic_features(self, df):
        """Create semantic features based on an ontology."""
        df = df.copy()
        df['volume_ratio'] = df['altcoin_volume_24h'] / (df['total_volume_24h'] + 1e-8)
        df['btc_share'] = df['btc_dominance'] / 100
        df['LCR'] = df['volume_ratio'] ** 2 * np.exp(-df['active_exchanges'] / 100)
        df['vol_to_cap'] = df['total_volume_24h'] / (df['total_market_cap'] + 1e-8)
        df['pair_token_ratio'] = df['active_market_pairs'] / (df['active_cryptocurrencies'] + 1)
        df['MDP'] = df['vol_to_cap'] * np.sqrt(df['pair_token_ratio'])
        df['vol_cv'] = df['vol_std_12'] / (df['vol_mean_12'] + 1e-8)
        df['sentiment_deviation'] = np.abs(df['value'] - 50) / 50
        df['SVA'] = df['sentiment_deviation'] * df['vol_cv']
        df['PCI'] = np.log(df['active_cryptocurrencies'] + 1) * df['pair_token_ratio']
        df['DAR'] = df['volume_ratio'] * (1 - df['btc_share'])
        df['relative_vol'] = df['alt_std_12'] / (df['vol_std_12'] + 1e-8)
        df['ILFI'] = df['relative_vol'] * df['volume_ratio']
        df['CME'] = np.log(df['active_exchanges'] + 1) * (1 - df['volume_ratio'] ** 2)
        df['alt_to_cap'] = df['altcoin_volume_24h'] / (df['altcoin_market_cap'] + 1e-8)
        df['LV'] = (df['vol_to_cap'] + df['alt_to_cap']) / 2
        semantic_cols = ['LCR', 'MDP', 'SVA', 'PCI', 'DAR', 'ILFI', 'CME', 'LV']
        for col in semantic_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].median())
        return df, semantic_cols

    def create_lag_features(self, df, target_col='log_total_volume_24h'):
        """Create correct lag features WITHOUT data leakage."""
        df = df.copy()
        lag_cols = []
        for lag in [1, 3, 6, 12, 24, 48]:
            col_name = f'{target_col}_lag_{lag}'
            df[col_name] = df[target_col].shift(lag)
            lag_cols.append(col_name)
        for window in [6, 12, 24, 48]:
            ma_col = f'{target_col}_ma_{window}'
            std_col = f'{target_col}_std_{window}'
            df[ma_col] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[std_col] = df[target_col].rolling(window=window, min_periods=1).std()
            lag_cols.extend([ma_col, std_col])
        return df, lag_cols

    def prepare_data(self, df, horizon=48, validation_split_ratio=0.15):
        """Prepare data and create a validation set for Optuna."""
        df, semantic_cols = self.create_semantic_features(df)
        df, lag_cols = self.create_lag_features(df)
        base_features = [
            'log_total_cap', 'log_altcoin_cap', 'btc_dominance', 'active_exchanges',
            'log_market_pairs', 'zscore_12', 'vol_mean_12', 'vol_std_12',
            'vol_mean_24', 'vol_std_24'
        ]
        features = base_features + lag_cols + (semantic_cols if self.use_semantic else [])

        for h in range(1, horizon + 1):
            df[f'target_{h}'] = df['log_total_volume_24h'].shift(-h)

        df_clean = df.dropna()
        X = df_clean[features].values
        y = df_clean[[f'target_{h}' for h in range(1, horizon + 1)]].values

        split_idx = int(0.8 * len(X))
        X_train_full, X_test = X[:split_idx], X[split_idx:]
        y_train_full, y_test = y[:split_idx], y[split_idx:]

        self.scalers['X'] = RobustScaler()
        self.scalers['y'] = StandardScaler()

        X_train_full_scaled = self.scalers['X'].fit_transform(X_train_full)
        X_test_scaled = self.scalers['X'].transform(X_test)
        y_train_full_scaled = self.scalers['y'].fit_transform(y_train_full)
        y_test_scaled = self.scalers['y'].transform(y_test)

        val_split_idx = int(len(X_train_full_scaled) * (1 - validation_split_ratio))
        self.data_for_opt = {
            'X_train': X_train_full_scaled[:val_split_idx],
            'y_train': y_train_full_scaled[:val_split_idx],
            'X_val': X_train_full_scaled[val_split_idx:],
            'y_val': y_train_full_scaled[val_split_idx:]
        }

        return X_train_full_scaled, X_test_scaled, y_train_full_scaled, y_test_scaled, features

    # --- Optuna Objective Functions ---
    def _objective_rf(self, trial):
        """Objective function for Random Forest optimization."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        from sklearn.multioutput import MultiOutputRegressor
        model = MultiOutputRegressor(rf)

        model.fit(self.data_for_opt['X_train'], self.data_for_opt['y_train'])
        y_pred = model.predict(self.data_for_opt['X_val'])

        mae = mean_absolute_error(self.data_for_opt['y_val'], y_pred)
        return mae

    def _objective_rnn(self, trial, model_type='lstm'):
        """Objective function for LSTM/GRU optimization."""
        seq_len = trial.suggest_categorical('sequence_length', [24, 48, 72])

        X_train_seq, y_train_seq = self._create_sequences(self.data_for_opt['X_train'], self.data_for_opt['y_train'],
                                                          seq_len)
        X_val_seq, y_val_seq = self._create_sequences(self.data_for_opt['X_val'], self.data_for_opt['y_val'], seq_len)

        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            raise optuna.exceptions.TrialPruned()

        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'n_units_1': trial.suggest_categorical('n_units_1', [64, 128, 256]),
            'n_units_2': trial.suggest_categorical('n_units_2', [32, 64, 128]),
        }

        RNN_LAYER = LSTM if model_type == 'lstm' else GRU

        model = Sequential([
            RNN_LAYER(params['n_units_1'], return_sequences=True, input_shape=(seq_len, X_train_seq.shape[2])),
            BatchNormalization(),
            Dropout(params['dropout_rate']),
            RNN_LAYER(params['n_units_2']),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dropout(params['dropout_rate']),
            Dense(y_train_seq.shape[1])
        ])

        model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='huber', metrics=['mae'])

        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=50,
            batch_size=32,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )

        val_mae = min(history.history['val_mae'])
        return val_mae

    # --- Main Optimization Runners ---
    def optimize_model(self, model_name, n_trials=30):
        """Runs Optuna study for a given model with debugging."""
        print(f"\nOptimizing {model_name.upper()}...")

        def print_callback(study, trial):
            print(f"Trial {trial.number} for {model_name.upper()} finished with value: {trial.value:.5f}")

        study = optuna.create_study(direction='minimize')

        if model_name == 'rf':
            study.optimize(self._objective_rf, n_trials=n_trials, callbacks=[print_callback])
        elif model_name in ['lstm', 'gru']:
            objective_func = lambda trial: self._objective_rnn(trial, model_type=model_name)
            study.optimize(objective_func, n_trials=n_trials, callbacks=[print_callback])
        else:
            raise ValueError("Unknown model name")

        self.best_params[model_name] = study.best_params
        print(f"\nOptimization for {model_name.upper()} complete.")
        print(f"Best validation MAE: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        self.visualize_study(study, model_name)

        return study.best_params

    # --- Updated Training Functions ---
    def train_model(self, model_name, X_train, y_train, X_test, y_test):
        """A unified function to train any model with optimized parameters."""
        print(f"\n=== Training {model_name.upper()} with best parameters ===")
        params = self.best_params.get(model_name, {})

        if model_name == 'rf':
            rf = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
            from sklearn.multioutput import MultiOutputRegressor
            model = MultiOutputRegressor(rf)
            model.fit(X_train, y_train)
            y_pred_scaled = model.predict(X_test)
            y_test_scaled = y_test

        elif model_name in ['lstm', 'gru']:
            seq_len = params.get('sequence_length', 48)
            X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, seq_len)
            X_test_seq, y_test_seq = self._create_sequences(X_test, y_test, seq_len)

            RNN_LAYER = LSTM if model_name == 'lstm' else GRU
            model = Sequential([
                RNN_LAYER(params.get('n_units_1', 128), return_sequences=True,
                          input_shape=(seq_len, X_train_seq.shape[2])),
                BatchNormalization(),
                Dropout(params.get('dropout_rate', 0.2)),
                RNN_LAYER(params.get('n_units_2', 64)),
                BatchNormalization(),
                Dense(64, activation='relu'),
                Dropout(params.get('dropout_rate', 0.2)),
                Dense(y_train_seq.shape[1])
            ])
            model.compile(optimizer=Adam(learning_rate=params.get('learning_rate', 0.001)), loss='huber',
                          metrics=['mae'])

            model.fit(
                X_train_seq, y_train_seq, epochs=100, batch_size=32, verbose=0,
                callbacks=[
                    EarlyStopping(patience=15, restore_best_weights=True),
                    ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            y_pred_scaled = model.predict(X_test_seq, verbose=0)
            y_test_scaled = y_test_seq
        else:
            raise ValueError("Unknown model name")

        y_pred = self.scalers['y'].inverse_transform(y_pred_scaled)
        y_test_orig = self.scalers['y'].inverse_transform(y_test_scaled)

        mae = mean_absolute_error(y_test_orig, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
        r2 = r2_score(y_test_orig, y_pred)

        print(f"Final Test Metrics -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        self.models[model_name] = model
        self.results[model_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        return model

    # --- Helper and Visualization ---
    def _create_sequences(self, X, y, seq_length):
        Xs, ys = [], []
        if len(X) <= seq_length: return np.array(Xs), np.array(ys)
        for i in range(seq_length, len(X)):
            Xs.append(X[i - seq_length:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def visualize_study(self, study, model_name):
        """Visualizes the Optuna study results."""
        try:
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.update_layout(title_text=f'Optimization History for {model_name.upper()}')
            fig1.show()

            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.update_layout(title_text=f'Hyperparameter Importance for {model_name.upper()}')
            fig2.show()

            fig3 = optuna.visualization.plot_slice(study)
            fig3.update_layout(title_text=f'Parameter Slices for {model_name.upper()}')
            fig3.show()

        except (ImportError, ValueError) as e:
            print(f"Could not generate plots for {model_name}: {e}. Please ensure Plotly is installed.")