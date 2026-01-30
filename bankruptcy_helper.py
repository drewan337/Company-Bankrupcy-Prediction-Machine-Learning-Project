import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

class Helper():
    def __init__(self):
        return
    
# COMPARE FEATURE RELATED TO BANKRUPCY AND NO BANKRUPCY
    def plot_feature_distributions_by_bankruptcy(self, data, features=None, target_col='Bankrupt', figsize=(15, 10), bins=50, save_path=None):
        """Plot overlapping histograms comparing bankrupt vs non-bankrupt"""
        import matplotlib.pyplot as plt
        
        if target_col not in data.columns:
            print(f"Error: '{target_col}' not found")
            return
        
        # Use the outlier features
        if features is None:
            features = ['X10','X2', 'X38', 'X3', 'X51', 'X25', 'X35', 'X42']
        
        # Filter to only valid features that exist in data
        valid_features = []
        for f in features:
            if f in data.columns and data[f].notna().sum() > 0:
                valid_features.append(f)
        
        features = valid_features
        
        if len(features) == 0:
            print("Error: No valid features with data")
            return
        
        print(f"Plotting {len(features)} features: {features}")
        
        bankrupt = data[data[target_col] == 1]
        non_bankrupt = data[data[target_col] == 0]
        
        print(f"  Bankrupt: {len(bankrupt)} samples")
        print(f"  Non-bankrupt: {len(non_bankrupt)} samples")
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            bankrupt_values = bankrupt[feature].dropna()
            non_bankrupt_values = non_bankrupt[feature].dropna()
            
            # Check if we have data
            if len(bankrupt_values) == 0 or len(non_bankrupt_values) == 0:
                ax.text(0.5, 0.5, f'No data\nfor {feature}', 
                    ha='center', va='center', fontsize=10)
                ax.set_title(feature)
                continue
            
            ax.hist(non_bankrupt_values, bins=bins, alpha=0.6, 
                label='Not Bankrupt', color='green', edgecolor='black', density=True)
            ax.hist(bankrupt_values, bins=bins, alpha=0.6, 
                label='Bankrupt', color='red', edgecolor='black', density=True)
            
            ax.axvline(non_bankrupt_values.mean(), color='darkgreen', 
                    linestyle='--', linewidth=2)
            ax.axvline(bankrupt_values.mean(), color='darkred', 
                    linestyle='--', linewidth=2)
            
            ax.set_xlabel('Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        
        for idx in range(len(features), len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle('Feature Distributions: Bankrupt vs Non-Bankrupt (Outlier Features)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to '{save_path}'")
        else:
            plt.show()
        
# CORRELATION
    def plot_top_features_by_correlation(self, df, target_col='Bankrupt', top_n=6, n_cols=3):
        """
        Plot histogram overlays for top features correlated with bankruptcy.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_col : str, default='Bankrupt'
            Target column name
        top_n : int, default=6
            Number of top features to plot
        n_cols : int, default=3
            Number of columns in grid
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found")
            return
        
        # Calculate correlations with target
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [f for f in numeric_features if f != target_col and f != 'Id']
        
        correlations = {}
        for feature in numeric_features:
            corr = df[feature].corr(df[target_col])
            correlations[feature] = abs(corr)  # Use absolute correlation
        
        # Get top features
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_feature_names = [feature for feature, corr in top_features]
        
        for feature, corr in top_features:
            print(f"   {feature}: {corr:.3f}")
        
        # Plot the top features
        self.plot_multiple_histogram_overlays(df, top_feature_names, target_col, n_cols)
    
# FOR IMPUTATION
    def impute_bankruptcy_data(self, data, drop_high_missing=True, missing_threshold=0.4):
        """
        Impute missing values in bankruptcy dataset
        
        Parameters:
        -----------
        data : DataFrame
            Input dataframe with all numeric data
        drop_high_missing : bool
            Whether to drop features with high missing percentage
        missing_threshold : float
            Threshold for dropping features (default: 0.4 = 40%)
        
        Returns:
        --------
        DataFrame with imputed values
        """
        from sklearn.impute import SimpleImputer
        data_imputed = data.copy()
        
        # Identify feature columns (exclude target and ID)
        exclude_cols = ['Bankrupt', 'Id']
        feature_cols = [col for col in data_imputed.columns if col not in exclude_cols]
        
        print(f"\nTotal features: {len(feature_cols)}")
        
        # Drop features with high missing percentage
        if drop_high_missing:
            print(f"\nStep 1: Dropping features with >{missing_threshold*100}% missing")
            
            missing_pct = (data_imputed[feature_cols].isnull().sum() / len(data_imputed))
            high_missing_features = missing_pct[missing_pct > missing_threshold].index.tolist()
            
            if high_missing_features:
                print(f"  Dropping {len(high_missing_features)} features:")
                for feat in high_missing_features:
                    pct = missing_pct[feat] * 100
                    print(f"    - {feat}: {pct:.1f}% missing")
                
                data_imputed = data_imputed.drop(columns=high_missing_features)
                feature_cols = [col for col in feature_cols if col not in high_missing_features]
            else:
                print(f"  No features with >{missing_threshold*100}% missing")
        
        # Categorize features by missing percentage
        print(f"\nStep 2: Categorizing features by missing percentage")
        
        missing_pct = (data_imputed[feature_cols].isnull().sum() / len(data_imputed)) * 100
        
        low_missing = missing_pct[(missing_pct > 0) & (missing_pct < 4)].index.tolist()
        high_missing = missing_pct[(missing_pct >= 4) & (missing_pct <= missing_threshold*100)].index.tolist()
        
        print(f"  Low missing (<4%): {len(low_missing)} features")
        print(f"  High missing (4-{missing_threshold*100}%): {len(high_missing)} features")
        
        # Simple imputation for low missing
        if low_missing:
            print(f"\nStep 3: Simple median imputation for {len(low_missing)} features")
            simple_imputer = SimpleImputer(strategy='mean')
            data_imputed[low_missing] = simple_imputer.fit_transform(data_imputed[low_missing])
            print(f"  Imputed {len(low_missing)} features")
        
        # Random Forest iterative imputation for high missing
        if high_missing:
            rf_estimator = RandomForestRegressor(
                n_estimators=15,
                max_depth=12,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            )
            
            iter_imputer = IterativeImputer(
                estimator=rf_estimator,
                max_iter=10,
                tol=1e-3,
                initial_strategy='median',
                random_state=42,
                verbose=1
            )
            
            # Fit on all features for context
            all_features_imputed = iter_imputer.fit_transform(data_imputed[feature_cols])
            
            # Only update the high_missing columns
            feature_indices = [feature_cols.index(col) for col in high_missing]
            data_imputed[high_missing] = all_features_imputed[:, feature_indices]
            
            print(f"Iterative imputation complete")
        
        # Validation
        remaining_missing = data_imputed[feature_cols].isnull().sum().sum()
        print(f"\nValidation: {remaining_missing} missing values remain")
        
        print("\nImputation complete!")
        return data_imputed


#FOR FEATURE ENGINEERING
    def calculate_altman_z_score(self, data, verbose=True):
        """
        Calculate Altman Z-Score for bankruptcy prediction
        
        Altman Z-Score Formula (Original - for manufacturing companies):
        Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        
        Where (Altman components):
        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets
        
        Mapped to given dataset:
        X1 → X3  (working capital / total assets)
        X2 → X6  (retained earnings / total assets)
        X3 → X7  (EBIT / total assets)
        X4 → X8  (book value of equity / total liabilities)
        X5 → X9  (sales / total assets)
        
        Parameters:
        -----------
        data : DataFrame
            Input dataframe with financial features (X1-X64)
        verbose : bool
            If True, print detailed analysis and statistics (default: True)
        
        Returns:
        --------
        DataFrame with Altman Z-Score features added
        """
        
        z_data = data.copy()
        
        try:
            # Altman Z-Score Components
            
            # Component 1: Working Capital / Total Assets
            # X3 = working capital / total assets 
            if 'X3' in z_data.columns:
                z_data['z1_working_capital_ratio'] = z_data['X3']

            
            # Component 2: Retained Earnings / Total Assets
            # X6 = retained earnings / total assets
            if 'X6' in z_data.columns:
                z_data['z2_retained_earnings_ratio'] = z_data['X6']

            # Component 3: EBIT / Total Assets
            # X7 = EBIT / total assets
            if 'X7' in z_data.columns:
                z_data['z3_ebit_ratio'] = z_data['X7']

            # Component 4: Book Value of Equity / Total Liabilities
            # X8 = book value of equity / total liabilities
            # Note: Altman uses Market Value, but Book Value seems to be a common substitute
            if 'X8' in z_data.columns:
                z_data['z4_equity_liability_ratio'] = z_data['X8']
                
            # Component 5: Sales / Total Assets
            # X9 = sales / total assets
            if 'X9' in z_data.columns:
                z_data['z5_asset_turnover'] = z_data['X9']
            

            # Calculate Altman Z-Score with original weights
            z_data['altman_z_score'] = (
                1.2 * z_data['z1_working_capital_ratio'] +      # X3
                1.4 * z_data['z2_retained_earnings_ratio'] +    # X6
                3.3 * z_data['z3_ebit_ratio'] +                 # X7
                0.6 * z_data['z4_equity_liability_ratio'] +     # X8
                1.0 * z_data['z5_asset_turnover']               # X9
            )

            if verbose:
                print(f"  Formula: Z = 1.2*X3 + 1.4*X6 + 3.3*X7 + 0.6*X8 + 1.0*X9")

            # Create categorical zones based on Altman thresholds
            z_data['z_score_zone'] = pd.cut(
                z_data['altman_z_score'],
                bins=[-np.inf, 1.81, 2.99, np.inf],
                labels=['distress', 'grey', 'safe']
            )
            
            # Binary flags for each zone
            z_data['z_distress_flag'] = (z_data['altman_z_score'] < 1.81).astype(int)
            z_data['z_grey_flag'] = (
                (z_data['altman_z_score'] >= 1.81) & 
                (z_data['altman_z_score'] < 2.99)
            ).astype(int)
            z_data['z_safe_flag'] = (z_data['altman_z_score'] >= 2.99).astype(int)
            

            # Additional useful features based on Z-Score

            # Distance from distress threshold
            z_data['z_distance_from_distress'] = z_data['altman_z_score'] - 1.81
            
            # Distance from safe threshold
            z_data['z_distance_from_safe'] = z_data['altman_z_score'] - 2.99
            
            # Severity of distress (for companies in distress zone)
            z_data['z_distress_severity'] = np.where(
                z_data['altman_z_score'] < 1.81,
                1.81 - z_data['altman_z_score'],  # How far below threshold
                0
            )

            # Summary statistics (only if verbose=True)
            if verbose:
                print("ALTMAN Z-SCORE SUMMARY")
                print(f"\nZ-Score Distribution:")
                print(f"  Mean:   {z_data['altman_z_score'].mean():.4f}")
                print(f"  Median: {z_data['altman_z_score'].median():.4f}")
                print(f"  Std:    {z_data['altman_z_score'].std():.4f}")
                print(f"  Min:    {z_data['altman_z_score'].min():.4f}")
                print(f"  Max:    {z_data['altman_z_score'].max():.4f}")
                print(f"  25%:    {z_data['altman_z_score'].quantile(0.25):.4f}")
                print(f"  75%:    {z_data['altman_z_score'].quantile(0.75):.4f}")
                
                print(f"\nZ-Score Zones:")
                zone_counts = z_data['z_score_zone'].value_counts()
                total = len(z_data)
                for zone in ['distress', 'grey', 'safe']:
                    if zone in zone_counts.index:
                        count = zone_counts[zone]
                        pct = count / total * 100
                        print(f"  {zone.capitalize():10s}: {count:5d} ({pct:5.1f}%)")
                
                # If Bankrupt column exists, show breakdown by bankruptcy status
                if 'Bankrupt' in z_data.columns:
                    print("BANKRUPTCY ANALYSIS")
                    print(f"\nZ-Score by Bankruptcy Status:")
                    for bankrupt_status in [0, 1]:
                        status_name = "Non-Bankrupt" if bankrupt_status == 0 else "Bankrupt"
                        mask = z_data['Bankrupt'] == bankrupt_status
                        if mask.sum() > 0:
                            mean_z = z_data.loc[mask, 'altman_z_score'].mean()
                            median_z = z_data.loc[mask, 'altman_z_score'].median()
                            std_z = z_data.loc[mask, 'altman_z_score'].std()
                            print(f"  {status_name:15s}: Mean={mean_z:6.3f}, Median={median_z:6.3f}, Std={std_z:6.3f}")
                    
                    # Zone breakdown by bankruptcy
                    print(f"\nBankruptcy Rate by Z-Score Zone:")
                    for zone in ['distress', 'grey', 'safe']:
                        zone_mask = z_data['z_score_zone'] == zone
                        if zone_mask.sum() > 0:
                            total_in_zone = zone_mask.sum()
                            bankrupt_in_zone = (zone_mask & (z_data['Bankrupt'] == 1)).sum()
                            bankruptcy_rate = bankrupt_in_zone / total_in_zone * 100
                            print(f"  {zone.capitalize():10s}: {bankrupt_in_zone:4d}/{total_in_zone:4d} bankrupt ({bankruptcy_rate:5.1f}%)")
                    
                    # Predictive power
                    print("Predictive power")
                    
                    # Determine how well does distress zone predict bankruptcy
                    distress_mask = z_data['z_distress_flag'] == 1
                    total_distress = distress_mask.sum()
                    bankrupt_in_distress = (distress_mask & (z_data['Bankrupt'] == 1)).sum()
                    
                    total_bankrupt = (z_data['Bankrupt'] == 1).sum()
                    bankrupt_caught_by_distress = bankrupt_in_distress / total_bankrupt * 100 if total_bankrupt > 0 else 0
                    
                    print(f"\nDistress Zone (Z < 1.81) Analysis:")
                    print(f"  Companies in distress: {total_distress}")
                    print(f"  Bankrupt companies in distress: {bankrupt_in_distress}")
                    if total_distress > 0:
                        precision = bankrupt_in_distress / total_distress * 100
                        print(f"  Precision: {precision:.1f}% (of distress predictions are correct)")
                    print(f"  Recall: {bankrupt_caught_by_distress:.1f}% (of bankruptcies caught by distress flag)")
                    
                    # Safe zone analysis
                    safe_mask = z_data['z_safe_flag'] == 1
                    total_safe = safe_mask.sum()
                    bankrupt_in_safe = (safe_mask & (z_data['Bankrupt'] == 1)).sum()
                    
                    if total_safe > 0:
                        print(f"\nSafe Zone (Z > 2.99) Analysis:")
                        print(f"  Companies in safe zone: {total_safe}")
                        print(f"  Bankrupt companies in safe zone: {bankrupt_in_safe}")
                        safety_rate = (1 - bankrupt_in_safe / total_safe) * 100 if total_safe > 0 else 0
                        print(f"  Safety rate: {safety_rate:.1f}% (correctly identified as safe)")
                

                print(f"Added {13} Altman Z-Score features")
                
                print("\nInterpretation:")
                print("  Z < 1.81:       High bankruptcy risk (Distress Zone)")
                print("  1.81 ≤ Z < 2.99: Moderate risk (Grey Zone)")
                print("  Z ≥ 2.99:       Low bankruptcy risk (Safe Zone)")
            
        except Exception as e:
            print(f"\n Error calculating Altman Z-Score: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        return z_data

    
# COMPARE WITH AND WITHOUT ALTMAN Z SCORE FEATURE
    def compare_xgboost_with_without_zscore(self, data, test_size=0.2, random_state=42):
        """
        Compare XGBoost performance WITH vs WITHOUT Altman Z-Score features
        
        Parameters:
        -----------
        data : DataFrame
            Input dataframe with all features and target
        test_size : float
            Proportion of data for testing (default: 0.2)
        random_state : int
            Random seed for reproducibility (default: 42)
        
        Returns:
        --------
        dict : Dictionary containing comparison results and trained models
        """

        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                                    recall_score, f1_score, classification_report, 
                                    confusion_matrix, roc_curve)
        from xgboost import XGBClassifier

        # Dataset 1: WITHOUT Altman Z-Score
        data_without_z = data.copy()
        
        # Drop non-feature columns
        exclude_cols = ['Bankrupt']
        X_without_z = data_without_z.drop(columns=[col for col in exclude_cols if col in data_without_z.columns])
        y = data_without_z['Bankrupt']
        
        # Dataset 2: WITH Altman Z-Score
        data_with_z = self.calculate_altman_z_score(data, verbose=False)
        
        # Drop non-feature columns and categorical zone
        drop_cols = [col for col in ['Bankrupt', 'z_score_zone'] if col in data_with_z.columns]
        X_with_z = data_with_z.drop(columns=drop_cols)

        # TRAIN-TEST SPLIT
        
        # Use same random state for fair comparison
        X_train_without, X_test_without, y_train, y_test = train_test_split(
            X_without_z, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_with, X_test_with, _, _ = train_test_split(
            X_with_z, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Train size: {X_train_without.shape[0]}")
        print(f"Test size: {X_test_without.shape[0]}")
        
        # MODEL PARAMETERS
        model_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': 5.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        
        # CROSS-VALIDATION

        # Create fresh model instances for CV
        xgb_cv_without = XGBClassifier(**model_params)
        xgb_cv_with = XGBClassifier(**model_params)

        cv_recall_without = cross_val_score(xgb_cv_without, X_train_without, y_train, 
                                            cv=5, scoring='recall', n_jobs=-1)
        cv_recall_with = cross_val_score(xgb_cv_with, X_train_with, y_train, 
                                        cv=5, scoring='recall', n_jobs=-1)

        
        # TRAIN FINAL MODELS
        
        # Model 1: WITHOUT Z-Score
        xgb_without_z = XGBClassifier(**model_params)
        xgb_without_z.fit(X_train_without, y_train)

        
        # Model 2: WITH Z-Score
        xgb_with_z = XGBClassifier(**model_params)
        xgb_with_z.fit(X_train_with, y_train)
        
        # PREDICTIONS
        
        # WITHOUT Z-Score predictions
        y_pred_without = xgb_without_z.predict(X_test_without)
        y_pred_proba_without = xgb_without_z.predict_proba(X_test_without)[:, 1]
        
        # WITH Z-Score predictions
        y_pred_with = xgb_with_z.predict(X_test_with)
        y_pred_proba_with = xgb_with_z.predict_proba(X_test_with)[:, 1]
        
        # CALCULATE METRICS
        def calculate_metrics(y_true, y_pred, y_pred_proba):
            """Calculate all performance metrics"""
            return {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, pos_label=1, average='binary'),
                'Recall': recall_score(y_true, y_pred, pos_label=1, average='binary'),
                'F1-Score': f1_score(y_true, y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
            }
        
        # Calculate metrics for both models
        metrics_without = calculate_metrics(y_test, y_pred_without, y_pred_proba_without)
        metrics_with = calculate_metrics(y_test, y_pred_with, y_pred_proba_with)
        
        # DISPLAY RESULTS

        print("COMPARISON RESULTS")

        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Metric': list(metrics_without.keys()),
            'WITHOUT Z-Score': list(metrics_without.values()),
            'WITH Z-Score': list(metrics_with.values())
        })
        
        comparison['Difference'] = comparison['WITH Z-Score'] - comparison['WITHOUT Z-Score']
        comparison['% Improvement'] = (comparison['Difference'] / comparison['WITHOUT Z-Score']) * 100
        
        print("\n" + comparison.to_string(index=False))

        # Add CV results
        print(f"\nCross-Validation Recall:")
        print(f"  WITHOUT Z-Score: {cv_recall_without.mean():.4f} (+/- {cv_recall_without.std()*2:.4f})")
        print(f"  WITH Z-Score:    {cv_recall_with.mean():.4f} (+/- {cv_recall_with.std()*2:.4f})")
        print(f"  Improvement:     {(cv_recall_with.mean() - cv_recall_without.mean())*100:+.2f}%")
        

        # CONFUSION MATRICES

        print("CONFUSION MATRICES")
        cm_without = confusion_matrix(y_test, y_pred_without)
        cm_with = confusion_matrix(y_test, y_pred_with)
        
        print("\nWITHOUT Z-Score:")
        print(f"                 Predicted")
        print(f"                 Not B.  Bankrupt")
        print(f"Actual Not B.    {cm_without[0,0]:5d}   {cm_without[0,1]:5d}")
        print(f"Actual Bankrupt  {cm_without[1,0]:5d}   {cm_without[1,1]:5d}")
        
        print("\nWITH Z-Score:")
        print(f"                 Predicted")
        print(f"                 Not B.  Bankrupt")
        print(f"Actual Not B.    {cm_with[0,0]:5d}   {cm_with[0,1]:5d}")
        print(f"Actual Bankrupt  {cm_with[1,0]:5d}   {cm_with[1,1]:5d}")
        
        # Calculate improvement in catching bankruptcies
        tp_without = cm_without[1, 1]
        tp_with = cm_with[1, 1]
        fn_without = cm_without[1, 0]
        fn_with = cm_with[1, 0]
        
        
        # VISUALIZATIONS

        # Plot 1: Metrics Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            without_val = comparison[comparison['Metric'] == metric]['WITHOUT Z-Score'].values[0]
            with_val = comparison[comparison['Metric'] == metric]['WITH Z-Score'].values[0]
            
            # Simple bar chart with uniform styling
            values = [without_val, with_val]
            labels = ['WITHOUT\nZ-Score', 'WITH\nZ-Score']
            bar_colors = [colors[idx], colors[idx]]
            
            bars = ax.bar(labels, values, color=bar_colors, alpha=0.8,
                        edgecolor='black', linewidth=2, width=0.6)
            
            # Highlight the better bar with darker color
            if with_val > without_val:
                bars[1].set_alpha(1.0)
                bars[0].set_alpha(0.5)
            else:
                bars[0].set_alpha(1.0)
                bars[1].set_alpha(0.5)
            
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{val:.4f}', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
            
            # Add improvement annotation
            improvement = with_val - without_val
            if improvement > 0:
                ax.text(0.5, 0.95, f'+{improvement*100:.2f}%', 
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=10, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        # Hide 6th subplot
        axes[1, 2].axis('off')

        plt.suptitle('XGBoost Performance: WITH vs WITHOUT Altman Z-Score', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        
        # Plot 2: ROC Curves
        fpr_without, tpr_without, _ = roc_curve(y_test, y_pred_proba_without)
        fpr_with, tpr_with, _ = roc_curve(y_test, y_pred_proba_with)
        
        _=plt.figure(figsize=(10, 8))
        _=plt.plot(fpr_without, tpr_without, linewidth=2, 
                label=f'WITHOUT Z-Score (AUC = {metrics_without["ROC-AUC"]:.4f})',
                color='#3498db')
        _=plt.plot(fpr_with, tpr_with, linewidth=2, 
                label=f'WITH Z-Score (AUC = {metrics_with["ROC-AUC"]:.4f})',
                color='#e74c3c')
        _=plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        _=plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        _=plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        _=plt.title('ROC Curve Comparison: XGBoost WITH vs WITHOUT Altman Z-Score', 
                fontsize=14, fontweight='bold')
        _=plt.legend(fontsize=11, loc='lower right')
        _=plt.grid(alpha=0.3)
        _=plt.tight_layout()

        plt.show()
        
        # FEATURE IMPORTANCE
        print("FEATURE IMPORTANCE ANALYSIS")
        feature_importance_with = pd.DataFrame({
            'Feature': X_with_z.columns,
            'Importance': xgb_with_z.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 20 Features (WITH Z-Score model):")
        print(feature_importance_with.head(20).to_string(index=False))

        # RETURN RESULTS
        results = {
            'comparison_df': comparison,
            'metrics_without': metrics_without,
            'metrics_with': metrics_with,
            'cv_scores_without': cv_recall_without,
            'cv_scores_with': cv_recall_with,
            'confusion_matrix_without': cm_without,
            'confusion_matrix_with': cm_with,
            'model_without_z': xgb_without_z,
            'model_with_z': xgb_with_z,
            'feature_importance': feature_importance_with,
            'y_test': y_test,
            'y_pred_proba_without': y_pred_proba_without,
            'y_pred_proba_with': y_pred_proba_with
        }
        
        return results
    
# ERROR ANALYSIS
    def error_analysis(self, data, model, test_size=0.2, random_state=42):
        """
        Comprehensive error analysis for XGBoost bankruptcy prediction model
        
        Analyzes:
        - Confusion matrix
        - False negatives (missed bankruptcies) 
        - False positives (false alarms)
        - Prediction probability calibration
        - Feature importance
        - Threshold optimization
        
        Parameters:
        -----------
        data : DataFrame
            Input dataframe with features and 'Bankrupt' target
        test_size : float
            Proportion of data for testing (default: 0.2)
        random_state : int
            Random seed for reproducibility (default: 42)
        
        Returns:
        --------
        dict : Dictionary containing analysis results, model, and predictions
        """
                
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import (confusion_matrix, precision_score, recall_score, brier_score_loss, 
                                    accuracy_score, f1_score, precision_recall_curve, roc_curve)
        from sklearn.calibration import calibration_curve
        
        # Prepare data
        X = data.drop(columns=['Bankrupt'])
        y = data['Bankrupt']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        

        # CONFUSION MATRIX
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Not Bankrupt  Bankrupt")
        print(f"Actual Not B.    {tn:6d}        {fp:6d}")
        print(f"Actual Bankrupt  {fn:6d}        {tp:6d}")
        
        print(f"\nDetailed Breakdown:")
        print(f"  True Negatives (TN):  {tn:4d} - Correctly predicted NOT bankrupt")
        print(f"  False Positives (FP): {fp:4d} - Incorrectly predicted bankrupt (Type I error)")
        print(f"  False Negatives (FN): {fn:4d} - Missed bankruptcies (Type II error)")
        print(f"  True Positives (TP):  {tp:4d} - Correctly predicted bankrupt")
        
        print(f"\nError Rates:")
        print(f"  False Positive Rate: {fp/(fp+tn):.4f} ({fp} out of {fp+tn} non-bankrupt)")
        print(f"  False Negative Rate: {fn/(fn+tp):.4f} ({fn} out of {fn+tp} bankrupt)")
        print(f"  Missed Bankruptcy Cost: {fn} companies we failed to catch")
        

        # ANALYZE FALSE NEGATIVES

        print("FALSE NEGATIVE ANALYSIS (Missed Bankruptcies)")

        
        # Create analysis DataFrame
        X_test_with_results = X_test.copy()
        X_test_with_results['Actual'] = y_test.values
        X_test_with_results['Predicted'] = y_pred
        X_test_with_results['Pred_Proba'] = y_pred_proba
        
        # Identify error types
        X_test_with_results['Error_Type'] = 'Correct'
        X_test_with_results.loc[(X_test_with_results['Actual'] == 1) & 
                                (X_test_with_results['Predicted'] == 0), 'Error_Type'] = 'False Negative'
        X_test_with_results.loc[(X_test_with_results['Actual'] == 0) & 
                                (X_test_with_results['Predicted'] == 1), 'Error_Type'] = 'False Positive'
        
        # Get false negatives
        false_negatives = X_test_with_results[X_test_with_results['Error_Type'] == 'False Negative']
        true_positives = X_test_with_results[(X_test_with_results['Actual'] == 1) & 
                                            (X_test_with_results['Predicted'] == 1)]
        
        print(f"\nMissed {len(false_negatives)} out of {len(false_negatives) + len(true_positives)} bankruptcies")
        print(f"Miss rate: {len(false_negatives)/(len(false_negatives) + len(true_positives))*100:.1f}%")
        
        # Analyze prediction probabilities of missed cases
        print(f"\nPrediction Probability Distribution for False Negatives:")
        print(false_negatives['Pred_Proba'].describe())
        print(f"\nThese companies were bankrupt but model gave them low bankruptcy probability")
        print(f"  Min probability: {false_negatives['Pred_Proba'].min():.4f}")
        print(f"  Max probability: {false_negatives['Pred_Proba'].max():.4f}")
        print(f"  Mean probability: {false_negatives['Pred_Proba'].mean():.4f}")
        print(f"  Median probability: {false_negatives['Pred_Proba'].median():.4f}")
        
        # Compare feature values: False Negatives vs True Positives
        print("Feature Comparison: Missed Bankruptcies vs Correctly Caught Bankruptcies")

        
        feature_cols = [col for col in X_test.columns if col.startswith('X')][:10]
        
        comparison_data = []
        for feature in feature_cols:
            fn_mean = false_negatives[feature].mean()
            tp_mean = true_positives[feature].mean()
            difference = abs(fn_mean - tp_mean)
            pct_diff = (difference / (tp_mean + 1e-8)) * 100
            
            comparison_data.append({
                'Feature': feature,
                'False_Negative_Mean': fn_mean,
                'True_Positive_Mean': tp_mean,
                'Absolute_Difference': difference,
                'Percent_Difference': pct_diff
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('Percent_Difference', ascending=False)
        print("\nTop 10 Features with Largest Differences:")
        print(comparison_df.head(10).to_string(index=False))
        
        # Visualize false negative probability distribution
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Probability distribution
        _=axes[0].hist(true_positives['Pred_Proba'], bins=20, alpha=0.6, 
                    label='Correctly Caught (TP)', color='green', edgecolor='black')
        _=axes[0].hist(false_negatives['Pred_Proba'], bins=20, alpha=0.6, 
                    label='Missed (FN)', color='red', edgecolor='black')
        _=axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        _=axes[0].set_xlabel('Predicted Bankruptcy Probability', fontsize=11)
        _=axes[0].set_ylabel('Count', fontsize=11)
        _=axes[0].set_title('Probability Distribution: Caught vs Missed Bankruptcies', fontsize=12, fontweight='bold')
        _=axes[0].legend()
        _=axes[0].grid(alpha=0.3)
        
        # Plot 2: Feature comparison (top 5 features)
        top_5_features = comparison_df.head(5)['Feature'].tolist()
        x_pos = np.arange(len(top_5_features))
        width = 0.35
        
        fn_means = [comparison_df[comparison_df['Feature']==f]['False_Negative_Mean'].values[0] 
                    for f in top_5_features]
        tp_means = [comparison_df[comparison_df['Feature']==f]['True_Positive_Mean'].values[0] 
                    for f in top_5_features]
        
        _=axes[1].bar(x_pos - width/2, tp_means, width, label='Caught (TP)', color='green', alpha=0.7)
        _=axes[1].bar(x_pos + width/2, fn_means, width, label='Missed (FN)', color='red', alpha=0.7)
        _=axes[1].set_xlabel('Features', fontsize=11)
        _=axes[1].set_ylabel('Mean Value', fontsize=11)
        _=axes[1].set_title('Top 5 Discriminating Features', fontsize=12, fontweight='bold')
        _=axes[1].set_xticks(x_pos)
        _=axes[1].set_xticklabels(top_5_features, rotation=45)
        _=axes[1].legend()
        _=axes[1].grid(axis='y', alpha=0.3)
        
        _=plt.tight_layout()
        plt.show()
        

        # ANALYZE FALSE POSITIVES (False Alarms)
        print("FALSE POSITIVE ANALYSIS (False Alarms)")
        
        false_positives = X_test_with_results[X_test_with_results['Error_Type'] == 'False Positive']
        true_negatives = X_test_with_results[(X_test_with_results['Actual'] == 0) & 
                                            (X_test_with_results['Predicted'] == 0)]
        
        print(f"\nFalse alarms: {len(false_positives)} out of {len(false_positives) + len(true_negatives)} non-bankrupt")
        print(f"False positive rate: {len(false_positives)/(len(false_positives) + len(true_negatives))*100:.1f}%")
        
        # Visualize false positive probabilities
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Full view
        _=axes[0].hist(true_negatives['Pred_Proba'], bins=30, alpha=0.6, 
                    label='Correctly Predicted Not Bankrupt (TN)', color='green', edgecolor='black')
        _=axes[0].hist(false_positives['Pred_Proba'], bins=30, alpha=0.6, 
                    label='False Alarms (FP)', color='red', edgecolor='black')
        _=axes[0].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        _=axes[0].set_xlabel('Predicted Bankruptcy Probability', fontsize=12)
        _=axes[0].set_ylabel('Count', fontsize=12)
        _=axes[0].set_title('False Positive Analysis - Full View', fontsize=14, fontweight='bold')
        _=axes[0].legend()
        _=axes[0].grid(alpha=0.3)
        
        #Zoomed view (excluding counts over 700 for better visibility)
        _=axes[1].hist(true_negatives['Pred_Proba'], bins=30, alpha=0.6, 
                    label='Correctly Predicted Not Bankrupt (TN)', color='green', edgecolor='black')
        _=axes[1].hist(false_positives['Pred_Proba'], bins=30, alpha=0.6, 
                    label='False Alarms (FP)', color='red', edgecolor='black')
        _=axes[1].axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        _=axes[1].set_xlabel('Predicted Bankruptcy Probability', fontsize=12)
        _=axes[1].set_ylabel('Count', fontsize=12)
        _=axes[1].set_title('False Positive Analysis - Zoomed View (y-axis capped at 60)', fontsize=14, fontweight='bold')
        _=axes[1].set_ylim([0, 60])  # Limit y-axis to 100 to see FP distribution better
        _=axes[1].legend()
        _=axes[1].grid(alpha=0.3)
        
        _=plt.tight_layout()
        plt.show()
        

        # PREDICTION PROBABILITY CALIBRATION

        # Use sklearn's calibration_curve
        prob_true, prob_pred = calibration_curve(
            y_test, 
            y_pred_proba, 
            n_bins=10, 
            strategy='quantile'
        )
        
        # Create probability bins
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        X_test_with_results['Prob_Bin'] = pd.cut(X_test_with_results['Pred_Proba'], bins=bins)
        
        # Calculate actual bankruptcy rate per bin
        calibration = X_test_with_results.groupby('Prob_Bin', observed=True).agg({
            'Actual': ['sum', 'count', 'mean']
        }).round(4)
        
        calibration.columns = ['Actual_Bankruptcies', 'Total_Count', 'Actual_Rate']
        calibration['Predicted_Rate'] = [(b.left + b.right)/2 for b in calibration.index]
        
        print("\nCalibration Analysis:")
        print("If model is well-calibrated, Actual_Rate ≈ Predicted_Rate")
        print(calibration)
        
        # Calculate Expected Calibration Error (ECE)
        # ECE measures average absolute difference between predicted and actual rates
        ece = np.mean(np.abs(calibration['Actual_Rate'] - calibration['Predicted_Rate']))
        
        # Calculate Brier Score (lower is better)
        brier_score = brier_score_loss(y_test, y_pred_proba)
        
        print(f"\n Calibration Metrics:")
        print(f"  Expected Calibration Error (ECE): {ece:.4f}")
        print(f"     Interpretation: Average deviation from perfect calibration")
        print(f"    Range: 0 (perfect) to 1 (worst)")
        
        print(f"\n  Brier Score: {brier_score:.4f}")
        print(f"     Interpretation: Mean squared error of probability predictions")
        print(f"     Range: 0 (perfect) to 1 (worst)")
        print(f"   → Good: < 0.10, Acceptable: < 0.20")
        
        # Assess calibration quality
        if ece < 0.05:
            calibration_quality = "Excellent"
        elif ece < 0.10:
            calibration_quality = "Good"
        elif ece < 0.15:
            calibration_quality = "Acceptable"
        else:
            calibration_quality = "Poor"
        
        print(f"\n  Overall Calibration Quality: {calibration_quality}")
        
        # Plot calibration curve
        plt.figure(figsize=(10, 6))
        actual_rates = calibration['Actual_Rate'].values
        predicted_rates = calibration['Predicted_Rate'].values
        
        plt.plot(predicted_rates, actual_rates, 'o-', linewidth=2, markersize=8, label='XGBoost')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
        plt.xlabel('Predicted Probability', fontsize=12)
        plt.ylabel('Actual Bankruptcy Rate', fontsize=12)
        plt.title(f'Calibration Curve - XGBoost (ECE={ece:.4f}, Brier={brier_score:.4f})', 
                fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add ECE annotation
        plt.text(0.05, 0.95, f'ECE: {ece:.4f}\nBrier: {brier_score:.4f}\n{calibration_quality}', 
                transform=plt.gca().transAxes, 
                fontsize=11, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        # THRESHOLD ANALYSIS

        print("THRESHOLD SENSITIVITY ANALYSIS")
        
        precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
        
        # Find different threshold scenarios
        thresholds_to_test = [0.2, 0.25, 0.3, 0.4, 0.5, 0.6]
        
        print("\nImpact of Different Thresholds:")
        print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'FN (Missed)':<15} {'FP (False Alarms)':<15}")
        print("-"*80)
        
        for threshold in thresholds_to_test:
            y_pred_at_threshold = (y_pred_proba >= threshold).astype(int)
            acc = accuracy_score(y_test, y_pred_at_threshold)
            prec = precision_score(y_test, y_pred_at_threshold, pos_label=1, average='binary')
            rec = recall_score(y_test, y_pred_at_threshold, pos_label=1, average='binary')
            f1 = f1_score(y_test, y_pred_at_threshold, zero_division=0)
            
            cm_temp = confusion_matrix(y_test, y_pred_at_threshold)
            tn_t, fp_t, fn_t, tp_t = cm_temp.ravel()
            
            print(f"{threshold:<12.2f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {fn_t:<15d} {fp_t:<15d}")
        
        # Visualize precision-recall tradeoff
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Precision-Recall curve
        _=axes[0].plot(recalls, precisions, linewidth=2, label='XGBoost')
        _=axes[0].set_xlabel('Recall', fontsize=12)
        _=axes[0].set_ylabel('Precision', fontsize=12)
        _=axes[0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        _=axes[0].grid(alpha=0.3)
        _=axes[0].legend()
        
        # Mark current threshold (0.5)
        idx_05 = np.argmin(np.abs(pr_thresholds - 0.5))
        _=axes[0].plot(recalls[idx_05], precisions[idx_05], 'ro', markersize=10, 
                    label=f'Threshold=0.5')
        _=axes[0].legend()
        
        # Plot 2: Threshold vs Metrics
        threshold_range = np.linspace(0.1, 0.9, 50)
        prec_at_thresh = []
        rec_at_thresh = []
        f1_at_thresh = []
        
        for thresh in threshold_range:
            y_pred_temp = (y_pred_proba >= thresh).astype(int)
            prec_at_thresh.append(precision_score(y_test, y_pred_temp, pos_label=1, average='binary'))
            rec_at_thresh.append(recall_score(y_test, y_pred_temp, average='binary', zero_division=0))
            f1_at_thresh.append(f1_score(y_test, y_pred_temp, zero_division=0))
        
        _=axes[1].plot(threshold_range, prec_at_thresh, label='Precision', linewidth=2)
        _=axes[1].plot(threshold_range, rec_at_thresh, label='Recall', linewidth=2)
        _=axes[1].plot(threshold_range, f1_at_thresh, label='F1-Score', linewidth=2)
        _=axes[1].axvline(0.5, color='red', linestyle='--', label='Current (0.5)')
        _=axes[1].set_xlabel('Threshold', fontsize=12)
        _=axes[1].set_ylabel('Score', fontsize=12)
        _=axes[1].set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
        _=axes[1].legend()
        _=axes[1].grid(alpha=0.3)
        
        _=plt.tight_layout()
        plt.show()

        optimal_f1_idx = np.argmax(f1_at_thresh)
        optimal_threshold = threshold_range[optimal_f1_idx]
        
        cv_precision_train = cross_val_score(model, X_train, y_train, 
                                         cv=5, scoring='precision', n_jobs=-1)
        cv_recall_train = cross_val_score(model, X_train, y_train, 
                                            cv=5, scoring='recall', n_jobs=-1)
        cv_f1_train = cross_val_score(model, X_train, y_train, 
                                        cv=5, scoring='f1', n_jobs=-1)

        # Test set performance (out-of-sample)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        # Create comparison
        print(f"\n{'Metric':<15} {'In-Sample (CV Mean)':<25} {'Out-of-Sample (Test)':<25} {'Difference':<15}")
        print("-"*80)

        # Precision comparison
        cv_prec_mean = cv_precision_train.mean()
        prec_diff = test_precision - cv_prec_mean
        prec_sign = "↑" if prec_diff > 0 else "↓"
        print(f"{'Precision':<15} {cv_prec_mean:>10.4f} ± {cv_precision_train.std():.4f}    {test_precision:>10.4f}              {prec_sign} {abs(prec_diff):>6.4f}")

        # Recall comparison
        cv_rec_mean = cv_recall_train.mean()
        rec_diff = test_recall - cv_rec_mean
        rec_sign = "↑" if rec_diff > 0 else "↓"
        print(f"{'Recall':<15} {cv_rec_mean:>10.4f} ± {cv_recall_train.std():.4f}    {test_recall:>10.4f}              {rec_sign} {abs(rec_diff):>6.4f}")

        # F1 comparison
        cv_f1_mean = cv_f1_train.mean()
        f1_diff = test_f1 - cv_f1_mean
        f1_sign = "↑" if f1_diff > 0 else "↓"
        print(f"{'F1-Score':<15} {cv_f1_mean:>10.4f} ± {cv_f1_train.std():.4f}    {test_f1:>10.4f}              {f1_sign} {abs(f1_diff):>6.4f}")

        
        
        # Return results
        results = {
            'model': model,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'false_negatives': false_negatives,
            'false_positives': false_positives,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'comparison_df': comparison_df,
            'calibration': {
                'prob_true': prob_true,
                'prob_pred': prob_pred,
                'calibration_df': calibration
            },
            'optimal_f1_threshold': optimal_threshold,
            'threshold_metrics': {
                'thresholds': threshold_range,
                'precision': prec_at_thresh,
                'recall': rec_at_thresh,
                'f1': f1_at_thresh
            }
        }
        
        return results