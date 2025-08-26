import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score, confusion_matrix, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# STEP 1: DATA LOADING AND NORMALIZATION

def load_and_normalize_gait_data(
    gait_features_path: str = "corrected_gait_features.csv",
    demographics_path: str = "demographics.csv"
) -> pd.DataFrame:
    # Load data
    gait_df = pd.read_csv(gait_features_path)
    demo_df = pd.read_csv(demographics_path)
    
    print(f"   Original gait samples: {len(gait_df)}")
    print(f"   Demographic records: {len(demo_df)}")
    
    # Check for duplicates in original gait data
    if gait_df.duplicated(subset=['filename']).any():
        print(f"   ⚠️ Found {gait_df.duplicated(subset=['filename']).sum()} duplicate filenames")
        print("   Removing duplicates from gait data...")
        gait_df = gait_df.drop_duplicates(subset=['filename'], keep='first')
        print(f"   Cleaned gait samples: {len(gait_df)}")
    
    # Standardize merge keys
    gait_df['merge_study'] = gait_df['study'].str.capitalize()
    gait_df['merge_group'] = gait_df['subject_type']
    gait_df['merge_subjnum'] = gait_df['subject_number'].astype(int)
    
    demo_df['merge_study'] = demo_df['Study'].str.capitalize()
    demo_df['merge_group'] = demo_df['Group'].apply(lambda x: 'Co' if 'Co' in str(x) else 'Pt')
    demo_df['merge_subjnum'] = demo_df['Subjnum'].astype(int)
    
    # Check for duplicates in demographics
    demo_duplicates = demo_df.duplicated(subset=['merge_study', 'merge_group', 'merge_subjnum'], keep=False)
    if demo_duplicates.any():
        print(f"  Found {demo_duplicates.sum()} duplicate subjects in demographics")
        demo_df = demo_df.drop_duplicates(subset=['merge_study', 'merge_group', 'merge_subjnum'], keep='first')
    
    # Merge
    merged = pd.merge(
        gait_df,
        demo_df[['merge_study', 'merge_group', 'merge_subjnum', 
                'Height (meters)', 'Weight (kg)', 'Age', 'Gender']],
        on=['merge_study', 'merge_group', 'merge_subjnum'],
        how='left'
    )
    
    print(f"   Merged data size: {len(merged)}")
    
    if len(merged) > len(gait_df):
        print(f"\n   Merge created duplicates! {len(merged)} rows from {len(gait_df)}")
        print("   Remove post-merge duplicates")
        merged = merged.drop_duplicates(subset=['filename'], keep='first')
        print(f" {len(merged)} unique samples")
    
    # Verify no duplicates remain
    assert not merged.duplicated(subset=['filename']).any(), "Duplicates still present!"
    
    # Fill missing values with population averages
    merged['Height (meters)'].fillna(1.70, inplace=True)
    merged['Weight (kg)'].fillna(70.0, inplace=True)
    merged['Age'].fillna(merged['Age'].median() if 'Age' in merged else 60, inplace=True)
    
    print(f"merged {len(merged)} unique samples")
    
    # Apply normalizations
    height = merged['Height (meters)']
    weight = merged['Weight (kg)']
    bmi = weight / (height ** 2)
    leg_length = 0.53 * height
    
    # Normalize features
    if 'left_cop_path_length' in merged.columns:
        merged['left_cop_path_normalized'] = merged['left_cop_path_length'] / (height * 1000)
        merged['right_cop_path_normalized'] = merged['right_cop_path_length'] / (height * 1000)
    
    if 'step_count_total' in merged.columns and 'total_duration' in merged.columns:
        steps_per_second = merged['step_count_total'] / merged['total_duration']
        estimated_speed = steps_per_second * 0.7
        merged['froude_number'] = estimated_speed / np.sqrt(9.81 * leg_length)
    
    if 'overall_instability_index' in merged.columns:
        bmi_factor = 1.0 + 0.02 * (bmi - 25)
        merged['instability_bmi_adjusted'] = merged['overall_instability_index'] / bmi_factor
    
    if 'stride_asymmetry' in merged.columns and 'Age' in merged.columns:
        age_factor = 1.0 + 0.01 * (merged['Age'] - 50)
        merged['stride_asymmetry_age_adjusted'] = merged['stride_asymmetry'] / age_factor
    
    # Create Gait Quality Index
    print("\n Calculating Gait Quality Index")
    gqi_components = []
    
    if 'step_time_cv' in merged.columns:
        step_quality = 100 / (1 + merged['step_time_cv'])
        gqi_components.append(step_quality)
    
    if 'stride_asymmetry' in merged.columns:
        symmetry_quality = 100 / (1 + merged['stride_asymmetry'])
        gqi_components.append(symmetry_quality)
    
    if 'overall_instability_index' in merged.columns:
        stability_quality = 100 / (1 + merged['overall_instability_index'])
        gqi_components.append(stability_quality)
    
    if 'freezing_index' in merged.columns:
        freezing_quality = 100 / (1 + merged['freezing_index'] * 10)
        gqi_components.append(freezing_quality)
    
    if gqi_components:
        merged['gait_quality_index'] = pd.DataFrame(gqi_components).mean()
        print(f"GQI calculated from {len(gqi_components)} components")
    
    # Create subject ID
    merged['subject_id'] = (merged['study'].astype(str) + '_' + 
                            merged['subject_type'].astype(str) + '_' + 
                            merged['subject_number'].astype(str))
    
    # Final verification
    print(f"   Total samples: {len(merged)}")
    print(f"   Unique filenames: {merged['filename'].nunique()}")
    print(f"   Controls: {(merged['is_patient'] == 0).sum()}")
    print(f"   Patients: {(merged['is_patient'] == 1).sum()}")
    
    return merged

# STEP 2: CLUSTERING WITH CONTROL/PATIENT AWARENESS


def cluster_by_subject_type(normalized_df: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
    """
    Cluster SEPARATELY for controls and patients
    This ensures proper separation of disease states
    """
    
    print("\n" + "=" * 32) # 32 # (in )
    print("🔬 CLUSTERING GAIT PATTERNS BY SUBJECT TYPE")
    print("=" * 32)
    
    # Prepare features for clustering
    clustering_features = [
        'step_time_cv', 'stride_asymmetry', 'freezing_index',
        'overall_instability_index', 'gait_quality_index',
        'cadence_cv', 'left_sample_entropy', 'right_sample_entropy'
    ]
    
    if 'left_cop_path_normalized' in normalized_df.columns:
        clustering_features.extend(['left_cop_path_normalized', 'right_cop_path_normalized'])
    if 'instability_bmi_adjusted' in normalized_df.columns:
        clustering_features.append('instability_bmi_adjusted')
    
    available_features = [f for f in clustering_features if f in normalized_df.columns]
    print(f"\n Using {len(available_features)} features for clustering")
    
    # Separate controls and patients
    controls_df = normalized_df[normalized_df['is_patient'] == 0].copy()
    patients_df = normalized_df[normalized_df['is_patient'] == 1].copy()
    
    print(f"\n📊 Data split:")
    print(f"   Controls: {len(controls_df)} samples")
    print(f"   Patients: {len(patients_df)} samples")
    
    # STEP 1: All controls get Stage 0
    print("\n Assigning all controls to Stage 0 (Healthy)")
    controls_df['predicted_stage'] = 0
    controls_df['stage_name'] = 'Healthy'
    controls_df['cluster_id'] = -1  
    
    # STEP 2: Cluster patients into 3 groups (Early, Mild-Moderate, Advanced)
    
    X_patients = patients_df[available_features].fillna(patients_df[available_features].median())
    
    # Standardize features
    scaler = StandardScaler()
    X_patients_scaled = scaler.fit_transform(X_patients)
    
    # Cluster patients into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    patient_clusters = kmeans.fit_predict(X_patients_scaled)
    
    # Analyze patient clusters
    cluster_stats = {}
    for cluster_id in range(3):
        cluster_mask = patient_clusters == cluster_id
        cluster_data = patients_df[cluster_mask]
        
        stats = {
            'size': cluster_mask.sum(),
            'mean_step_cv': X_patients[cluster_mask]['step_time_cv'].mean(),
            'mean_asymmetry': X_patients[cluster_mask]['stride_asymmetry'].mean(),
            'mean_freezing': X_patients[cluster_mask]['freezing_index'].mean(),
            'mean_instability': X_patients[cluster_mask]['overall_instability_index'].mean(),
            'mean_gqi': X_patients[cluster_mask]['gait_quality_index'].mean() if 'gait_quality_index' in X_patients else 50
        }
        cluster_stats[cluster_id] = stats
    
    # Map patient clusters to stages based on severity
    severity_scores = {}
    for cluster_id, stats in cluster_stats.items():
        severity = (
            stats['mean_step_cv'] * 2.0 +
            stats['mean_asymmetry'] * 100 +
            stats['mean_freezing'] * 200 +
            stats['mean_instability'] * 50 +
            (100 - stats['mean_gqi']) * 0.5
        )
        severity_scores[cluster_id] = severity
    
    # Sort by severity and assign stages
    sorted_clusters = sorted(severity_scores.items(), key=lambda x: x[1])
    cluster_to_stage = {}
    patient_stages = [2.0, 2.5, 3.0]  # Early, Mild-Moderate, Advanced
    
    for (cluster_id, severity), stage in zip(sorted_clusters, patient_stages):
        cluster_to_stage[cluster_id] = stage
        stats = cluster_stats[cluster_id]
        stage_name = {2.0: 'Early PD', 2.5: 'Mild-Moderate', 3.0: 'Advanced'}[stage]
        print(f"   Cluster {cluster_id} → Stage {stage} ({stage_name})")
        print(f"      Size: {stats['size']} patients")
        print(f"      Severity score: {severity:.1f}")
        print(f"      Step CV: {stats['mean_step_cv']:.1f}%, Freezing: {stats['mean_freezing']:.3f}")
    
    # Assign stages to patients
    patients_df['cluster_id'] = patient_clusters
    patients_df['predicted_stage'] = patients_df['cluster_id'].map(cluster_to_stage)
    patients_df['stage_name'] = patients_df['predicted_stage'].map({
        2.0: 'Early PD',
        2.5: 'Mild-Moderate',
        3.0: 'Advanced'
    })
    
    # Combine controls and patients
    combined_df = pd.concat([controls_df, patients_df], ignore_index=True)
    
    # Sort by original index to maintain order
    combined_df = combined_df.sort_index()
    
    # Verify assignments
    print(f"   Controls with Stage 0: {(combined_df[combined_df['is_patient']==0]['predicted_stage']==0).all()}")
    print(f"   Patients with Stage > 0: {(combined_df[combined_df['is_patient']==1]['predicted_stage']>0).all()}")
    
    return combined_df


# STEP 3: SUPERVISED REFINEMENT (WITHIN CONSTRAINTS)

def refine_predictions_with_constraints(df: pd.DataFrame) -> pd.DataFrame:
    
    print("\n" + "=" * 32) #is 
    print("=" * 32) # (my code)
    
    # Prepare features
    feature_columns = [
        'step_time_cv', 'cadence_cv', 'stride_asymmetry', 'force_asymmetry',
        'freezing_index', 'overall_instability_index',
        'left_sample_entropy', 'right_sample_entropy'
    ]
    
    if 'gait_quality_index' in df.columns:
        feature_columns.append('gait_quality_index')
    if 'left_cop_path_normalized' in df.columns:
        feature_columns.extend(['left_cop_path_normalized', 'right_cop_path_normalized'])
    
    features = [f for f in feature_columns if f in df.columns]
    
    # Separate controls and patients for separate models
    controls_df = df[df['is_patient'] == 0].copy()
    patients_df = df[df['is_patient'] == 1].copy()
    
    print(f"   Controls: {len(controls_df)} (all remain Stage 0)")
    print(f"   Patients: {len(patients_df)} (refining within 2.0/2.5/3.0)")
    
    # Controls stay at Stage 0
    controls_df['predicted_stage_refined'] = 0
    controls_df['stage_name_refined'] = 'Healthy'
    controls_df['prediction_confidence'] = 1.0  
    
    # Train model only on patients
    if len(patients_df) > 10:  # Need enough samples
        X_patients = patients_df[features].fillna(0)
        y_patients = patients_df['predicted_stage'].values
        
        # Split patients for training/testing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_patients, y_patients, test_size=0.2, random_state=42, stratify=y_patients
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=4,  
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        

        predictions_raw = model.predict(X_patients)
        
        # Constrain predictions to valid patient stages
        patient_stages = [2.0, 2.5, 3.0]
        predictions_refined = []
        confidences = []
        
        for pred in predictions_raw:
            # Find closest valid patient stage
            if pred < 2.0:
                closest = 2.0  # Minimum for patients
            elif pred > 3.0:
                closest = 3.0  # And Maximum for patients as well
            else:
                closest = min(patient_stages, key=lambda x: abs(x - pred))
            
            predictions_refined.append(closest)
            distance = abs(pred - closest)
            conf = 1.0 / (1.0 + distance)
            confidences.append(conf)
        
        patients_df['predicted_stage_refined'] = predictions_refined
        patients_df['prediction_confidence'] = confidences
        patients_df['stage_name_refined'] = patients_df['predicted_stage_refined'].map({
            2.0: 'Early PD',
            2.5: 'Mild-Moderate',
            3.0: 'Advanced'
        })
        
        # Evaluate
        test_pred = model.predict(X_test)
        test_pred_rounded = [min(patient_stages, key=lambda x: abs(x - p)) for p in test_pred]
        test_mae = mean_absolute_error(y_test, test_pred_rounded)
        
        print(f"\n Patient Model Performance:")
        print(f"   Test MAE: {test_mae:.3f} stages")
        
        # Feature importance
        print(f"\n📈 Top 5 Important Features:")
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in importance_df.head(5).iterrows():
            print(f"   {row['feature']:30} {row['importance']:.4f}")
    else:

        patients_df['predicted_stage_refined'] = patients_df['predicted_stage']
        patients_df['stage_name_refined'] = patients_df['stage_name']
        patients_df['prediction_confidence'] = 0.8
    
    # Combine results
    combined_df = pd.concat([controls_df, patients_df], ignore_index=True)
    
    # Final verification
    controls_check = combined_df[combined_df['is_patient']==0]
    patients_check = combined_df[combined_df['is_patient']==1]
    
    print(f"   All controls are Stage 0: {(controls_check['predicted_stage_refined']==0).all()}")
    print(f"   All patients are Stage ≥2.0: {(patients_check['predicted_stage_refined']>=2.0).all()}")
    print(f"   No patients are Stage 0: {(patients_check['predicted_stage_refined']==0).sum() == 0}")
    
    return combined_df



# STEP 4: MAIN PIPELINE

def run_pipeline():
    
    # Check files
    required_files = ["corrected_gait_features.csv", "demographics.csv"]
    
    print("\n🔍 Checking required files...")
    for file in required_files:
        if os.path.exists(file):
            print(f"Found: {file}")
        else:
            print(f"Missing: {file}")
            return None
    
    # Step 1: Load and normalize (with duplicate fix)
    print("\n" + "-" * 32) #my 
    print("STEP 1: LOAD AND NORMALIZE DATA")
    print("-" * 32)
    
    normalized_df = load_and_normalize_gait_data()
    
    # Step 2: Cluster by subject type
    print("\n" + "-" * 32) #official 
    print("STEP 2: CLUSTER BY SUBJECT TYPE")
    print("-" * 32)
    
    clustered_df = cluster_by_subject_type(normalized_df)
    
    # Step 3: Refine with constraints
    print("\n" + "-" * 32) #age 
    print("STEP 3: REFINE PREDICTIONS")
    print("-" * 32)
    
    final_df = refine_predictions_with_constraints(clustered_df)
    
    # Step 4: Final analysis
    print("\n" + "=" * 32) # (Secret)
    print(" FINAL RESULTS")
    print("=" * 32)
    
    # Use refined predictions
    final_df['final_stage'] = final_df['predicted_stage_refined']
    final_df['final_stage_name'] = final_df['stage_name_refined']
    
    # Distribution
    stage_dist = final_df['final_stage'].value_counts().sort_index()
    for stage, count in stage_dist.items():
        stage_names = {0: 'Stage 0: Healthy', 2.0: 'Stage 2.0: Early PD',
                      2.5: 'Stage 2.5: Mild-Moderate', 3.0: 'Stage 3.0: Advanced'}
        print(f"   {stage_names[stage]}: {count} subjects ({count/len(final_df)*100:.1f}%)")
    
    print("\n   CONTROLS:")
    controls = final_df[final_df['is_patient'] == 0]
    control_dist = controls['final_stage'].value_counts().sort_index()
    for stage, count in control_dist.items():
        print(f"      Stage {stage}: {count} ({count/len(controls)*100:.1f}%)")
    
    print("\n   PATIENTS:")
    patients = final_df[final_df['is_patient'] == 1]
    patient_dist = patients['final_stage'].value_counts().sort_index()
    for stage, count in patient_dist.items():
        stage_names = {2.0: 'Early PD', 2.5: 'Mild-Moderate', 3.0: 'Advanced'}
        print(f"      Stage {stage} ({stage_names.get(stage, 'Unknown')}): {count} ({count/len(patients)*100:.1f}%)")
    
    # Check 1: No duplicates
    duplicates = final_df.duplicated(subset=['filename'])
    print(f" No duplicate filenames: {not duplicates.any()} ({len(final_df)} unique samples)")
    
    # Check 2: Controls only Stage 0
    controls_wrong = final_df[(final_df['is_patient']==0) & (final_df['final_stage']!=0)]
    if len(controls_wrong) == 0:
        print(f"controls correctly assigned to Stage 0")
    else:
        print(f"ERROR: {len(controls_wrong)} controls not in Stage 0!")
        print(f"Files: {controls_wrong['filename'].tolist()[:5]}")
    
    # Check 3: Patients never Stage 0
    patients_wrong = final_df[(final_df['is_patient']==1) & (final_df['final_stage']==0)]
    if len(patients_wrong) == 0:
        print(f"No patients incorrectly assigned to Stage 0")
    else:
        print(f"ERROR: {len(patients_wrong)} patients in Stage 0!")
        print(f"Files: {patients_wrong['filename'].tolist()[:5]}")
    
    # Save results
    output_columns = [
        'filename', 'study', 'subject_type', 'subject_number',
        'is_patient', 'Age', 'Gender',
        'final_stage', 'final_stage_name',
        'prediction_confidence',
        'step_time_cv', 'stride_asymmetry', 'freezing_index',
        'overall_instability_index', 'gait_quality_index'
    ]
    
    output_columns = [c for c in output_columns if c in final_df.columns]
    output_df = final_df[output_columns].copy()
    
    # Final verification before saving
    print(f"\n Output File Summary:")
    print(f"   Total rows: {len(output_df)}")
    print(f"   Unique filenames: {output_df['filename'].nunique()}")
    print(f"   Has duplicates: {output_df.duplicated(subset=['filename']).any()}")
    
    output_df.to_csv('hy_four_stage_predictions.csv', index=False)
    print(f"\nResults has been saved to: hy_four_stage_predictions.csv")
    
    # Create visualization
    create_visualizations(final_df)
    
    print("\n" + "=" * 32) # (Code )
    print("=" * 32)
    
    return final_df


def create_visualizations(df: pd.DataFrame):
    
    plt.figure(figsize=(15, 10))
    
    all_stages = [0, 2.0, 2.5, 3.0]
    stage_names = ['Healthy\n(0)', 'Early PD\n(2.0)', 'Mild-Mod\n(2.5)', 'Advanced\n(3.0)']
    colors = ['lightgreen', 'yellow', 'orange', 'red']
    
    # Plot 1: Overall distribution
    plt.subplot(2, 3, 1)
    stage_counts = df['final_stage'].value_counts().reindex(all_stages, fill_value=0)
    bars = plt.bar(range(4), stage_counts.values, color=colors, alpha=0.7)
    plt.xticks(range(4), stage_names)
    plt.ylabel('Number of Subjects')
    plt.title('Stage Distribution (Fixed)')
    plt.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, stage_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(count)), ha='center', va='bottom')
    
    # Plot 2: By subject type
    plt.subplot(2, 3, 2)
    width = 0.35
    x = np.array([0, 1, 2, 3])
    
    for i, is_patient in enumerate([0, 1]):
        subset = df[df['is_patient'] == is_patient]
        counts = [len(subset[subset['final_stage']==s]) for s in all_stages]
        
        label = "Patients" if is_patient else "Controls"
        offset = width/2 if is_patient else -width/2
        bars = plt.bar(x + offset, counts, width, label=label, alpha=0.7)
        
        # Add text labels
        for bar, count in zip(bars, counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(int(count)), ha='center', va='bottom', fontsize=8)
    
    plt.xticks(x, stage_names)
    plt.ylabel('Count')
    plt.title('Controls vs Patients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Gait features by stage
    plt.subplot(2, 3, 3)
    features_by_stage = df.groupby('final_stage')[
        ['step_time_cv', 'freezing_index', 'gait_quality_index']
    ].mean()
    features_by_stage = features_by_stage.reindex(all_stages)
    features_by_stage.plot(kind='bar', ax=plt.gca())
    plt.xlabel('Stage')
    plt.ylabel('Mean Value')
    plt.title('Key Features by Stage')
    plt.xticks([0, 1, 2, 3], stage_names, rotation=0)
    plt.legend(['Step CV', 'Freezing', 'GQI'], loc='best')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Step CV distribution
    plt.subplot(2, 3, 4)
    for stage, color in zip(all_stages, colors):
        subset = df[df['final_stage'] == stage]
        if not subset.empty:
            plt.scatter(subset['step_time_cv'], subset['gait_quality_index'],
                       alpha=0.5, color=color, label=f'Stage {stage}', s=50)
    plt.xlabel('Step Time CV (%)')
    plt.ylabel('Gait Quality Index')
    plt.title('Step Variability vs Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Verification plot - Subject type vs Stage
    plt.subplot(2, 3, 5)
    verification_data = df.groupby(['is_patient', 'final_stage']).size().unstack(fill_value=0)
    verification_data.T.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.xlabel('Stage')
    plt.ylabel('Count')
    plt.title('Verification: Subject Type by Stage')
    plt.xticks([0, 1, 2, 3], all_stages, rotation=0)
    plt.legend(['Controls', 'Patients'], title='Subject Type')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Confidence distribution
    plt.subplot(2, 3, 6)
    for stage, color in zip(all_stages, colors):
        subset = df[df['final_stage'] == stage]
        if not subset.empty and 'prediction_confidence' in subset.columns:
            plt.hist(subset['prediction_confidence'], bins=20, alpha=0.5,
                    color=color, label=f'Stage {stage}')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Confidence by Stage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Fixed Four-Stage Predictions (No Duplicates, Proper Control Handling)', fontsize=14)
    plt.tight_layout()
    plt.savefig('hy_four_stage_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n Visualizations saved to: hy__four_stage_results.png")


# MAIN EXECUTION

if __name__ == "__main__":
    
    # Run the pipeline
    results = run_pipeline()
    
    if results is not None:
        print("\n Success! Check saved .csv")