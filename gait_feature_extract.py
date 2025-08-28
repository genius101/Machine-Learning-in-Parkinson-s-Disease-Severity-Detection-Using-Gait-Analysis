import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import os
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class CorrectedGaitFeatureExtractor:
    
    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        
        # Sensor positions for COP calculation
        self.left_sensors = np.array([
            [-500, -800], [-700, -400], [-300, -400], [-700, 0],
            [-300, 0], [-700, 400], [-300, 400], [-500, 800]
        ])
        
        self.right_sensors = np.array([
            [500, -800], [700, -400], [300, -400], [700, 0],
            [300, 0], [700, 400], [300, 400], [500, 800]
        ])
    
    def load_data(self, filepath: str) -> Optional[np.ndarray]:
        try:
            data = np.loadtxt(filepath, dtype=np.float32)
            print(f"  Loaded {len(data)} samples ({data[-1,0]:.1f} seconds)")
            return data
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        basename = os.path.basename(filename).replace('.txt', '')
        parts = basename.split('_')
        
        if len(parts) != 2:
            raise ValueError(f"Invalid filename format: {filename}")
        
        first_part = parts[0]
        study_type = first_part[:2]
        subject_type = first_part[2:4]
        subject_num = first_part[4:]
        walk_num = parts[1]
        
        return {
            'study': study_type,
            'subject_type': subject_type,
            'subject_number': subject_num,
            'walk_number': walk_num,
            'is_patient': subject_type == 'Pt',
            'is_dual_task': walk_num == '10' and study_type == 'Ga'
        }
    
    def detect_steps_optimized(self, force_signal: np.ndarray, threshold: float = 50.0) -> List[tuple]:
        above_threshold = force_signal > threshold
        
        # Find transitions
        transitions = np.diff(above_threshold.astype(int))
        step_starts = np.where(transitions == 1)[0] + 1
        step_ends = np.where(transitions == -1)[0] + 1
        
        # Handle edge cases
        if above_threshold[0]:
            step_starts = np.concatenate([[0], step_starts])
        if above_threshold[-1]:
            step_ends = np.concatenate([step_ends, [len(force_signal)]])
        
        # Ensure matching pairs
        min_length = min(len(step_starts), len(step_ends))
        if min_length == 0:
            return []
        
        step_starts = step_starts[:min_length]
        step_ends = step_ends[:min_length]
        
        # Filter by minimum duration
        min_duration = 10  # samples
        step_durations = step_ends - step_starts
        valid_steps = step_durations >= min_duration
        
        valid_starts = step_starts[valid_steps]
        valid_ends = step_ends[valid_steps]
        
        return list(zip(valid_starts, valid_ends))
    
    def calculate_cop_optimized(self, sensor_forces: np.ndarray, sensor_positions: np.ndarray) -> tuple:
        total_forces = np.sum(sensor_forces, axis=1)
        
        valid_forces = total_forces > 0
        cop_x = np.zeros(len(sensor_forces))
        cop_y = np.zeros(len(sensor_forces))
        
        if np.any(valid_forces):
            cop_x[valid_forces] = np.sum(sensor_forces[valid_forces] * sensor_positions[:, 0], axis=1) / total_forces[valid_forces]
            cop_y[valid_forces] = np.sum(sensor_forces[valid_forces] * sensor_positions[:, 1], axis=1) / total_forces[valid_forces]
        
        return cop_x, cop_y
    
    def extract_corrected_features(self, data: np.ndarray) -> Dict[str, float]:
        features = {}
        
        # Basic data extraction
        left_total = data[:, 17].astype(np.float32)
        right_total = data[:, 18].astype(np.float32)
        left_sensors = data[:, 1:9].astype(np.float32)
        right_sensors = data[:, 9:17].astype(np.float32)
        time_vector = data[:, 0]
        
        left_steps = self.detect_steps_optimized(left_total)
        right_steps = self.detect_steps_optimized(right_total)
        
        print(f"      Left steps: {len(left_steps)}, Right steps: {len(right_steps)}")
        
        def get_true_step_times(left_steps, right_steps):
            all_strikes = []
            
            # Combine all heel strikes with foot labels
            for start, _ in left_steps:
                all_strikes.append((start, 'left'))
            for start, _ in right_steps:
                all_strikes.append((start, 'right'))
            
            # Sort by time
            all_strikes.sort()
            
            step_times = []
            for i in range(len(all_strikes) - 1):
                # Time between consecutive heel strikes (alternating feet)
                step_time = (all_strikes[i+1][0] - all_strikes[i][0]) / self.sampling_rate
                step_times.append(step_time)
            
            return np.array(step_times)
        
        true_step_times = get_true_step_times(left_steps, right_steps)
        
        # Step time CV (primary PD detection measure)
        if len(true_step_times) > 0:
            features['step_time_cv'] = (np.std(true_step_times) / np.mean(true_step_times)) * 100
        else:
            features['step_time_cv'] = 0
        
        def get_gait_phases_exact(steps):
            if len(steps) < 2:
                return np.array([]), np.array([]), np.array([])
            
            stance_times = []  # Tstance(j) = tend(j) - tstart(j)
            swing_times = []   # Tswing(j) = tstart(j+1) - tend(j)
            stride_times = []  # GCT(j) = tstart(j+1) - tstart(j)
            
            for i in range(len(steps) - 1):
                start_i, end_i = steps[i]
                start_next, _ = steps[i + 1]
                
                # Stance duration: foot contact time
                stance_duration = (end_i - start_i) / self.sampling_rate
                stance_times.append(stance_duration)
                
                # Swing duration: airborne time
                swing_duration = (start_next - end_i) / self.sampling_rate
                if swing_duration > 0:  # Valid swing phase
                    swing_times.append(swing_duration)
                
                # Stride time: time between consecutive heel strikes of same foot
                stride_time = (start_next - start_i) / self.sampling_rate
                stride_times.append(stride_time)
            
            return np.array(stance_times), np.array(swing_times), np.array(stride_times)
        
        # Calculate for both feet
        left_stance, left_swing, left_strides = get_gait_phases_exact(left_steps)
        right_stance, right_swing, right_strides = get_gait_phases_exact(right_steps)
        
        # CV measures for basic temporal features (FOCUS: Variability for PD detection)
        for foot, stance, swing, strides in [('left', left_stance, left_swing, left_strides), 
                                           ('right', right_stance, right_swing, right_strides)]:
            
            # Stance time CV (primary measure)
            if len(stance) > 0:
                features[f'{foot}_stance_time_cv'] = (np.std(stance) / np.mean(stance)) * 100
            else:
                features[f'{foot}_stance_time_cv'] = 0
            
            # Swing interval CV (primary measure)
            if len(swing) > 0:
                features[f'{foot}_swing_interval_cv'] = (np.std(swing) / np.mean(swing)) * 100
            else:
                features[f'{foot}_swing_interval_cv'] = 0
            
            # Stride time CV (primary measure)
            if len(strides) > 0:
                features[f'{foot}_stride_time_cv'] = (np.std(strides) / np.mean(strides)) * 100
            else:
                features[f'{foot}_stride_time_cv'] = 0
        
        def calculate_temporal_ratios(stance_times, swing_times, stride_times):
            ratios = {}
            
            # Swing-to-stance interval ratio: Rswing_stance(j) = Tswing(j)/Tstance(j)
            if len(stance_times) > 0 and len(swing_times) > 0:
                min_len = min(len(stance_times), len(swing_times))
                if min_len > 0:
                    swing_stance_ratios = swing_times[:min_len] / stance_times[:min_len]
                    ratios['swing_to_stance_ratio_cv'] = (np.std(swing_stance_ratios) / np.mean(swing_stance_ratios)) * 100
            
            # Stance period to stride time ratio: Rstance(j) = Tstance(j)/Tstride(j)
            if len(stance_times) > 0 and len(stride_times) > 0:
                min_len = min(len(stance_times), len(stride_times))
                if min_len > 0:
                    stance_ratios = stance_times[:min_len] / stride_times[:min_len]
                    ratios['stance_ratio_cv'] = (np.std(stance_ratios) / np.mean(stance_ratios)) * 100
            
            # Swing period to stride duration ratio: Rswing(j) = Tswing(j)/Tstride(j)
            if len(swing_times) > 0 and len(stride_times) > 0:
                min_len = min(len(swing_times), len(stride_times))
                if min_len > 0:
                    swing_ratios = swing_times[:min_len] / stride_times[:min_len]
                    ratios['standardized_swing_ratio_cv'] = (np.std(swing_ratios) / np.mean(swing_ratios)) * 100
            
            return ratios
        
        # Add ratios for both feet
        left_ratios = calculate_temporal_ratios(left_stance, left_swing, left_strides)
        right_ratios = calculate_temporal_ratios(right_stance, right_swing, right_strides)
        
        for key, value in left_ratios.items():
            features[f'left_{key}'] = value
        for key, value in right_ratios.items():
            features[f'right_{key}'] = value
        
        # Calculate cadence in sliding windows for CV
        window_duration = 10.0  # seconds
        window_samples = int(window_duration * self.sampling_rate)
        cadences = []
        
        if len(data) >= window_samples:
            for i in range(0, len(data) - window_samples, window_samples // 2):
                window_start = i
                window_end = i + window_samples
                
                # Count steps in this window
                window_steps = 0
                for step_start, step_end in left_steps + right_steps:
                    if window_start <= step_start < window_end:
                        window_steps += 1
                
                # Convert to steps per minute
                cadence = (window_steps / window_duration) * 60
                if cadence > 0:
                    cadences.append(cadence)
        
        if len(cadences) > 1:
            features['cadence_cv'] = (np.std(cadences) / np.mean(cadences)) * 100
        else:
            features['cadence_cv'] = 0
        
        
        def extract_force_events(steps, sensor_data, total_force):
            heel_forces = []
            toe_forces = []
            
            for start, end in steps:
                step_duration = end - start
                
                # Heel-strike force: mean force in first 5% of step
                heel_window = max(1, int(0.05 * step_duration))
                if start + heel_window < len(total_force):
                    heel_force = np.mean(total_force[start:start + heel_window])
                    heel_forces.append(heel_force)
                
                # Toe-off force: mean force in last 5% of step
                toe_window = max(1, int(0.05 * step_duration))
                if end - toe_window >= 0:
                    toe_force = np.mean(total_force[end - toe_window:end])
                    toe_forces.append(toe_force)
            
            return np.array(heel_forces), np.array(toe_forces)
        
        # Extract force events for both feet
        left_heel_forces, left_toe_forces = extract_force_events(left_steps, left_sensors, left_total)
        right_heel_forces, right_toe_forces = extract_force_events(right_steps, right_sensors, right_total)
        
        # Add force event features (CV focus for PD detection)
        for foot, heel_forces, toe_forces in [('left', left_heel_forces, left_toe_forces),
                                            ('right', right_heel_forces, right_toe_forces)]:
            
            if len(heel_forces) > 0:
                features[f'{foot}_heel_strike_force_cv'] = (np.std(heel_forces) / np.mean(heel_forces)) * 100
            else:
                features[f'{foot}_heel_strike_force_cv'] = 0
            
            if len(toe_forces) > 0:
                features[f'{foot}_toe_off_force_cv'] = (np.std(toe_forces) / np.mean(toe_forces)) * 100
            else:
                features[f'{foot}_toe_off_force_cv'] = 0
        
        
        def estimate_lengths_biomechanical(stance_times, swing_times, stride_times):

            if len(stride_times) == 0:
                return np.array([]), np.array([])
            
            stride_lengths = []
            step_lengths = []

            for i, stride_time in enumerate(stride_times):
                base_length = 1.25 * stride_time / 1.0  # Normalize to typical stride
                
                # Adjust based on stance/swing ratio if available
                if i < len(stance_times) and i < len(swing_times) and swing_times[i] > 0:
                    stance_swing_ratio = stance_times[i] / swing_times[i]
                    # Normal ratio ~1.5-2.0; deviations suggest gait changes
                    ratio_factor = 1.0 + 0.1 * (stance_swing_ratio - 1.7) / 1.7
                    base_length *= max(0.5, min(1.5, ratio_factor))  # Constrain adjustment
                
                stride_lengths.append(max(0.1, base_length))  # Minimum realistic length
                step_lengths.append(max(0.05, base_length / 2))  # Step ≈ stride/2
            
            return np.array(stride_lengths), np.array(step_lengths)
        
        # Calculate improved length estimates
        left_stride_lengths, left_step_lengths = estimate_lengths_biomechanical(left_stance, left_swing, left_strides)
        right_stride_lengths, right_step_lengths = estimate_lengths_biomechanical(right_stance, right_swing, right_strides)
        
        for foot, stride_lengths, step_lengths in [('left', left_stride_lengths, left_step_lengths),
                                                 ('right', right_stride_lengths, right_step_lengths)]:
            
            if len(stride_lengths) > 0:
                features[f'{foot}_stride_length_cv'] = (np.std(stride_lengths) / np.mean(stride_lengths)) * 100
            else:
                features[f'{foot}_stride_length_cv'] = 0
            
            if len(step_lengths) > 0:
                features[f'{foot}_step_length_cv'] = (np.std(step_lengths) / np.mean(step_lengths)) * 100
            else:
                features[f'{foot}_step_length_cv'] = 0
        
        # Stride asymmetry (critical for unilateral PD symptoms)
        if len(left_strides) > 0 and len(right_strides) > 0:
            left_mean = np.mean(left_strides)
            right_mean = np.mean(right_strides)
            features['stride_asymmetry'] = abs(left_mean - right_mean) / ((left_mean + right_mean) / 2) * 100
        else:
            features['stride_asymmetry'] = 0
        
        # Force asymmetry
        left_force_mean = np.mean(left_total)
        right_force_mean = np.mean(right_total)
        if (left_force_mean + right_force_mean) > 0:
            features['force_asymmetry'] = abs(left_force_mean - right_force_mean) / (left_force_mean + right_force_mean) * 100
        else:
            features['force_asymmetry'] = 0

        # Calculate COP efficiently
        left_cop_x, left_cop_y = self.calculate_cop_optimized(left_sensors, self.left_sensors)
        right_cop_x, right_cop_y = self.calculate_cop_optimized(right_sensors, self.right_sensors)
        
        # COP path length and variability
        left_cop_path = np.sum(np.sqrt(np.diff(left_cop_x)**2 + np.diff(left_cop_y)**2))
        right_cop_path = np.sum(np.sqrt(np.diff(right_cop_x)**2 + np.diff(right_cop_y)**2))
        
        features['left_cop_path_length'] = left_cop_path
        features['right_cop_path_length'] = right_cop_path
        
        # COP variability
        features['left_cop_x_cv'] = (np.std(left_cop_x) / np.mean(np.abs(left_cop_x))) * 100 if np.mean(np.abs(left_cop_x)) > 0 else 0
        features['left_cop_y_cv'] = (np.std(left_cop_y) / np.mean(np.abs(left_cop_y))) * 100 if np.mean(np.abs(left_cop_y)) > 0 else 0
        features['right_cop_x_cv'] = (np.std(right_cop_x) / np.mean(np.abs(right_cop_x))) * 100 if np.mean(np.abs(right_cop_x)) > 0 else 0
        features['right_cop_y_cv'] = (np.std(right_cop_y) / np.mean(np.abs(right_cop_y))) * 100 if np.mean(np.abs(right_cop_y)) > 0 else 0
        
        # ============= FORCE VARIABILITY =============
        features['left_force_cv'] = (np.std(left_total) / np.mean(left_total)) * 100 if np.mean(left_total) > 0 else 0
        features['right_force_cv'] = (np.std(right_total) / np.mean(right_total)) * 100 if np.mean(right_total) > 0 else 0

        
        def get_dominant_frequency(signal_data):
            if len(signal_data) < 256:
                return 0
            
            signal_data = signal_data - np.mean(signal_data)
            n_fft = min(2048, len(signal_data))
            
            fft_vals = fft(signal_data[:n_fft])
            freqs = fftfreq(n_fft, 1/self.sampling_rate)
            psd = np.abs(fft_vals)**2
            
            # Find dominant frequency in gait range (0.5-3 Hz)
            gait_mask = (freqs >= 0.5) & (freqs <= 3.0)
            if np.any(gait_mask):
                dominant_idx = np.argmax(psd[gait_mask])
                return freqs[gait_mask][dominant_idx]
            return 0
        
        features['left_dominant_freq'] = get_dominant_frequency(left_total)
        features['right_dominant_freq'] = get_dominant_frequency(right_total)
        
        # ============= FREEZING INDEX =============
        def calculate_freezing_index_fast(signal_data):
            if len(signal_data) < 512:
                return 0
                
            freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=min(512, len(signal_data)//4))
            
            freeze_mask = (freqs >= 3) & (freqs <= 8)
            locomotor_mask = (freqs >= 0.5) & (freqs <= 3)
            
            freeze_power = np.sum(psd[freeze_mask])
            locomotor_power = np.sum(psd[locomotor_mask])
            
            return freeze_power / locomotor_power if locomotor_power > 0 else 0
        
        features['freezing_index'] = (calculate_freezing_index_fast(left_total) + 
                                    calculate_freezing_index_fast(right_total)) / 2
        
        # ============= ENTROPY FEATURES =============
        def sample_entropy_fast(signal_data, m=2, r=0.15):
            N = len(signal_data)
            if N < 50:
                return 0
            
            try:
                if N > 2000:
                    indices = np.linspace(0, N-1, 2000, dtype=int)
                    signal_data = signal_data[indices]
                    N = 2000
                
                if N < m + 1:
                    return 0
                
                patterns_m = []
                patterns_m1 = []
                
                for i in range(N - m):
                    if i + m < N:
                        patterns_m.append(signal_data[i:i+m])
                    if i + m + 1 < N:
                        patterns_m1.append(signal_data[i:i+m+1])
                
                if len(patterns_m) == 0 or len(patterns_m1) == 0:
                    return 0
                
                patterns_m = np.array(patterns_m)
                patterns_m1 = np.array(patterns_m1)
                
                tolerance = r * np.std(signal_data)
                if tolerance == 0:
                    return 0
                
                max_comparisons = min(100, len(patterns_m), len(patterns_m1))
                if max_comparisons < 2:
                    return 0
                
                comparison_indices = np.random.choice(
                    min(len(patterns_m), len(patterns_m1)), 
                    max_comparisons, 
                    replace=False
                )
                
                matches_m = 0
                matches_m1 = 0
                
                for i in comparison_indices:
                    if i < len(patterns_m):
                        dists_m = np.max(np.abs(patterns_m - patterns_m[i]), axis=1)
                        matches_m += np.sum(dists_m <= tolerance) - 1
                    
                    if i < len(patterns_m1):
                        dists_m1 = np.max(np.abs(patterns_m1 - patterns_m1[i]), axis=1)
                        matches_m1 += np.sum(dists_m1 <= tolerance) - 1
                
                if matches_m > 0 and matches_m1 > 0:
                    return -np.log(matches_m1 / matches_m)
                else:
                    return 0
                    
            except Exception as e:
                return 0
        
        features['left_sample_entropy'] = sample_entropy_fast(left_total)
        features['right_sample_entropy'] = sample_entropy_fast(right_total)

        
        # Overall instability index
        stride_instability = (features.get('left_stride_time_cv', 0) + features.get('right_stride_time_cv', 0)) / 2
        force_instability = (features.get('left_force_cv', 0) + features.get('right_force_cv', 0)) / 2
        
        features['overall_instability_index'] = (stride_instability + force_instability + 
                                               features.get('freezing_index', 0) * 10) / 3
        
        # Step count for reference
        features['step_count_total'] = len(left_steps) + len(right_steps)
        
        return features
    
    def extract_all_features(self, filepath: str) -> Optional[Dict[str, float]]:
        print(f"Processing: {os.path.basename(filepath)}")
        
        # Load data
        data = self.load_data(filepath)
        if data is None:
            return None
        
        # Parse metadata
        metadata = self.parse_filename(filepath)
        
        # Initialize with metadata
        features = {
            'filename': os.path.basename(filepath),
            'study': metadata['study'],
            'subject_type': metadata['subject_type'],
            'subject_number': int(metadata['subject_number']) if metadata['subject_number'].isdigit() else metadata['subject_number'],
            'walk_number': metadata['walk_number'],
            'is_patient': 1 if metadata['is_patient'] else 0,
            'is_dual_task': 1 if metadata['is_dual_task'] else 0,
            'total_duration': data[-1, 0] - data[0, 0]
        }
        
        # Extract corrected features
        try:
            extracted_features = self.extract_corrected_features(data)
            features.update(extracted_features)
            
            print(f"  ✅ Extracted {len(extracted_features)} corrected features")
            
        except Exception as e:
            print(f" Error extracting features: {e}")
            return None
        
        return features
    
    def process_multiple_files(self, file_paths: Union[List[str], str], output_csv: str = None) -> pd.DataFrame:
        
        # Handle directory input
        if isinstance(file_paths, str) and os.path.isdir(file_paths):
            directory_path = file_paths
            txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
            file_paths = [os.path.join(directory_path, f) for f in txt_files]
            print(f"Found {len(file_paths)} .txt files in directory")
        
        # Handle list of files
        elif isinstance(file_paths, list):
            valid_files = []
            for fp in file_paths:
                if os.path.exists(fp) and fp.endswith('.txt'):
                    valid_files.append(fp)
                else:
                    print(f"Warning: File not found: {fp}")
            file_paths = valid_files
        
        if not file_paths:
            print("No valid files found to process")
            return pd.DataFrame()
        
        all_features = []
        
        for i, filepath in enumerate(file_paths):
            print(f"\n[{i+1}/{len(file_paths)}]", end=" ")
            
            features = self.extract_all_features(filepath)
            if features is not None:
                all_features.append(features)
            
        print(f"\n" + "=" * 60)
        print(f"✅ Successfully processed {len(all_features)} files")
        
        # Create DataFrame
        if all_features:
            df = pd.DataFrame(all_features)
            
            # Reorder columns
            metadata_cols = ['filename', 'study', 'subject_type', 'subject_number', 
                           'walk_number', 'is_patient', 'is_dual_task', 'total_duration']
            feature_cols = [col for col in df.columns if col not in metadata_cols]
            df = df[metadata_cols + feature_cols]
            
            if output_csv:
                df.to_csv(output_csv, index=False)
                print(f"Features saved to: {output_csv}")
            
            print(f"📊 Dataset shape: {df.shape}")
            print(f"Corrected features: {len(feature_cols)}")
            
            return df
        else:
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    
    # Initialize corrected extractor
    extractor = CorrectedGaitFeatureExtractor(sampling_rate=100.0)
    
    # Directory processing example
    directory_path = "./Smaller_samples"  # Change this to your actual directory
    print(f"\nProcessing directory: {directory_path}")
    
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        print(f"Found {len(txt_files)} .txt files in directory")
        
        if len(txt_files) > 0:
            df_corrected = extractor.process_multiple_files(
                directory_path, 
                output_csv="gait_features.csv"
            )
            
            if not df_corrected.empty:
                print(f"Files processed: {len(df_corrected)}")
                print(f"Total features: {len(df_corrected.columns) - 8}")
                print(f"Saved as: gait_features.csv")
                
                # Show corrected CV measures
                cv_cols = [col for col in df_corrected.columns if '_cv' in col]
                print(f"CV measures: {len(cv_cols)}")
                
                # Show new features
                new_feature_keywords = ['ratio', 'heel_strike', 'toe_off']
                new_features = [col for col in df_corrected.columns 
                              if any(keyword in col for keyword in new_feature_keywords)]
                print(f"   🆕 New features added: {len(new_features)}")
                
        else:
            print(f"No .txt files found in directory: {directory_path}")
    else:
        print(f"Directory not found: {directory_path}")
        print("Update the directory_path variable with your actual path")