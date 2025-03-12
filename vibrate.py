import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define material-specific properties
# These values are approximations of real physical properties
# that affect ultrasonic wave propagation
material_properties = {
    "PVC": {
        "density": 1400,              # kg/m³
        "sound_velocity": 2380,       # m/s
        "attenuation_coefficient": 40, # dB/m at 40kHz (higher for polymers)
        "acoustic_impedance": 3.27e6, # kg/(m²·s)
        "elasticity": 3.4e9,          # Young's modulus in Pa
        "noise_level": 0.08,          # Higher noise for PVC due to internal structure
        "crack_reflection_factor": 0.7 # Reflection coefficient at crack interfaces
    },
    "Iron": {
        "density": 7870,              # kg/m³
        "sound_velocity": 5120,       # m/s
        "attenuation_coefficient": 3,  # dB/m at 40kHz
        "acoustic_impedance": 40.3e6, # kg/(m²·s)
        "elasticity": 211e9,          # Young's modulus in Pa
        "noise_level": 0.04,          # Lower noise for homogeneous metal
        "crack_reflection_factor": 0.5 # Reflection coefficient at crack interfaces
    },
    "Aluminum": {
        "density": 2700,              # kg/m³
        "sound_velocity": 6320,       # m/s
        "attenuation_coefficient": 0.8, # dB/m at 40kHz
        "acoustic_impedance": 17e6,   # kg/(m²·s)
        "elasticity": 69e9,           # Young's modulus in Pa
        "noise_level": 0.03,          # Very clean signal transmission
        "crack_reflection_factor": 0.45 # Reflection coefficient at crack interfaces
    },
    "Steel": {
        "density": 7800,              # kg/m³
        "sound_velocity": 5900,       # m/s
        "attenuation_coefficient": 2.5, # dB/m at 40kHz
        "acoustic_impedance": 46e6,   # kg/(m²·s)
        "elasticity": 200e9,          # Young's modulus in Pa
        "noise_level": 0.035,         # Low noise for homogeneous metal
        "crack_reflection_factor": 0.48 # Reflection coefficient at crack interfaces
    },
    "Copper": {
        "density": 8960,              # kg/m³
        "sound_velocity": 4700,       # m/s
        "attenuation_coefficient": 4,  # dB/m at 40kHz
        "acoustic_impedance": 42e6,   # kg/(m²·s)
        "elasticity": 130e9,          # Young's modulus in Pa
        "noise_level": 0.05,          # Moderate noise for copper
        "crack_reflection_factor": 0.53 # Reflection coefficient at crack interfaces
    }
}

# Physically accurate crack modeling
def model_crack_effects(signal, t, material, thickness, crack_properties=None):
    """Model the effects of a crack on the ultrasonic signal in a physically accurate way"""
    
    # If no crack, just return the original signal with appropriate attenuation
    if crack_properties is None:
        # Calculate material-specific attenuation (simplified)
        distance = thickness / 1000  # Convert mm to m
        attenuation_factor = np.exp(-material['attenuation_coefficient'] * distance)
        return signal * attenuation_factor
    
    # Extract crack properties
    crack_size = crack_properties['size']
    crack_depth = crack_properties['depth']
    crack_orientation = crack_properties['orientation']
    
    # Calculate position of crack in time domain
    material_velocity = material['sound_velocity']  # m/s
    crack_position = (thickness * crack_depth) / 1000  # in meters
    crack_time = crack_position / material_velocity  # in seconds
    
    # Convert time to samples
    samples_per_second = len(t) / (t[-1] - t[0])
    crack_sample = int(crack_time * samples_per_second)
    
    # Avoid index errors
    if crack_sample >= len(signal):
        crack_sample = len(signal) - 1
    
    # Create modified signal
    modified_signal = np.copy(signal)
    
    # Calculate reflection coefficient based on crack size and orientation
    # Larger cracks and perpendicular orientation cause stronger reflections
    orientation_factor = np.sin(crack_orientation * np.pi/180)
    size_factor = np.clip(crack_size / 10, 0.1, 1.0)
    reflection_coefficient = material['crack_reflection_factor'] * size_factor * orientation_factor
    
    # Calculate transmission coefficient
    transmission_coefficient = 1 - reflection_coefficient
    
    # Model the crack effects:
    
    # 1. Reflected wave from crack
    reflected_signal = np.zeros_like(signal)
    if crack_sample > 0:
        reflected_signal[0:len(signal)-crack_sample] = reflection_coefficient * signal[crack_sample:]
    
    # 2. Transmitted wave (attenuated after crack)
    if crack_sample < len(modified_signal):
        modified_signal[crack_sample:] *= transmission_coefficient
    
    # 3. Additional attenuation due to crack scattering
    scattering_factor = 1 - (0.1 * size_factor) 
    if crack_sample < len(modified_signal):
        modified_signal[crack_sample:] *= scattering_factor
    
    # 4. Add phase shifts due to crack-induced diffraction
    phase_shift = np.pi/4 * size_factor
    if crack_sample < len(modified_signal):
        phase_component = np.sin(2 * np.pi * 40000 * t[crack_sample:] + phase_shift)
        modified_signal[crack_sample:] += 0.2 * reflection_coefficient * phase_component
    
    # 5. Add the reflection to the signal
    modified_signal += reflected_signal
    
    # 6. Apply material-specific attenuation
    distance = thickness / 1000  # Convert mm to m
    attenuation_factor = np.exp(-material['attenuation_coefficient'] * distance)
    modified_signal *= attenuation_factor
    
    return modified_signal

def generate_sample(material_name, thickness, has_crack=False, sample_rate=1000000, duration=0.0005):
    """
    Generate a synthetic ultrasonic signal sample for a specific material
    
    Parameters:
    -----------
    material_name: str
        Name of the material
    thickness: float
        Thickness of the object in mm
    has_crack: bool
        Whether the sample contains a crack
    sample_rate: int
        Sampling rate in Hz
    duration: float
        Duration of the sample in seconds
    
    Returns:
    --------
    time_series: numpy array
        Time series data of the ultrasonic signal
    features: dict
        Extracted features from the signal
    """
    # Get material properties
    material = material_properties[material_name]
    
    # Time points (high resolution for accurate simulation)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Base frequency (40 kHz as specified in the document)
    base_freq = 40000
    
    # Create base signal (clean pulse with multiple cycles)
    # Use a Gaussian modulated sinusoidal pulse to model ultrasonic transducer behavior
    num_cycles = 10
    center = duration / 2
    width = duration / (num_cycles * 4)
    
    # Create Gaussian envelope
    gaussian = np.exp(-0.5 * ((t - center) / width) ** 2)
    
    # Create base signal
    clean_signal = gaussian * np.sin(2 * np.pi * base_freq * t)
    
    # Crack properties if needed
    crack_props = None
    if has_crack:
        crack_props = {
            'size': np.random.uniform(0.5, 8.0),  # crack size in mm
            'depth': np.random.uniform(0.2, 0.8),  # relative depth (fraction of thickness)
            'orientation': np.random.uniform(45, 135)  # orientation in degrees (90° = perpendicular)
        }
    
    # Model signal propagation considering material properties and crack effects
    processed_signal = model_crack_effects(clean_signal, t, material, thickness, crack_props)
    
    # Add material-specific noise
    noise_level = material['noise_level'] * (1 + 0.5 * has_crack)  # More noise for cracked samples
    noise = np.random.normal(0, noise_level, size=len(t))
    processed_signal = processed_signal + noise
    
    # Extract meaningful features for ML model training
    
    # Time domain features
    signal_abs = np.abs(processed_signal)
    envelope = signal.hilbert(processed_signal)
    envelope_abs = np.abs(envelope)
    
    # Calculate FFT for frequency domain analysis
    signal_fft = fft(processed_signal)
    fft_magnitude = np.abs(signal_fft)[:len(processed_signal)//2]
    frequencies = np.fft.fftfreq(len(processed_signal), 1/sample_rate)[:len(processed_signal)//2]
    
    # Define feature extraction window sizes for temporal analysis
    early_window = len(processed_signal) // 3
    mid_window = early_window * 2
    
    # Extract features for ML model
    features = {
        'material': material_name,
        'thickness_mm': thickness,
        'has_crack': int(has_crack),
        
        # Time domain features
        'mean_amplitude': np.mean(signal_abs),
        'std_amplitude': np.std(processed_signal),
        'max_amplitude': np.max(signal_abs),
        'signal_energy': np.sum(processed_signal**2),
        'rms_amplitude': np.sqrt(np.mean(processed_signal**2)),
        
        # Envelope features
        'envelope_max': np.max(envelope_abs),
        'envelope_mean': np.mean(envelope_abs),
        'envelope_std': np.std(envelope_abs),
        
        # Statistical features
        'kurtosis': np.mean((processed_signal - np.mean(processed_signal))**4) / (np.std(processed_signal)**4) if np.std(processed_signal) > 0 else 0,
        'skewness': np.mean((processed_signal - np.mean(processed_signal))**3) / (np.std(processed_signal)**3) if np.std(processed_signal) > 0 else 0,
        
        # Frequency domain features
        'dominant_freq': frequencies[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0,
        'spectral_centroid': np.sum(frequencies * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0,
        'spectral_bandwidth': np.sqrt(np.sum(((frequencies - frequencies[np.argmax(fft_magnitude)])**2) * fft_magnitude) / np.sum(fft_magnitude)) if np.sum(fft_magnitude) > 0 else 0,
        'spectral_rolloff': np.percentile(fft_magnitude, 85) if len(fft_magnitude) > 0 else 0,
        
        # Temporal features - beginning, middle, end of signal
        'early_energy': np.sum(processed_signal[:early_window]**2),
        'mid_energy': np.sum(processed_signal[early_window:mid_window]**2),
        'late_energy': np.sum(processed_signal[mid_window:]**2),
        
        # Signal to noise ratio estimation
        'snr_estimate': np.mean(signal_abs)**2 / np.var(noise) if np.var(noise) > 0 else 0,
        
        # Ratio features
        'early_to_late_ratio': np.sum(processed_signal[:early_window]**2) / np.sum(processed_signal[mid_window:]**2) if np.sum(processed_signal[mid_window:]**2) > 0 else 0,
    }
    
    # Add crack-specific features if applicable
    if has_crack:
        features.update({
            'crack_size_mm': crack_props['size'],
            'crack_depth_ratio': crack_props['depth'],
            'crack_orientation_degrees': crack_props['orientation']
        })
    
    return processed_signal, features

def generate_material_dataset(material_name, n_samples=1000, crack_ratio=0.5):
    """
    Generate a dataset of n_samples for a specific material
    
    Parameters:
    -----------
    material_name: str
        Name of the material to generate samples for
    n_samples: int
        Number of samples to generate
    crack_ratio: float
        Ratio of cracked samples to generate (between 0 and 1)
    
    Returns:
    --------
    df: pandas.DataFrame
        DataFrame containing the extracted features
    signals: list
        List of raw time series signals
    """
    features_list = []
    signals = []
    
    n_crack_samples = int(n_samples * crack_ratio)
    n_normal_samples = n_samples - n_crack_samples
    
    # Generate realistic thickness values based on material
    if material_name == "PVC":
        # PVC pipes and sheets typically range from 1mm to 30mm
        thicknesses = np.random.uniform(1, 30, n_samples)
    elif material_name == "Aluminum":
        # Aluminum sheets and parts typically range from 0.5mm to 25mm
        thicknesses = np.random.uniform(0.5, 25, n_samples)
    elif material_name in ["Steel", "Iron"]:
        # Steel and iron typically range from 1mm to 50mm
        thicknesses = np.random.uniform(1, 50, n_samples)
    elif material_name == "Copper":
        # Copper typically ranges from 0.5mm to 20mm
        thicknesses = np.random.uniform(0.5, 20, n_samples)
    else:
        # Default range
        thicknesses = np.random.uniform(1, 40, n_samples)
    
    print(f"Generating {n_samples} samples for {material_name}...")
    
    # Generate samples for intact objects
    for i in range(n_normal_samples):
        signal, features = generate_sample(material_name, thicknesses[i], has_crack=False)
        features['sample_id'] = f"{material_name.lower()}_intact_{i}"
        signals.append(signal)
        features_list.append(features)
        
        if i % 100 == 0 and i > 0:
            print(f"  Generated {i}/{n_normal_samples} intact samples")
    
    # Generate samples for objects with cracks
    for i in range(n_crack_samples):
        signal, features = generate_sample(material_name, thicknesses[i + n_normal_samples], has_crack=True)
        features['sample_id'] = f"{material_name.lower()}_cracked_{i}"
        signals.append(signal)
        features_list.append(features)
        
        if i % 100 == 0 and i > 0:
            print(f"  Generated {i}/{n_crack_samples} cracked samples")
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    return df, signals

def plot_material_examples(material_name, df, signals, n_examples=2):
    """Plot examples of signals for a specific material"""
    plt.figure(figsize=(14, 8))
    
    # Get indices for intact and cracked samples
    intact_idx = df[df['has_crack'] == 0].index.tolist()
    cracked_idx = df[df['has_crack'] == 1].index.tolist()
    
    # Randomly select examples
    if len(intact_idx) >= n_examples and len(cracked_idx) >= n_examples:
        intact_samples = np.random.choice(intact_idx, n_examples, replace=False)
        cracked_samples = np.random.choice(cracked_idx, n_examples, replace=False)
        
        # Plot intact samples
        for i, idx in enumerate(intact_samples):
            plt.subplot(2, n_examples, i+1)
            plt.plot(signals[idx])
            thickness = df.iloc[idx]['thickness_mm']
            plt.title(f"{material_name} Intact\nThickness: {thickness:.1f}mm")
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
        
        # Plot cracked samples
        for i, idx in enumerate(cracked_samples):
            plt.subplot(2, n_examples, i+1+n_examples)
            plt.plot(signals[idx])
            thickness = df.iloc[idx]['thickness_mm']
            crack_size = df.iloc[idx]['crack_size_mm']
            crack_depth = df.iloc[idx]['crack_depth_ratio']
            plt.title(f"{material_name} Cracked\nThickness: {thickness:.1f}mm\n" +
                     f"Crack: {crack_size:.1f}mm at {crack_depth:.2f} depth")
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
    
    plt.tight_layout()
    return plt

# Function to generate datasets for all materials
def generate_all_material_datasets(n_samples_per_material=1000, output_dir='material_datasets'):
    """Generate and save datasets for all materials"""
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_materials = ["PVC", "Iron", "Aluminum", "Steel", "Copper"]
    
    # Create summary data for comparisons
    summary_rows = []
    
    # Generate dataset for each material
    for material_name in all_materials:
        print(f"\nGenerating dataset for {material_name}...")
        
        # Generate data
        df, signals = generate_material_dataset(material_name, n_samples=n_samples_per_material)
        
        # Save the dataset to CSV
        output_file = os.path.join(output_dir, f"{material_name.lower()}_crack_detection.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved dataset to {output_file}")
        
        # Plot examples
        plt_obj = plot_material_examples(material_name, df, signals)
        plot_file = os.path.join(output_dir, f"{material_name.lower()}_examples.png")
        plt_obj.savefig(plot_file)
        plt_obj.close()
        print(f"Saved example plots to {plot_file}")
        
        # Calculate summary statistics for intact and cracked samples
        intact_df = df[df['has_crack'] == 0]
        cracked_df = df[df['has_crack'] == 1]
        
        # Create summary for this material
        summary_rows.append({
            'material': material_name,
            'samples': len(df),
            'intact_samples': len(intact_df),
            'cracked_samples': len(cracked_df),
            'avg_thickness_mm': df['thickness_mm'].mean(),
            'intact_mean_amplitude': intact_df['mean_amplitude'].mean(),
            'cracked_mean_amplitude': cracked_df['mean_amplitude'].mean(),
            'intact_signal_energy': intact_df['signal_energy'].mean(),
            'cracked_signal_energy': cracked_df['signal_energy'].mean(),
            'intact_early_to_late_ratio': intact_df['early_to_late_ratio'].mean(),
            'cracked_early_to_late_ratio': cracked_df['early_to_late_ratio'].mean(),
        })
    
    # Create and save summary dataframe
    summary_df = pd.DataFrame(summary_rows)
    summary_file = os.path.join(output_dir, "dataset_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary statistics to {summary_file}")
    
    return summary_df

# Example ML model training code for a single material
def train_material_model(material_name, data_dir='material_datasets'):
    """
    Train an ML model for a specific material and evaluate its performance
    
    This function demonstrates how to use the generated datasets for training
    a machine learning model to detect cracks in a specific material.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    
    # Load the dataset
    file_path = os.path.join(data_dir, f"{material_name.lower()}_crack_detection.csv")
    df = pd.read_csv(file_path)
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['has_crack', 'sample_id', 'material', 
                                               'crack_size_mm', 'crack_depth_ratio', 
                                               'crack_orientation_degrees']]
    
    X = df[feature_cols]
    y = df['has_crack']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    
    print(f"\nModel Evaluation for {material_name}:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = sorted(zip(feature_cols, model.feature_importances_), 
                              key=lambda x: x[1], reverse=True)
    print("\nTop 5 Most Important Features:")
    for feat, importance in feature_importance[:5]:
        print(f"{feat}: {importance:.4f}")
    
    return model, scaler, feature_importance

# Generate datasets for all materials
if __name__ == "__main__":
    samples_per_material = 1000  # Adjust as needed
    output_directory = 'material_datasets'
    
    summary = generate_all_material_datasets(n_samples_per_material=samples_per_material, 
                                           output_dir=output_directory)
    
    print("\nDataset generation complete! Summary of generated datasets:")
    print(summary)
    
    # Uncomment to train and evaluate a model for a specific material
    """
    material_to_model = "Aluminum"  # Change this to model a different material
    model, scaler, importance = train_material_model(material_to_model, output_directory)
    """