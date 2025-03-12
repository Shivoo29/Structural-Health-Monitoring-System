import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
import random

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample(has_crack=False, sample_rate=40000, duration=0.01):
    """
    Generate a synthetic ultrasonic signal sample
    
    Parameters:
    -----------
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
    # Time points
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Base frequency (40 kHz)
    base_freq = 40000
    
    # Create base signal (clean)
    signal = np.sin(2 * np.pi * base_freq * t)
    
    if has_crack:
        # Add effects that would occur with a crack
        
        # 1. Amplitude reduction (attenuation)
        attenuation = np.random.uniform(0.4, 0.8)
        signal = signal * attenuation
        
        # 2. Phase shift
        phase_shift = np.random.uniform(0.1, 0.5)
        signal_shifted = np.sin(2 * np.pi * base_freq * t + phase_shift)
        signal = signal + signal_shifted * np.random.uniform(0.1, 0.3)
        
        # 3. Reflections (echoes)
        reflection_delay = int(sample_rate * np.random.uniform(0.001, 0.003))
        if reflection_delay < len(signal):
            reflection = np.zeros_like(signal)
            reflection[reflection_delay:] = signal[:-reflection_delay] * np.random.uniform(0.2, 0.5)
            signal = signal + reflection
            
        # 4. Add harmonic distortion
        harmonic = np.sin(2 * np.pi * base_freq * 1.5 * t) * np.random.uniform(0.1, 0.3)
        signal = signal + harmonic
        
        # 5. Add random noise (more noise in cracked samples)
        noise = np.random.normal(0, 0.15, size=len(t))
    else:
        # Add minimal noise for intact samples
        noise = np.random.normal(0, 0.05, size=len(t))
    
    # Add noise to signal
    signal = signal + noise
    
    # Normalize to range roughly between -1 and 1
    signal = signal / np.max(np.abs(signal))
    
    # Calculate FFT for frequency domain analysis
    signal_fft = fft(signal)
    fft_magnitude = np.abs(signal_fft)[:len(signal)//2]
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(signal)//2]
    
    # Extract features
    features = {
        'has_crack': int(has_crack),
        'mean_amplitude': np.mean(np.abs(signal)),
        'std_amplitude': np.std(signal),
        'max_amplitude': np.max(np.abs(signal)),
        'min_amplitude': np.min(np.abs(signal)),
        'peak_frequency': frequencies[np.argmax(fft_magnitude)],
        'energy': np.sum(signal**2),
        'zero_crossings': np.sum(np.diff(np.signbit(signal).astype(int)) != 0),
        'main_freq_energy': fft_magnitude[np.argmin(np.abs(frequencies - base_freq))],
        'harmonic_ratio': np.sum(fft_magnitude[frequencies > base_freq]) / np.sum(fft_magnitude),
        'envelope_mean': np.mean(np.abs(signal)),
        'envelope_std': np.std(np.abs(signal)),
        'kurtosis': np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4),
        'skewness': np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3)
    }
    
    return signal, features

# Generate dataset
def generate_dataset(n_samples=1000, crack_ratio=0.5):
    """
    Generate a dataset of n_samples with the specified ratio of cracked samples
    
    Parameters:
    -----------
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
    
    # Material properties will affect signal propagation
    materials = ['Aluminum', 'Steel', 'Copper', 'Iron', 'Titanium']
    thicknesses = np.random.uniform(5, 50, n_samples)  # Thickness in mm
    
    # Generate samples for intact objects
    for i in range(n_normal_samples):
        signal, features = generate_sample(has_crack=False)
        features['material'] = random.choice(materials)
        features['thickness_mm'] = thicknesses[i]
        features['sample_id'] = f"intact_{i}"
        signals.append(signal)
        features_list.append(features)
    
    # Generate samples for objects with cracks
    for i in range(n_crack_samples):
        # For cracked samples, vary the crack properties
        crack_size = np.random.uniform(0.5, 10)  # crack size in mm
        crack_depth = np.random.uniform(0.1, 0.9)  # relative depth (fraction of thickness)
        
        signal, features = generate_sample(has_crack=True)
        features['material'] = random.choice(materials)
        features['thickness_mm'] = thicknesses[i + n_normal_samples]
        features['crack_size_mm'] = crack_size
        features['crack_depth_ratio'] = crack_depth
        features['sample_id'] = f"cracked_{i}"
        signals.append(signal)
        features_list.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(features_list)
    
    return df, signals

# Generate the dataset
df, signals = generate_dataset(n_samples=1000, crack_ratio=0.5)

# Save the dataset to CSV
df.to_csv('crack_detection_dataset.csv', index=False)

# Plot a few examples to visualize the difference
def plot_examples():
    plt.figure(figsize=(15, 10))
    
    # Get indices for intact and cracked samples
    intact_idx = np.where(df['has_crack'] == 0)[0]
    cracked_idx = np.where(df['has_crack'] == 1)[0]
    
    # Plot 3 intact samples
    for i in range(3):
        plt.subplot(2, 3, i+1)
        idx = np.random.choice(intact_idx)
        plt.plot(signals[idx])
        plt.title(f"Intact Sample (ID: {df.iloc[idx]['sample_id']})")
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
    
    # Plot 3 cracked samples
    for i in range(3):
        plt.subplot(2, 3, i+4)
        idx = np.random.choice(cracked_idx)
        plt.plot(signals[idx])
        plt.title(f"Cracked Sample (ID: {df.iloc[idx]['sample_id']})")
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig('crack_detection_examples.png')
    plt.close()

# Print information about the dataset
print(f"Dataset created with {len(df)} samples")
print(f"Number of cracked samples: {sum(df['has_crack'])}")
print(f"Number of intact samples: {len(df) - sum(df['has_crack'])}")
print("\nFeatures in the dataset:")
for col in df.columns:
    print(f"- {col}")

print("\nSample data (first 5 rows):")
print(df.head())

# Plot examples
plot_examples()

# Example ML model training code (commented out)
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Prepare data for ML
X = df.drop(['has_crack', 'sample_id', 'crack_size_mm', 'crack_depth_ratio'], axis=1, errors='ignore')
y = df['has_crack']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Feature importance
for feat, importance in zip(X.columns, model.feature_importances_):
    print(f"{feat}: {importance:.4f}")
"""