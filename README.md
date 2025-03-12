# Structural Health Monitoring System

## Overview

This project focuses on detecting internal cracks in solid metallic objects using ultrasonic vibrations. By transmitting ultrasonic signals through a material and analyzing the received data, the system identifies structural anomalies indicative of cracks.

## Features

- **Ultrasonic Signal Generation**: Produces 40 kHz ultrasonic vibrations using piezoelectric discs.
- **Signal Reception and Amplification**: Captures transmitted signals and amplifies them for analysis.
- **Data Analysis**: Processes signals to detect anomalies caused by internal cracks.
- **Visualization**: Provides graphical representations of signal patterns to differentiate between intact and cracked samples.

## Repository Contents

- `material_datasets/`: Contains datasets related to various materials used in testing.
- `crack_detection_dataset.csv`: Dataset with extracted features from ultrasonic signals.
- `crack_detection_examples.png`: Visual examples of ultrasonic signals for both intact and cracked samples.
- `vibrate.py` and `vibrator.py`: Scripts for generating and processing ultrasonic vibrations.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `scipy`

Install the necessary libraries using:

```bash
pip install numpy pandas matplotlib scipy
```

### Running the Code

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/05Ashish/Structural-Health-Monitoring-System.git
   cd Structural-Health-Monitoring-System
   ```

2. **Generate Dataset**:

   Run `vibrate.py` to create the ultrasonic signal dataset:

   ```bash
   python vibrate.py
   ```

   This will produce `crack_detection_dataset.csv` and `crack_detection_examples.png`.

3. **Analyze Data**:

   Use the generated dataset to train machine learning models or perform further analysis as needed.

## Usage

- **Data Generation**: Modify `vibrate.py` to adjust parameters like sample size or crack characteristics.
- **Visualization**: Refer to `crack_detection_examples.png` for sample signal patterns.
- **Machine Learning**: Utilize `crack_detection_dataset.csv` to train models for automated crack detection.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the MIT License. 