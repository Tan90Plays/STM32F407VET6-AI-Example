<<<<<<< HEAD
import wfdb
import numpy as np
import os
from typing import Literal, Optional, List


def _list_local_records(data_path: str) -> List[str]:
    record_bases = []
    for root, _, files in os.walk(data_path):
        for filename in files:
            if filename.lower().endswith('.hea'):
                record_base = os.path.join(root, filename[:-4])
                record_bases.append(record_base)
    record_bases.sort()
    return record_bases


def _load_record(record_ref: str, source: Literal['local', 'network'], db_name: str):
    if source == 'network':
        return wfdb.rdrecord(record_ref, pn_dir=db_name)
    return wfdb.rdrecord(record_ref)


def prepare_data(
    source: Literal['local', 'network'] = 'network',
    db_name: str = 'mitdb',
    data_path: Optional[str] = None,
    sample_length: int = 360,
    samples_per_class: int = 5000,
):
    ecg_data = []
    sine_data = []
    noise_data = []

    # 1. Load ECG data
    if source == 'network':
        record_refs = wfdb.get_record_list(db_name)
        print(f"Loading ECG data from network ({db_name}), {len(record_refs)} records...")
    else:
        if not data_path:
            raise ValueError("data_path is required when source='local'")
        record_refs = _list_local_records(data_path)
        print(f"Loading ECG data from local folder ({data_path}), {len(record_refs)} records...")
    
    for record_ref in record_refs:
        if len(ecg_data) >= samples_per_class:
            break
        
        try:
            # Read record (both channels)
            record = _load_record(record_ref, source=source, db_name=db_name)
            signal = record.p_signal[:, 0] # Use channel 0
            
            # Segment the signal
            for i in range(0, len(signal) - sample_length, sample_length):
                segment = signal[i:i+sample_length]
                # Simple normalization to [0, 1]
                min_val = np.min(segment)
                max_val = np.max(segment)
                if max_val > min_val:
                    segment = (segment - min_val) / (max_val - min_val)
                    ecg_data.append(segment)
                
                if len(ecg_data) >= samples_per_class:
                    break
        except Exception as e:
            print(f"Error loading record {record_ref}: {e}")
            continue

    ecg_data = np.array(ecg_data[:samples_per_class])
    print(f"Extracted {len(ecg_data)} ECG segments.")

    # 2. Generate Sine data
    print("Generating Sine data...")
    t = np.linspace(0, 1, sample_length)
    for _ in range(samples_per_class):
        freq = np.random.uniform(0.5, 40.0) # 0.5Hz to 40Hz
        phase = np.random.uniform(0, 2 * np.pi)
        sine_wave = np.sin(2 * np.pi * freq * t + phase)
        # Normalize to [0, 1]
        sine_wave = (sine_wave + 1) / 2.0
        # Add a tiny bit of noise to make it realistic
        sine_wave += np.random.normal(0, 0.01, sample_length)
        sine_data.append(sine_wave)
    
    sine_data = np.array(sine_data)

    # 3. Generate Noise data
    print("Generating Noise data...")
    for _ in range(samples_per_class):
        noise = np.random.normal(0.5, 0.15, sample_length) # Mean 0.5, Std 0.15
        noise = np.clip(noise, 0, 1) # Keep in [0, 1]
        noise_data.append(noise)
    
    noise_data = np.array(noise_data)

    # Combine all data
    X = np.concatenate([ecg_data, sine_data, noise_data], axis=0)
    y = np.concatenate([
        np.zeros(len(ecg_data)), # 0: ECG
        np.ones(len(sine_data)),  # 1: Sine
        2 * np.ones(len(noise_data)) # 2: Noise
    ], axis=0)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split into train, val, test (70, 15, 15)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Reshape for CNN: (samples, length, 1)
    X_train = X_train.reshape(-1, sample_length, 1)
    X_val = X_val.reshape(-1, sample_length, 1)
    X_test = X_test.reshape(-1, sample_length, 1)

    # Save to disk
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    print("Data preparation complete.")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    config = {
        'source': 'network',# 'network' 或 'local'
        'db_name': 'mitdb',
        'data_path': r"mit-bih-arrhythmia-database-1.0.0",# 本地模式需要；网络模式可留着不影响
        'sample_length': 360,
        'samples_per_class': 5000,
    }

    prepare_data(**config)
=======
import wfdb
import numpy as np
import os
from typing import Literal, Optional, List


def _list_local_records(data_path: str) -> List[str]:
    record_bases = []
    for root, _, files in os.walk(data_path):
        for filename in files:
            if filename.lower().endswith('.hea'):
                record_base = os.path.join(root, filename[:-4])
                record_bases.append(record_base)
    record_bases.sort()
    return record_bases


def _load_record(record_ref: str, source: Literal['local', 'network'], db_name: str):
    if source == 'network':
        return wfdb.rdrecord(record_ref, pn_dir=db_name)
    return wfdb.rdrecord(record_ref)


def prepare_data(
    source: Literal['local', 'network'] = 'network',
    db_name: str = 'mitdb',
    data_path: Optional[str] = None,
    sample_length: int = 360,
    samples_per_class: int = 5000,
):
    ecg_data = []
    sine_data = []
    noise_data = []

    # 1. Load ECG data
    if source == 'network':
        record_refs = wfdb.get_record_list(db_name)
        print(f"Loading ECG data from network ({db_name}), {len(record_refs)} records...")
    else:
        if not data_path:
            raise ValueError("data_path is required when source='local'")
        record_refs = _list_local_records(data_path)
        print(f"Loading ECG data from local folder ({data_path}), {len(record_refs)} records...")
    
    for record_ref in record_refs:
        if len(ecg_data) >= samples_per_class:
            break
        
        try:
            # Read record (both channels)
            record = _load_record(record_ref, source=source, db_name=db_name)
            signal = record.p_signal[:, 0] # Use channel 0
            
            # Segment the signal
            for i in range(0, len(signal) - sample_length, sample_length):
                segment = signal[i:i+sample_length]
                # Simple normalization to [0, 1]
                min_val = np.min(segment)
                max_val = np.max(segment)
                if max_val > min_val:
                    segment = (segment - min_val) / (max_val - min_val)
                    ecg_data.append(segment)
                
                if len(ecg_data) >= samples_per_class:
                    break
        except Exception as e:
            print(f"Error loading record {record_ref}: {e}")
            continue

    ecg_data = np.array(ecg_data[:samples_per_class])
    print(f"Extracted {len(ecg_data)} ECG segments.")

    # 2. Generate Sine data
    print("Generating Sine data...")
    t = np.linspace(0, 1, sample_length)
    for _ in range(samples_per_class):
        freq = np.random.uniform(0.5, 40.0) # 0.5Hz to 40Hz
        phase = np.random.uniform(0, 2 * np.pi)
        sine_wave = np.sin(2 * np.pi * freq * t + phase)
        # Normalize to [0, 1]
        sine_wave = (sine_wave + 1) / 2.0
        # Add a tiny bit of noise to make it realistic
        sine_wave += np.random.normal(0, 0.01, sample_length)
        sine_data.append(sine_wave)
    
    sine_data = np.array(sine_data)

    # 3. Generate Noise data
    print("Generating Noise data...")
    for _ in range(samples_per_class):
        noise = np.random.normal(0.5, 0.15, sample_length) # Mean 0.5, Std 0.15
        noise = np.clip(noise, 0, 1) # Keep in [0, 1]
        noise_data.append(noise)
    
    noise_data = np.array(noise_data)

    # Combine all data
    X = np.concatenate([ecg_data, sine_data, noise_data], axis=0)
    y = np.concatenate([
        np.zeros(len(ecg_data)), # 0: ECG
        np.ones(len(sine_data)),  # 1: Sine
        2 * np.ones(len(noise_data)) # 2: Noise
    ], axis=0)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split into train, val, test (70, 15, 15)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Reshape for CNN: (samples, length, 1)
    X_train = X_train.reshape(-1, sample_length, 1)
    X_val = X_val.reshape(-1, sample_length, 1)
    X_test = X_test.reshape(-1, sample_length, 1)

    # Save to disk
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    print("Data preparation complete.")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == "__main__":
    config = {
        'source': 'network',# 'network' 或 'local'
        'db_name': 'mitdb',
        'data_path': r"mit-bih-arrhythmia-database-1.0.0",# 本地模式需要；网络模式可留着不影响
        'sample_length': 360,
        'samples_per_class': 5000,
    }

    prepare_data(**config)
>>>>>>> bf07b64fccac035e7f8635d7251966020ce4728c
