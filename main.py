import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from dotenv import load_dotenv


load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', 'data')
TRAIN_FILE_NAME = os.getenv('TRAIN_FILE_NAME', 'KDDTrain+.txt')
TEST_FILE_NAME = os.getenv('TEST_FILE_NAME', 'KDDTest+.txt')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1000'))
EPOCHS = int(os.getenv('EPOCHS', '10'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.5'))
HIDDEN_UNITS = int(os.getenv('HIDDEN_UNITS', '500'))
OUTPUT_UNITS = int(os.getenv('OUTPUT_UNITS', '24'))

if os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'):
    os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
    os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')


FEATURE_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]


class NSLKDDPreprocessor:
    def __init__(self, train_attack_types, num_classes=24):
        self.cat_cols = ['protocol_type', 'service', 'flag']
        self.num_cols = [col for col in FEATURE_NAMES if col not in self.cat_cols]
        self.num_classes = num_classes
        
        self.known_attacks = list(set(train_attack_types))
        self.label_to_idx = {'normal': 0}
        
        idx = 1
        for attack in sorted(self.known_attacks):
            if attack != 'normal' and idx < (self.num_classes - 1):
                self.label_to_idx[attack] = idx
                idx += 1
                
        self.unknown_idx = self.num_classes - 1

        try:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        self.ct = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_cols),
                ('cat', ohe, self.cat_cols)
            ]
        )

    def fit_transform_x(self, df):
        return self.ct.fit_transform(df[FEATURE_NAMES])

    def transform_x(self, df):
        return self.ct.transform(df[FEATURE_NAMES])

    def encode_labels(self, labels):
        one_hot = np.zeros((len(labels), self.num_classes), dtype=np.float32)
        for i, label in enumerate(labels):
            clean_label = str(label).rstrip('.')
            idx = self.label_to_idx.get(clean_label, self.unknown_idx)
            one_hot[i, idx] = 1.0
        return one_hot


class SecurityAttackNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SecurityAttackNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, train_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            targets = torch.argmax(y_batch, dim=1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Training Loss: {epoch_loss:.4f}")

def evaluate_model(model, loader):
    model.eval()
    correct_multiclass = 0
    correct_binary = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            predictions = torch.argmax(outputs, dim=1)
            true_labels = torch.argmax(y_batch, dim=1)

            correct_multiclass += (predictions == true_labels).sum().item()

            pred_binary = (predictions > 0)
            true_binary = (true_labels > 0)
            correct_binary += (pred_binary == true_binary).sum().item()

            total += X_batch.size(0)

    acc_multi = (correct_multiclass / total) * 100
    acc_bin = (correct_binary / total) * 100
    return acc_multi, acc_bin

def resolve_file_paths(base_dir, train_name, test_name):
    train_path, test_path = None, None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower() == train_name.lower():
                train_path = os.path.join(root, file)
            elif file.lower() == test_name.lower():
                test_path = os.path.join(root, file)
    
    if not train_path or not test_path:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if 'train' in file.lower() and file.endswith('.txt'):
                    train_path = os.path.join(root, file)
                elif 'test' in file.lower() and file.endswith('.txt'):
                    test_path = os.path.join(root, file)
    return train_path, test_path


if __name__ == "__main__":
    train_path, test_path = resolve_file_paths(DATA_DIR, TRAIN_FILE_NAME, TEST_FILE_NAME)

    if not train_path or not test_path:
        raise FileNotFoundError(f"Could not locate dataset files in '{DATA_DIR}'. Check your .env file or dataset directory.")

    print(f"Loaded train file: {train_path}")
    print(f"Loaded test file : {test_path}")

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    cols = FEATURE_NAMES + ['label'] + (['difficulty'] if train_df.shape[1] == 43 else [])
    train_df.columns = cols
    test_df.columns = cols

    preprocessor = NSLKDDPreprocessor(train_attack_types=train_df['label'].unique(), num_classes=OUTPUT_UNITS)
    X_train_np = preprocessor.fit_transform_x(train_df)
    y_train_np = preprocessor.encode_labels(train_df['label'])

    X_test_np = preprocessor.transform_x(test_df)
    y_test_np = preprocessor.encode_labels(test_df['label'])

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_np, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_np, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    input_dim = X_train_np.shape[1]
    model = SecurityAttackNN(input_dim=input_dim, hidden_dim=HIDDEN_UNITS, output_dim=OUTPUT_UNITS)

    print(f"\nModel Configuration:")
    print(f"- Input Features: {input_dim}")
    print(f"- Hidden Units:   {HIDDEN_UNITS}")
    print(f"- Output Classes: {OUTPUT_UNITS}")
    print(f"- Learning Rate:  {LEARNING_RATE}")
    print(f"- Batch Size:     {BATCH_SIZE}")
    print(f"- Epochs:         {EPOCHS}\n")

    print("Starting Training...")
    train_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE)

    train_acc_multi, train_acc_bin = evaluate_model(model, train_loader)
    test_acc_multi, test_acc_bin = evaluate_model(model, test_loader)

    print("\n================ RESULTS ================")
    print(f"Train Multi-class Accuracy : {train_acc_multi:.2f}%")
    print(f"Train Binary Detection Acc : {train_acc_bin:.2f}%")
    print(f"Test Multi-class Accuracy  : {test_acc_multi:.2f}%")
    print(f"Test Binary Detection Acc  : {test_acc_bin:.2f}%")