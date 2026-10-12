import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dotenv import load_dotenv

warnings.filterwarnings('ignore', category=FutureWarning)

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR', 'data')
TRAIN_FILE_NAME = os.getenv('TRAIN_FILE_NAME', 'KDDTrain+.txt')
TEST_FILE_NAME = os.getenv('TEST_FILE_NAME', 'KDDTest+.txt')
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1000'))
EPOCHS = int(os.getenv('EPOCHS', '10'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.5'))
HIDDEN_UNITS = int(os.getenv('HIDDEN_UNITS', '500'))
OUTPUT_UNITS = int(os.getenv('OUTPUT_UNITS', '24'))
RANDOM_STATE = 42

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
        return self.ct.fit_transform(df[FEATURE_NAMES]).astype(np.float32)

    def transform_x(self, df):
        return self.ct.transform(df[FEATURE_NAMES]).astype(np.float32)

    def encode_labels(self, labels):
        encoded = np.zeros(len(labels), dtype=np.int64)
        for i, label in enumerate(labels):
            clean_label = str(label).rstrip('.')
            encoded[i] = self.label_to_idx.get(clean_label, self.unknown_idx)
        return encoded


class SecurityAttackNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def train_ann(model, train_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'ANN Epoch {epoch + 1:02d}/{epochs:02d} - Loss: {epoch_loss:.4f}', flush=True)


def predict_ann(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.tensor(X[start:start + BATCH_SIZE], dtype=torch.float32)
            preds.append(torch.argmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(preds)


def calculate_metrics(y_true, y_pred):
    multi_acc = accuracy_score(y_true, y_pred) * 100
    multi_p, multi_r, multi_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    true_binary = y_true > 0
    pred_binary = y_pred > 0
    binary_acc = accuracy_score(true_binary, pred_binary) * 100
    binary_p, binary_r, binary_f1, _ = precision_recall_fscore_support(
        true_binary, pred_binary, average='binary', zero_division=0
    )

    return {
        'Multiclass Accuracy (%)': multi_acc,
        'Multiclass Precision': multi_p,
        'Multiclass Recall': multi_r,
        'Multiclass F1': multi_f1,
        'Binary Accuracy (%)': binary_acc,
        'Binary Precision': binary_p,
        'Binary Recall': binary_r,
        'Binary F1': binary_f1,
    }


def resolve_file_paths(base_dir, train_name, test_name):
    train_path, test_path = None, None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower() == train_name.lower():
                train_path = os.path.join(root, file)
            elif file.lower() == test_name.lower():
                test_path = os.path.join(root, file)

    # Fallback: only fill in whichever file is still missing
    if not train_path or not test_path:
        for root, _, files in os.walk(base_dir):
            for file in files:
                low = file.lower()
                if not low.endswith('.txt'):
                    continue
                if not train_path and 'train' in low:
                    train_path = os.path.join(root, file)
                elif not test_path and 'test' in low:
                    test_path = os.path.join(root, file)
    return train_path, test_path


def add_result(results, name, y_test, y_pred, train_seconds):
    row = {'Algorithm': name, 'Training Time (s)': train_seconds}
    row.update(calculate_metrics(y_test, y_pred))
    results.append(row)


def run_sklearn(results, name, model, X_train, y_train, X_test, y_test):
    """Fit one sklearn model; a failure is reported but never stops the run."""
    print(f'\n========== {name} ==========', flush=True)
    try:
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        y_pred = model.predict(X_test)
        add_result(results, name, y_test, y_pred, elapsed)
        print(f'{name} done in {elapsed:.1f}s', flush=True)
        # Save partial results after every model so nothing is lost on a later crash
        pd.DataFrame(results).to_csv('model_comparison_partial.csv', index=False)
    except Exception as e:
        print(f'{name} FAILED: {type(e).__name__}: {e}', flush=True)


if __name__ == '__main__':
    train_path, test_path = resolve_file_paths(DATA_DIR, TRAIN_FILE_NAME, TEST_FILE_NAME)

    if not train_path or not test_path:
        raise FileNotFoundError(
            f"Could not locate dataset files in '{DATA_DIR}'. "
            'Put KDDTrain+.txt and KDDTest+.txt there or configure DATA_DIR in .env.'
        )

    print(f'Loaded train file: {train_path}')
    print(f'Loaded test file : {test_path}')

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)

    # NSL-KDD files can contain an optional difficulty column.
    train_cols = FEATURE_NAMES + ['label'] + (['difficulty'] if train_df.shape[1] == 43 else [])
    test_cols = FEATURE_NAMES + ['label'] + (['difficulty'] if test_df.shape[1] == 43 else [])
    train_df.columns = train_cols
    test_df.columns = test_cols

    preprocessor = NSLKDDPreprocessor(
        train_attack_types=train_df['label'].unique(),
        num_classes=OUTPUT_UNITS
    )

    X_train = preprocessor.fit_transform_x(train_df)
    y_train = preprocessor.encode_labels(train_df['label'])
    X_test = preprocessor.transform_x(test_df)
    y_test = preprocessor.encode_labels(test_df['label'])

    print(f'\nPrepared data: X_train={X_train.shape}, X_test={X_test.shape}', flush=True)

    results = []

    # ------------------------------------------------------------
    # 1. Original ANN baseline
    # ------------------------------------------------------------
    print('\n========== 1. ANN ==========', flush=True)
    try:
        torch.manual_seed(RANDOM_STATE)
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            ),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        ann = SecurityAttackNN(
            input_dim=X_train.shape[1],
            hidden_dim=HIDDEN_UNITS,
            output_dim=OUTPUT_UNITS
        )

        start = time.perf_counter()
        train_ann(ann, train_loader, EPOCHS, LEARNING_RATE)
        ann_time = time.perf_counter() - start
        ann_pred = predict_ann(ann, X_test)
        add_result(results, 'ANN (original)', y_test, ann_pred, ann_time)
        print(f'ANN done in {ann_time:.1f}s', flush=True)
        pd.DataFrame(results).to_csv('model_comparison_partial.csv', index=False)
    except Exception as e:
        print(f'ANN FAILED: {type(e).__name__}: {e}', flush=True)

    # ------------------------------------------------------------
    # 2. Random Forest (n_jobs=-1 uses all cores)
    # ------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    run_sklearn(results, 'Random Forest', rf, X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------
    # 3. Decision Tree
    # ------------------------------------------------------------
    dt = DecisionTreeClassifier(
        criterion='gini',
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    run_sklearn(results, 'Decision Tree', dt, X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------
    # 4. Logistic Regression
    # 'multi_class' removed (deprecated/removed in newer scikit-learn;
    # multinomial is already the default). 'lbfgs' is faster and
    # converges better than 'saga' on this data.
    # ------------------------------------------------------------
    lr_model = LogisticRegression(
        max_iter=500,
        solver='lbfgs',
        random_state=RANDOM_STATE
    )
    run_sklearn(results, 'Logistic Regression', lr_model, X_train, y_train, X_test, y_test)

    # ------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------
    if not results:
        raise RuntimeError('No model finished successfully; see errors above.')

    comparison = pd.DataFrame(results)
    metric_cols = [c for c in comparison.columns if c != 'Algorithm']
    comparison[metric_cols] = comparison[metric_cols].round(4)

    print('\n\n================ MODEL COMPARISON ================')
    print(comparison.to_string(index=False))

    comparison.to_csv('model_comparison_results.csv', index=False)
    print('\nSaved comparison to: model_comparison_results.csv')

    print('\nInterpretation:')
    print('- ANN is your original neural-network baseline.')
    print('- Random Forest is an ensemble of decision trees and is often strong on tabular NIDS data.')
    print('- Decision Tree provides a simple, interpretable tree-based baseline.')
    print('- Logistic Regression provides a linear baseline.')
    print('- Binary metrics are the fairest comparison; multiclass is skewed by the "unknown" bucket.')
