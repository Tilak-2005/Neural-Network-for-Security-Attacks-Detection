"""
ANN (paper's original model) vs Decision Tree (best F1 in your CSV) on NSL-KDD.

Outputs (saved in ./plots):
  1_accuracy_side_by_side.png   ANN vs Decision Tree: accuracy / precision / recall / F1
  2_all_models_from_csv.png     all four models, read from model_comparison_results.csv
  3_ann_accuracy_vs_epoch.png   train/test accuracy per epoch (like Figure 2 in the paper)
  4_confusion_binary.png        Normal vs Attack confusion matrices
  5_confusion_category.png      Normal/DoS/Probe/R2L/U2R confusion matrices
  6_per_category_detection.png  % of each test category predicted correctly
  7_confusion_full_classes.png  full class-level confusion matrices
  test_predictions.csv          every test row: actual label vs ANN vs Decision Tree
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
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
RESULTS_CSV = os.getenv('RESULTS_CSV', 'model_comparison_results.csv')
PLOT_DIR = os.getenv('PLOT_DIR', 'plots')
SHOW_PLOTS = os.getenv('SHOW_PLOTS', '1') == '1'
RANDOM_STATE = 42

ANN_COLOR = '#d95f02'
DT_COLOR = '#1b9e77'

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

# Table 2 of the paper: attack -> broad category
CATEGORY_OF = {'normal': 'Normal'}
for _name in ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'mailbomb',
              'apache2', 'processtable', 'udpstorm']:
    CATEGORY_OF[_name] = 'DoS'
for _name in ['ipsweep', 'nmap', 'portsweep', 'satan', 'mscan', 'saint']:
    CATEGORY_OF[_name] = 'Probe'
for _name in ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy',
              'warezclient', 'warezmaster', 'sendmail', 'named', 'snmpgetattack',
              'snmpguess', 'xlock', 'xsnoop', 'worm']:
    CATEGORY_OF[_name] = 'R2L'
for _name in ['buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'httptunnel',
              'ps', 'sqlattack', 'xterm']:
    CATEGORY_OF[_name] = 'U2R'

TRUE_CATS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
PRED_CATS = TRUE_CATS + ['Unseen attack']


# ----------------------------------------------------------------------------
# Data preparation (same as the paper / main_compare.py)
# ----------------------------------------------------------------------------
class NSLKDDPreprocessor:
    def __init__(self, train_attack_types, num_classes=24):
        self.cat_cols = ['protocol_type', 'service', 'flag']
        self.num_cols = [c for c in FEATURE_NAMES if c not in self.cat_cols]
        self.num_classes = num_classes

        self.label_to_idx = {'normal': 0}
        idx = 1
        for attack in sorted(set(train_attack_types)):
            if attack != 'normal' and idx < (num_classes - 1):
                self.label_to_idx[attack] = idx
                idx += 1
        self.unknown_idx = num_classes - 1
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.idx_to_label[self.unknown_idx] = 'unseen_attack'

        try:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        except TypeError:
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        self.ct = ColumnTransformer([
            ('num', StandardScaler(), self.num_cols),
            ('cat', ohe, self.cat_cols),
        ])

    def fit_transform_x(self, df):
        return self.ct.fit_transform(df[FEATURE_NAMES]).astype(np.float32)

    def transform_x(self, df):
        return self.ct.transform(df[FEATURE_NAMES]).astype(np.float32)

    def encode_labels(self, labels):
        out = np.zeros(len(labels), dtype=np.int64)
        for i, label in enumerate(labels):
            out[i] = self.label_to_idx.get(str(label).rstrip('.'), self.unknown_idx)
        return out


class SecurityAttackNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def predict_ann(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, len(X), BATCH_SIZE):
            batch = torch.tensor(X[s:s + BATCH_SIZE], dtype=torch.float32)
            preds.append(torch.argmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_ann_with_history(model, loader, X_train, y_train, X_test, y_test):
    """Train the ANN and record train/test accuracy after every epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    history = {'loss': [], 'train_acc': [], 'test_acc': [], 'test_bin_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        tr_pred = predict_ann(model, X_train)
        te_pred = predict_ann(model, X_test)
        history['loss'].append(running / len(loader.dataset))
        history['train_acc'].append(accuracy_score(y_train, tr_pred) * 100)
        history['test_acc'].append(accuracy_score(y_test, te_pred) * 100)
        history['test_bin_acc'].append(accuracy_score(y_test > 0, te_pred > 0) * 100)
        print(f"ANN epoch {epoch + 1:02d}/{EPOCHS:02d} | loss {history['loss'][-1]:.4f} "
              f"| train acc {history['train_acc'][-1]:.2f}% "
              f"| test acc {history['test_acc'][-1]:.2f}% "
              f"| test binary acc {history['test_bin_acc'][-1]:.2f}%", flush=True)
    return history


def calculate_metrics(y_true, y_pred):
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    bp, br, bf, _ = precision_recall_fscore_support(
        y_true > 0, y_pred > 0, average='binary', zero_division=0)
    return {
        'Multiclass Accuracy (%)': accuracy_score(y_true, y_pred) * 100,
        'Multiclass Precision': mp, 'Multiclass Recall': mr, 'Multiclass F1': mf,
        'Binary Accuracy (%)': accuracy_score(y_true > 0, y_pred > 0) * 100,
        'Binary Precision': bp, 'Binary Recall': br, 'Binary F1': bf,
    }


def resolve_file_paths(base_dir, train_name, test_name):
    train_path, test_path = None, None
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower() == train_name.lower():
                train_path = os.path.join(root, f)
            elif f.lower() == test_name.lower():
                test_path = os.path.join(root, f)
    if not train_path or not test_path:
        for root, _, files in os.walk(base_dir):
            for f in files:
                low = f.lower()
                if not low.endswith('.txt'):
                    continue
                if not train_path and 'train' in low:
                    train_path = os.path.join(root, f)
                elif not test_path and 'test' in low:
                    test_path = os.path.join(root, f)
    return train_path, test_path


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')


def annotate_bars(ax, bars, fmt='{:.1f}'):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                    textcoords='offset points')


def plot_accuracy_side_by_side(m_ann, m_dt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    acc_keys = ['Multiclass Accuracy (%)', 'Binary Accuracy (%)']
    x = np.arange(len(acc_keys))
    w = 0.36
    b1 = axes[0].bar(x - w / 2, [m_ann[k] for k in acc_keys], w, label='ANN (paper)', color=ANN_COLOR)
    b2 = axes[0].bar(x + w / 2, [m_dt[k] for k in acc_keys], w, label='Decision Tree', color=DT_COLOR)
    annotate_bars(axes[0], b1)
    annotate_bars(axes[0], b2)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Multiclass\n(24 classes)', 'Binary\n(Normal vs Attack)'])
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel('Accuracy on test data (%)')
    axes[0].set_title('Test accuracy')
    axes[0].legend()

    f_keys = ['Multiclass F1', 'Binary Precision', 'Binary Recall', 'Binary F1']
    x = np.arange(len(f_keys))
    b1 = axes[1].bar(x - w / 2, [m_ann[k] for k in f_keys], w, label='ANN (paper)', color=ANN_COLOR)
    b2 = axes[1].bar(x + w / 2, [m_dt[k] for k in f_keys], w, label='Decision Tree', color=DT_COLOR)
    annotate_bars(axes[1], b1, '{:.3f}')
    annotate_bars(axes[1], b2, '{:.3f}')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([k.replace(' ', '\n', 1) for k in f_keys])
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title('F1 / precision / recall')
    axes[1].legend()

    fig.suptitle('ANN vs Decision Tree on NSL-KDD test set', fontsize=13)
    fig.tight_layout()
    save(fig, '1_accuracy_side_by_side.png')


def plot_all_models_from_csv():
    if not os.path.exists(RESULTS_CSV):
        print(f'{RESULTS_CSV} not found, skipping all-models chart.')
        return
    df = pd.read_csv(RESULTS_CSV)
    cols = ['Multiclass Accuracy (%)', 'Binary Accuracy (%)']
    f1_cols = ['Multiclass F1', 'Binary F1']
    colors = ['#d95f02', '#7570b3', '#1b9e77', '#e7298a']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, keys, ylim, fmt, title in [
        (axes[0], cols, 105, '{:.1f}', 'Accuracy (%)'),
        (axes[1], f1_cols, 1.1, '{:.3f}', 'F1 score'),
    ]:
        x = np.arange(len(keys))
        w = 0.8 / len(df)
        for i, row in df.iterrows():
            bars = ax.bar(x + (i - (len(df) - 1) / 2) * w, [row[k] for k in keys], w,
                          label=row['Algorithm'], color=colors[i % len(colors)])
            annotate_bars(ax, bars, fmt)
        ax.set_xticks(x)
        ax.set_xticklabels([k.replace(' (%)', '') for k in keys])
        ax.set_ylim(0, ylim)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle('All four models (from model_comparison_results.csv)', fontsize=13)
    fig.tight_layout()
    save(fig, '2_all_models_from_csv.png')


def plot_ann_history(h):
    ep = np.arange(1, len(h['loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(ep, h['train_acc'], color='red', marker='o')
    axes[0].set_title('(a) TRAIN accuracy (24-class)')
    axes[1].plot(ep, h['test_acc'], color='blue', marker='o', label='24-class')
    axes[1].plot(ep, h['test_bin_acc'], color='purple', marker='s', label='Normal vs Attack')
    axes[1].set_title('(b) TEST accuracy')
    axes[1].legend()
    axes[2].plot(ep, h['loss'], color='black', marker='o')
    axes[2].set_title('Training loss')
    for ax in axes:
        ax.set_xlabel('epoch')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('accuracy (%)')
    fig.suptitle('ANN accuracy vs epoch (cf. Figure 2 of the paper)', fontsize=13)
    fig.tight_layout()
    save(fig, '3_ann_accuracy_vs_epoch.png')


def draw_cm(ax, cm, row_labels, col_labels, title, normalize=True, fontsize=9):
    shown = cm.astype(float)
    if normalize:
        sums = shown.sum(axis=1, keepdims=True)
        shown = np.divide(shown, sums, out=np.zeros_like(shown), where=sums > 0)
    im = ax.imshow(shown, cmap='Blues', vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=fontsize)
    ax.set_yticklabels(row_labels, fontsize=fontsize)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    thresh = shown.max() * 0.6 if shown.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0 and cm.shape[0] > 6:
                continue
            txt = f'{cm[i, j]}\n({shown[i, j] * 100:.0f}%)' if normalize and cm.shape[0] <= 6 \
                else f'{cm[i, j]}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize - 1,
                    color='white' if shown[i, j] > thresh else 'black')
    return im


def plot_confusion_binary(y_true, ann_pred, dt_pred):
    labels = ['Normal', 'Attack']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, pred, name in [(axes[0], ann_pred, 'ANN (paper)'),
                           (axes[1], dt_pred, 'Decision Tree')]:
        cm = confusion_matrix(y_true > 0, pred > 0, labels=[False, True])
        acc = accuracy_score(y_true > 0, pred > 0) * 100
        draw_cm(ax, cm, labels, labels, f'{name}\nbinary accuracy {acc:.2f}%', fontsize=11)
    fig.suptitle('Confusion matrix: Normal vs Attack (counts, % of actual row)', fontsize=13)
    fig.tight_layout()
    save(fig, '4_confusion_binary.png')


def category_matrix(true_cats, pred_cats):
    cm = np.zeros((len(TRUE_CATS), len(PRED_CATS)), dtype=int)
    for t, p in zip(true_cats, pred_cats):
        cm[TRUE_CATS.index(t), PRED_CATS.index(p)] += 1
    return cm


def plot_confusion_category(true_cats, ann_cats, dt_cats):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, pred, name in [(axes[0], ann_cats, 'ANN (paper)'),
                           (axes[1], dt_cats, 'Decision Tree')]:
        cm = category_matrix(true_cats, pred)
        draw_cm(ax, cm, TRUE_CATS, PRED_CATS, name, fontsize=10)
    fig.suptitle('Confusion matrix by attack category (actual category from the raw label; '
                 '"Unseen attack" = model output for classes not seen in training)', fontsize=11)
    fig.tight_layout()
    save(fig, '5_confusion_category.png')


def plot_per_category(true_cats, ann_cats, dt_cats):
    true_cats = np.array(true_cats)
    ann_cats = np.array(ann_cats)
    dt_cats = np.array(dt_cats)
    ann_ok, dt_ok = [], []
    for c in TRUE_CATS:
        mask = true_cats == c
        if c == 'Normal':
            ann_ok.append((ann_cats[mask] == 'Normal').mean() * 100 if mask.any() else 0)
            dt_ok.append((dt_cats[mask] == 'Normal').mean() * 100 if mask.any() else 0)
        else:
            # attack is "detected" when the model predicts anything other than Normal
            ann_ok.append((ann_cats[mask] != 'Normal').mean() * 100 if mask.any() else 0)
            dt_ok.append((dt_cats[mask] != 'Normal').mean() * 100 if mask.any() else 0)
    counts = [int((true_cats == c).sum()) for c in TRUE_CATS]

    x = np.arange(len(TRUE_CATS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, ann_ok, w, label='ANN (paper)', color=ANN_COLOR)
    b2 = ax.bar(x + w / 2, dt_ok, w, label='Decision Tree', color=DT_COLOR)
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(TRUE_CATS, counts)])
    ax.set_ylim(0, 110)
    ax.set_ylabel('% of test records handled correctly')
    ax.set_title('Per-category result on the test set\n'
                 '(Normal = predicted normal; attacks = flagged as an attack)')
    ax.legend()
    fig.tight_layout()
    save(fig, '6_per_category_detection.png')


def plot_confusion_full(y_true, ann_pred, dt_pred, idx_to_label):
    present = sorted(set(y_true) | set(ann_pred) | set(dt_pred))
    names = [idx_to_label.get(i, str(i)) for i in present]
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    for ax, pred, name in [(axes[0], ann_pred, 'ANN (paper)'),
                           (axes[1], dt_pred, 'Decision Tree')]:
        cm = confusion_matrix(y_true, pred, labels=present)
        acc = accuracy_score(y_true, pred) * 100
        draw_cm(ax, cm, names, names, f'{name}\n24-class accuracy {acc:.2f}% (colour = row %)',
                fontsize=7)
    fig.suptitle('Full class-level confusion matrices (class 23 = unseen attacks)', fontsize=13)
    fig.tight_layout()
    save(fig, '7_confusion_full_classes.png')


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    train_path, test_path = resolve_file_paths(DATA_DIR, TRAIN_FILE_NAME, TEST_FILE_NAME)
    if not train_path or not test_path:
        raise FileNotFoundError(
            f"Could not locate dataset files in '{DATA_DIR}'. "
            'Put KDDTrain+.txt and KDDTest+.txt there or set DATA_DIR in .env.')
    print(f'Train file: {train_path}\nTest file : {test_path}')

    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    train_df.columns = FEATURE_NAMES + ['label'] + (['difficulty'] if train_df.shape[1] == 43 else [])
    test_df.columns = FEATURE_NAMES + ['label'] + (['difficulty'] if test_df.shape[1] == 43 else [])

    pre = NSLKDDPreprocessor(train_df['label'].unique(), num_classes=OUTPUT_UNITS)
    X_train = pre.fit_transform_x(train_df)
    y_train = pre.encode_labels(train_df['label'])
    X_test = pre.transform_x(test_df)
    y_test = pre.encode_labels(test_df['label'])
    print(f'X_train={X_train.shape}, X_test={X_test.shape}')

    # ---- ANN (original, from the paper) ----
    print('\n===== ANN =====')
    torch.manual_seed(RANDOM_STATE)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True)
    ann = SecurityAttackNN(X_train.shape[1], HIDDEN_UNITS, OUTPUT_UNITS)
    t0 = time.perf_counter()
    history = train_ann_with_history(ann, loader, X_train, y_train, X_test, y_test)
    print(f'ANN trained in {time.perf_counter() - t0:.1f}s')
    ann_pred = predict_ann(ann, X_test)

    # ---- Decision Tree (best F1 in the CSV) ----
    print('\n===== Decision Tree =====')
    dt = DecisionTreeClassifier(criterion='gini', random_state=RANDOM_STATE,
                                class_weight='balanced')
    t0 = time.perf_counter()
    dt.fit(X_train, y_train)
    print(f'Decision Tree trained in {time.perf_counter() - t0:.1f}s')
    dt_pred = dt.predict(X_test)

    m_ann = calculate_metrics(y_test, ann_pred)
    m_dt = calculate_metrics(y_test, dt_pred)
    summary = pd.DataFrame([{'Algorithm': 'ANN (paper)', **m_ann},
                            {'Algorithm': 'Decision Tree', **m_dt}]).round(4)
    print('\n=========== SIDE-BY-SIDE METRICS ===========')
    print(summary.to_string(index=False))
    os.makedirs(PLOT_DIR, exist_ok=True)
    summary.to_csv(os.path.join(PLOT_DIR, 'ann_vs_decision_tree_metrics.csv'), index=False)

    # ---- Show how each model predicts the given test data ----
    raw_labels = test_df['label'].astype(str).str.rstrip('.').values
    # Any raw test label missing from Table 2 is an unknown name; treat it as its own bucket.
    true_cats = []
    for l in raw_labels:
        cat = CATEGORY_OF.get(l)
        if cat is None:
            print(f'Warning: label "{l}" not in Table 2; counted as DoS for the category chart.')
            cat = 'DoS'
        true_cats.append(cat)

    def pred_to_cat(idx):
        label = pre.idx_to_label.get(int(idx), 'unseen_attack')
        return 'Unseen attack' if label == 'unseen_attack' else CATEGORY_OF.get(label, 'Unseen attack')

    ann_cats = [pred_to_cat(i) for i in ann_pred]
    dt_cats = [pred_to_cat(i) for i in dt_pred]

    pred_df = pd.DataFrame({
        'actual_label': raw_labels,
        'actual_category': true_cats,
        'actual_is_attack': y_test > 0,
        'ann_prediction': [pre.idx_to_label.get(int(i)) for i in ann_pred],
        'ann_correct_binary': (ann_pred > 0) == (y_test > 0),
        'dt_prediction': [pre.idx_to_label.get(int(i)) for i in dt_pred],
        'dt_correct_binary': (dt_pred > 0) == (y_test > 0),
    })
    pred_df.to_csv('test_predictions.csv', index=False)
    print('\nSaved every test prediction to test_predictions.csv')
    print('\nSample of test predictions (random 20 rows):')
    print(pred_df.sample(min(20, len(pred_df)), random_state=RANDOM_STATE)
          .to_string(index=False))

    # ---- Graphs ----
    plot_accuracy_side_by_side(m_ann, m_dt)
    plot_all_models_from_csv()
    plot_ann_history(history)
    plot_confusion_binary(y_test, ann_pred, dt_pred)
    plot_confusion_category(true_cats, ann_cats, dt_cats)
    plot_per_category(true_cats, ann_cats, dt_cats)
    plot_confusion_full(y_test, ann_pred, dt_pred, pre.idx_to_label)

    if SHOW_PLOTS:
        plt.show()
