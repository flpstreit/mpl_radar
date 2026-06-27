import os
import gc
import numpy as np
import h5py
from scipy import io
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight


# ============================================================================
# 1. CONFIGURAÇÃO
# ============================================================================

DATA_PATH = r'C:\Users\felip\Documents\Mestrado-ITA\ET-287\ProjetoGrupo\dataset'
OUTPUT_PATH = r'.\outputs'
os.makedirs(OUTPUT_PATH, exist_ok=True)

CLASS_NAMES_FULL = [
    'LFM', '2FSK', '4FSK', '8FSK', 'FM_Costas', '2PSK', '4PSK', '8PSK',
    'PM_Barker', 'PM_Huffman', 'PM_Frank', 'PM_P1', 'PM_P2', 'PM_P3',
    'PM_P4', 'PM_Px', 'PM_ZadoffChu', 'PM_T1', 'PM_T2', 'PM_T3', 'PM_T4',
    'NM', 'Noise'
]

IDX_CLASSES_PAPER = [0, 4, 5, 8, 10, 11, 12, 13, 14, 21, 22]
CLASS_NAMES_PAPER = [CLASS_NAMES_FULL[i] for i in IDX_CLASSES_PAPER]
NUM_CLASSES = len(CLASS_NAMES_PAPER)
REMAP_ORIGINAL_TO_SEQ = {old: new for new, old in enumerate(IDX_CLASSES_PAPER)}

SNR_MIN, SNR_MAX, SNR_STEP = -12, 20, 2
SNR_VALUES = list(range(SNR_MIN, SNR_MAX + 1, SNR_STEP))
ACCURACY_THRESHOLD = 0.90

MLP_HIDDEN_LAYERS = [512, 256, 128]
MLP_DROPOUT = 0.3
MLP_BATCH_SIZE = 256
MLP_EPOCHS = 100
MLP_LR_MIN = 1e-7
MLP_LR_MAX = 1e-3
EARLY_STOPPING_PATIENCE = 15

RANDOM_SEED = 42




def flatten_for_mlp(X: np.ndarray) -> np.ndarray:
    """Achata cada amostra de (1024, 2) para (2048,)."""
    return X.reshape(X.shape[0], -1)


def autocorrelation_feature(complex_signal: np.ndarray, max_lag: int = 64) -> np.ndarray:
    """
    Calcula a magnitude da autocorrelação do sinal complexo para os lags
    1..max_lag. Essa propriedade é justamente o que diferencia matematicamente
    os códigos polifásicos entre si (Frank, P1, P2, P3, P4) — cada um é
    derivado de uma matriz de fase com uma "assinatura" de autocorrelação
    própria, que a diferença de fase ponto-a-ponto não captura sozinha.

    Implementação vetorizada via FFT (Wiener–Khinchin), evitando loop
    explícito em lag para todas as N amostras.

    Parameters
    ----------
    complex_signal : np.ndarray
        Sinal complexo I + jQ, shape (N, L).
    max_lag : int
        Número de lags (deslocamentos) retornados.

    Returns
    -------
    np.ndarray
        Magnitude da autocorrelação normalizada, shape (N, max_lag).
    """
    N_samples, L = complex_signal.shape

    # Autocorrelação completa via FFT (zero-padding para evitar wrap-around)
    n_fft = 2 * L
    F = np.fft.fft(complex_signal, n=n_fft, axis=1)
    power_spectrum = F * np.conj(F)
    acf_full = np.fft.ifft(power_spectrum, axis=1).real  # shape (N, n_fft)

    # Pega os lags de 1 a max_lag (lag 0 fica de fora: é só a energia total)
    acf = acf_full[:, 1:max_lag + 1]

    # Normaliza pela autocorrelação no lag 0 (energia do sinal)
    acf_lag0 = acf_full[:, 0:1] + 1e-8
    acf_norm = np.abs(acf) / acf_lag0

    return acf_norm.astype('float32')


def extract_features(X: np.ndarray, max_lag: int = 64) -> np.ndarray:
    """
    Extrai features manuais do sinal I/Q bruto, dando à rede acesso direto
    a informações que o sinal achatado não expõe explicitamente:
      - Amplitude instantânea                   (1024 valores)
      - Diferença de fase (freq. instantânea)   (1023 valores)
      - Magnitude do espectro (FFT normalizada) (1024 valores)
      - Autocorrelação normalizada (lags 1..max_lag) (max_lag valores)

    A autocorrelação foi adicionada especificamente para ajudar a distinguir
    códigos de fase polifásicos (Frank, P1, P2, P3, P4), que compartilham
    estatísticas de amplitude/fase instantânea muito parecidas, mas têm
    propriedades de autocorrelação distintas por construção matemática.

    Substitui o uso de `flatten_for_mlp` quando se deseja uma representação
    mais informativa do sinal (recomendado).

    Parameters
    ----------
    X : np.ndarray
        Sinal bruto, shape (N, 1024, 2) — ainda NÃO achatado, NÃO normalizado.
    max_lag : int
        Número de lags de autocorrelação a incluir.

    Returns
    -------
    np.ndarray
        Features concatenadas, shape (N, 3071 + max_lag).
    """
    I, Q = X[:, :, 0], X[:, :, 1]
    complex_signal = I + 1j * Q

    amplitude = np.abs(complex_signal)
    phase = np.angle(complex_signal)
    phase_diff = np.diff(np.unwrap(phase, axis=1), axis=1)  # variação de fase

    fft_mag = np.abs(np.fft.fft(complex_signal, axis=1))
    fft_mag = fft_mag / (fft_mag.max(axis=1, keepdims=True) + 1e-8)

    acf = autocorrelation_feature(complex_signal, max_lag=max_lag)

    features = np.concatenate([
        amplitude,       # 1024
        phase_diff,      # 1023
        fft_mag,         # 1024
        acf,             # max_lag (default 64)
    ], axis=1)

    return features.astype('float32')


# ============================================================================
# 3. NORMALIZAÇÃO (Z-SCORE)
# ============================================================================

def compute_normalization_stats(X_train: np.ndarray):
    """
    Calcula média e desvio padrão a partir do conjunto de TREINO apenas
    (para evitar data leakage). Funciona tanto para dados achatados (2D)
    quanto para dados no formato (N, 1024, 2) — neste caso, calcula uma
    estatística por canal (I e Q separadamente).
    """
    if X_train.ndim == 3:
        # (N, 1024, 2) -> estatística por canal (I, Q)
        mean = X_train.mean(axis=(0, 1), keepdims=True)   # shape (1, 1, 2)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    else:
        # já achatado (N, 2048) -> estatística global por feature
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8

    return mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Aplica a normalização z-score: (X - mean) / std."""
    return (X - mean) / std


# ============================================================================
# 4. ANÁLISE EXPLORATÓRIA (EDA)
# ============================================================================

def plot_class_distribution(y: np.ndarray, save_path: str = None):
    classes, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar([CLASS_NAMES_PAPER[c] for c in classes], counts, color='steelblue')
    plt.xlabel('Modulação')
    plt.ylabel('Número de amostras')
    plt.title('Distribuição de classes no conjunto de dados')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_snr_distribution(snr: np.ndarray, save_path: str = None):
    values, counts = np.unique(snr, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar(values, counts, width=1.5, color='darkorange')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Número de amostras')
    plt.title('Distribuição de SNR no conjunto de dados')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_signal_examples(X: np.ndarray, y: np.ndarray, snr: np.ndarray,
                          snr_target: int = 10, save_path: str = None):
    """Plota um exemplo de sinal I/Q por modulação (espera X no formato (N,1024,2))."""
    n_classes_to_plot = len(CLASS_NAMES_PAPER)
    fig, axes = plt.subplots(n_classes_to_plot, 1, figsize=(10, 2 * n_classes_to_plot), sharex=True)

    for class_idx in range(n_classes_to_plot):
        mask = (y == class_idx) & (snr == snr_target)
        if not np.any(mask):
            mask = (y == class_idx)

        sample = X[mask][0]
        ax = axes[class_idx]

        ax.plot(sample[:, 0], label='I', linewidth=0.8)
        ax.plot(sample[:, 1], label='Q', linewidth=0.8)
        ax.set_ylabel(CLASS_NAMES_PAPER[class_idx], fontsize=8)
        ax.legend(loc='upper right', fontsize=6)

    axes[-1].set_xlabel('Amostra (tempo)')
    fig.suptitle(f'Exemplos de sinais I/Q por modulação (SNR ≈ {snr_target} dB)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_power_spectrum(X: np.ndarray, y: np.ndarray, snr: np.ndarray,
                         snr_target: int = 10, save_path: str = None):
    """Plota o espectro de potência (FFT do sinal complexo I+jQ) por modulação."""
    plt.figure(figsize=(12, 6))

    for class_idx in range(len(CLASS_NAMES_PAPER)):
        mask = (y == class_idx) & (snr == snr_target)
        if not np.any(mask):
            mask = (y == class_idx)

        sample = X[mask][0]
        complex_signal = sample[:, 0] + 1j * sample[:, 1]
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(complex_signal)))

        plt.plot(spectrum, label=CLASS_NAMES_PAPER[class_idx], linewidth=0.8)

    plt.xlabel('Frequência (bin)')
    plt.ylabel('Magnitude')
    plt.title(f'Espectro de potência por modulação (SNR ≈ {snr_target} dB)')
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_basic_stats(X: np.ndarray, y: np.ndarray):
    """Imprime média e desvio padrão por classe (espera X no formato (N,1024,2))."""
    print(f"{'Classe':<15} {'Média I':>10} {'Std I':>10} {'Média Q':>10} {'Std Q':>10}")
    for class_idx in range(len(CLASS_NAMES_PAPER)):
        mask = y == class_idx
        Xi = X[mask][:, :, 0]
        Xq = X[mask][:, :, 1]
        print(f"{CLASS_NAMES_PAPER[class_idx]:<15} "
              f"{Xi.mean():>10.4f} {Xi.std():>10.4f} "
              f"{Xq.mean():>10.4f} {Xq.std():>10.4f}")


def run_eda(X: np.ndarray, y: np.ndarray, snr: np.ndarray, save_figs: bool = True):
    """Executa toda a rotina de EDA. Espera X no formato (N, 1024, 2) (antes de achatar)."""
    out = OUTPUT_PATH if save_figs else None

    print('\n--- Estatísticas básicas por classe ---')
    print_basic_stats(X, y)

    print('\n--- Gerando gráficos de EDA ---')
    plot_class_distribution(y, save_path=os.path.join(out, 'eda_class_distribution.png') if out else None)
    plot_snr_distribution(snr, save_path=os.path.join(out, 'eda_snr_distribution.png') if out else None)
    plot_signal_examples(X, y, snr, save_path=os.path.join(out, 'eda_signal_examples.png') if out else None)
    plot_power_spectrum(X, y, snr, save_path=os.path.join(out, 'eda_power_spectrum.png') if out else None)


# ============================================================================
# 5. MODELO MLP
# ============================================================================

def build_mlp(input_dim: int = 2048,
              hidden_layers=None,
              dropout: float = None,
              num_classes: int = NUM_CLASSES) -> Sequential:
    """MLP: Input -> [Dense + ReLU + Dropout] x N -> Dense(softmax)."""
    hidden_layers = hidden_layers or MLP_HIDDEN_LAYERS
    dropout = dropout if dropout is not None else MLP_DROPOUT

    model = Sequential(name='MLP_AMC')
    model.add(Input(shape=(input_dim,)))

    for i, units in enumerate(hidden_layers):
        model.add(Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(BatchNormalization(name=f'bn_{i+1}'))
        model.add(Dropout(dropout, name=f'dropout_{i+1}'))

    model.add(Dense(num_classes, activation='softmax', name='output'))

    return model


def compile_model(model: Sequential, learning_rate: float = 1e-3):
    """Compila o modelo com otimizador Adam e cross-entropy categórica."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# 6. TREINAMENTO
# ============================================================================

def set_seeds(seed: int = RANDOM_SEED):
    """Fixa as seeds para reprodutibilidade."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


class CyclicalLR(tf.keras.callbacks.Callback):
    """Callback de Cyclical Learning Rate (triangular)."""

    def __init__(self, lr_min=MLP_LR_MIN, lr_max=MLP_LR_MAX, step_size=10):
        super().__init__()
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.step_size = step_size

    def on_epoch_begin(self, epoch, logs=None):
        cycle = np.floor(1 + epoch / (2 * self.step_size))
        x = np.abs(epoch / self.step_size - 2 * cycle + 1)
        lr = self.lr_min + (self.lr_max - self.lr_min) * max(0, (1 - x))
        self.model.optimizer.learning_rate.assign(lr)


def prepare_labels(y_train, y_val, num_classes: int = NUM_CLASSES):
    """Converte rótulos sequenciais (0..N-1) para one-hot."""
    Y_train = to_categorical(y_train, num_classes=num_classes)
    Y_val = to_categorical(y_val, num_classes=num_classes)
    return Y_train, Y_val


def train_mlp(X_train, y_train, X_val, y_val,
              input_dim: int = None,
              batch_size: int = MLP_BATCH_SIZE,
              epochs: int = MLP_EPOCHS,
              model_save_path: str = None,
              use_class_weight: bool = True):
    """
    Treina o MLP e retorna o modelo treinado e o histórico.
    X_train, X_val devem já estar achatados/transformados (extract_features
    ou flatten_for_mlp) E normalizados.

    Parameters
    ----------
    use_class_weight : bool
        Se True, calcula pesos balanceados por classe (útil quando algumas
        modulações são sistematicamente mais difíceis de classificar,
        como observado nas curvas de acurácia x SNR).
    """
    set_seeds()

    input_dim = input_dim or X_train.shape[1]
    Y_train, Y_val = prepare_labels(y_train, y_val)

    model = build_mlp(input_dim=input_dim)
    model = compile_model(model, learning_rate=MLP_LR_MIN)

    callbacks = [
        CyclicalLR(),
        EarlyStopping(monitor='val_accuracy', patience=EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True, verbose=1)
    ]

    class_weight_dict = None
    if use_class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
        print(f'Class weights: {class_weight_dict}')

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=2
    )

    model_save_path = model_save_path or os.path.join(OUTPUT_PATH, 'mlp_amc.keras')
    model.save(model_save_path)
    print(f'Modelo salvo em: {model_save_path}')

    return model, history


# ============================================================================
# 7. AVALIAÇÃO
# ============================================================================

def get_predictions(model, X_test):
    """Retorna as classes previstas (índices) para o conjunto de teste."""
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    return y_pred, y_prob


def overall_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES_PAPER, digits=4))


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           normalize_cm: bool = False, save_path: str = None):
    """Plota a matriz de confusão (similar às Figuras 5 e 8 do paper)."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize_cm:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=CLASS_NAMES_PAPER, yticklabels=CLASS_NAMES_PAPER,
                cbar_kws={'label': 'Frequência' if not normalize_cm else 'Proporção'})
    plt.xlabel('Predicted')
    plt.ylabel('True Label')
    plt.title('Matriz de Confusão - MLP')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    return cm


def accuracy_per_snr_general(y_true: np.ndarray, y_pred: np.ndarray,
                              snr: np.ndarray, snr_values=SNR_VALUES) -> dict:
    """Acurácia geral (todas as modulações juntas) para cada valor de SNR."""
    acc_by_snr = {}
    for s in snr_values:
        mask = snr == s
        if not np.any(mask):
            continue
        acc_by_snr[s] = float(np.mean(y_true[mask] == y_pred[mask]))
    return acc_by_snr


def accuracy_per_snr_per_class(y_true: np.ndarray, y_pred: np.ndarray,
                                snr: np.ndarray, snr_values=SNR_VALUES,
                                num_classes: int = NUM_CLASSES) -> dict:
    """Acurácia por SNR, separadamente para cada modulação."""
    acc_by_class_snr = {c: {} for c in range(num_classes)}

    for c in range(num_classes):
        class_mask = y_true == c
        for s in snr_values:
            mask = class_mask & (snr == s)
            if not np.any(mask):
                continue
            acc_by_class_snr[c][s] = float(np.mean(y_true[mask] == y_pred[mask]))

    return acc_by_class_snr


def plot_general_accuracy_vs_snr(acc_by_snr: dict, save_path: str = None):
    """Gráfico único: acurácia geral em função do SNR (Figuras 3 e 6 do paper)."""
    snrs = sorted(acc_by_snr.keys())
    accs = [acc_by_snr[s] for s in snrs]

    plt.figure(figsize=(8, 5))
    plt.plot(snrs, accs, marker='o', label='General Accuracy')
    plt.axhline(ACCURACY_THRESHOLD, color='gray', linestyle='--',
                label=f'{int(ACCURACY_THRESHOLD*100)}% threshold')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Acurácia Geral em função do SNR - MLP')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_accuracy_per_class_vs_snr(acc_by_class_snr: dict, save_path: str = None):
    """Gráfico único com uma curva de acurácia x SNR para cada modulação (Figuras 4 e 7)."""
    plt.figure(figsize=(10, 6))

    for class_idx, acc_dict in acc_by_class_snr.items():
        snrs = sorted(acc_dict.keys())
        accs = [acc_dict[s] for s in snrs]
        plt.plot(snrs, accs, marker='.', linewidth=1, label=CLASS_NAMES_PAPER[class_idx])

    plt.axhline(ACCURACY_THRESHOLD, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy (%)')
    plt.title('Acurácia por Modulação em função do SNR - MLP')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def min_snr_for_threshold(acc_by_snr: dict, threshold: float = ACCURACY_THRESHOLD):
    """Menor SNR a partir do qual a acurácia se mantém >= threshold."""
    snrs = sorted(acc_by_snr.keys())
    for s in snrs:
        if all(acc_by_snr[s2] >= threshold for s2 in snrs if s2 >= s):
            return s
    return None


def min_snr_per_class(acc_by_class_snr: dict, threshold: float = ACCURACY_THRESHOLD):
    """Para cada classe, retorna o SNR mínimo necessário para atingir o threshold."""
    results = {}
    for class_idx, acc_dict in acc_by_class_snr.items():
        snrs = sorted(acc_dict.keys())
        min_snr = None
        for s in snrs:
            if all(acc_dict[s2] >= threshold for s2 in snrs if s2 >= s):
                min_snr = s
                break
        results[CLASS_NAMES_PAPER[class_idx]] = min_snr
    return results


def run_evaluation(model, X_test, y_test, snr_test, save_figs: bool = True):
    """Executa toda a rotina de avaliação e imprime um resumo final."""
    out = OUTPUT_PATH if save_figs else None

    y_pred, _ = get_predictions(model, X_test)

    acc = overall_accuracy(y_test, y_pred)
    print(f'\nAcurácia geral no teste: {acc*100:.2f}%\n')

    print_classification_report(y_test, y_pred)

    plot_confusion_matrix(
        y_test, y_pred,
        save_path=os.path.join(out, 'confusion_matrix.png') if out else None
    )

    acc_by_snr = accuracy_per_snr_general(y_test, y_pred, snr_test)
    plot_general_accuracy_vs_snr(
        acc_by_snr,
        save_path=os.path.join(out, 'accuracy_vs_snr_general.png') if out else None
    )

    acc_by_class_snr = accuracy_per_snr_per_class(y_test, y_pred, snr_test)
    plot_accuracy_per_class_vs_snr(
        acc_by_class_snr,
        save_path=os.path.join(out, 'accuracy_vs_snr_per_class.png') if out else None
    )

    snr_min_general = min_snr_for_threshold(acc_by_snr)
    snr_min_per_class = min_snr_per_class(acc_by_class_snr)
    limiting_mod = max(
        (k for k, v in snr_min_per_class.items() if v is not None),
        key=lambda k: snr_min_per_class[k],
        default=None
    )

    print(f'\nSNR mínimo (geral) para {int(ACCURACY_THRESHOLD*100)}% de acurácia: {snr_min_general} dB')
    print(f'SNR mínimo por modulação: {snr_min_per_class}')
    print(f'Modulação limitante: {limiting_mod} '
          f'({snr_min_per_class.get(limiting_mod)} dB)')

    return {
        'y_pred': y_pred,
        'overall_accuracy': acc,
        'acc_by_snr': acc_by_snr,
        'acc_by_class_snr': acc_by_class_snr,
        'snr_min_general': snr_min_general,
        'snr_min_per_class': snr_min_per_class,
    }