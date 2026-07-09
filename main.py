import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models import Word2Vec

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
RANDOM_SEED = 42


def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Define sementes para tornar os resultados reproduzíveis."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def download_nltk_resources() -> None:
    """Baixa recursos básicos do NLTK, se necessário."""
    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            if resource == "stopwords":
                nltk.data.find("corpora/stopwords")
            elif resource == "punkt_tab":
                nltk.data.find("tokenizers/punkt_tab")
            else:
                nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download(resource)


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Carrega o dataset BBC com colunas 'category' e 'text'."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Arquivo de dados não encontrado: {csv_path}. "
            "Baixe o 'bbc-text.csv' do enunciado e coloque no mesmo diretório deste script."
        )
    df = pd.read_csv(csv_path)
    if not {"category", "text"}.issubset(df.columns):
        raise ValueError("CSV deve conter as colunas 'category' e 'text'.")
    return df


def preprocess_texts(texts: pd.Series) -> Tuple[list, list]:
    """
    Limpeza de textos: minúsculas, remoção de pontuação/stopwords e tokenização.
    """
    download_nltk_resources()

    stop_words = set(stopwords.words("english"))
    cleaned_texts = []
    tokenized_texts = []

    for doc in texts:
        tokens = word_tokenize(str(doc).lower())
        tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
        tokenized_texts.append(tokens)
        cleaned_texts.append(" ".join(tokens))

    return cleaned_texts, tokenized_texts


def train_word2vec_model(
    tokenized_texts_train: List[List[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 1,
) -> Word2Vec:
    """Treina Word2Vec com Gensim no conjunto de treino."""
    return Word2Vec(
        sentences=tokenized_texts_train,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=RANDOM_SEED,
    )


def build_embedding_matrix(
    word_index: Dict[str, int],
    w2v_model: Word2Vec,
    embedding_dim: int,
) -> np.ndarray:
    """Monta matriz de embeddings pré-treinados a partir do Word2Vec."""
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, index in word_index.items():
        if word in w2v_model.wv.key_to_index:
            embedding_matrix[index] = w2v_model.wv[word]

    return embedding_matrix


def document_vector(tokens: List[str], w2v_model: Word2Vec, vector_size: int) -> np.ndarray:
    """Representa um documento pela média dos vetores das palavras."""
    vectors = [
        w2v_model.wv[word] for word in tokens if word in w2v_model.wv.key_to_index
    ]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: List[str],
    title: str,
    output_path: str,
) -> None:
    """Gera e salva a matriz de confusão."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def build_and_train_deep_model(
    X_train_texts,
    y_train,
    X_test_texts,
    y_test,
    w2v_model: Word2Vec,
    class_names: List[str],
    num_words: int = 20000,
    max_len: int = 200,
    embedding_dim: int = 100,
    epochs: int = 15,
    batch_size: int = 64,
) -> Dict:
    """
    Abordagem Deep Learning com embeddings Word2Vec pré-treinados + LSTM.
    """
    start_time = time.time()

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_texts)

    X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
    X_test_seq = tokenizer.texts_to_sequences(X_test_texts)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

    embedding_matrix = build_embedding_matrix(tokenizer.word_index, w2v_model, embedding_dim)
    vocab_size = min(num_words, len(tokenizer.word_index) + 1)

    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_len,
                weights=[embedding_matrix[:vocab_size]],
                trainable=True,
            ),
            LSTM(128, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(len(class_names), activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stopping],
        verbose=1,
    )

    print("\nHistórico de treinamento (Deep Learning):")
    for epoch, (loss, acc, val_loss, val_acc) in enumerate(
        zip(
            history.history["loss"],
            history.history["accuracy"],
            history.history["val_loss"],
            history.history["val_accuracy"],
        ),
        start=1,
    ):
        print(
            f"  Época {epoch:02d} | loss={loss:.4f} | acc={acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    y_pred_probs = model.predict(X_test_pad, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    elapsed = time.time() - start_time
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_deep_learning.png")
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        "Matriz de Confusão - Deep Learning (Word2Vec + LSTM)",
        cm_path,
    )

    return {
        "name": "Deep Learning (Word2Vec pré-treinado + LSTM)",
        "classification_report": report,
        "time_seconds": elapsed,
        "y_pred": y_pred,
        "confusion_matrix_path": cm_path,
        "training_history": history.history,
    }


def train_word2vec_classifier(
    tokenized_texts_train,
    tokenized_texts_test,
    y_train,
    y_test,
    w2v_model: Word2Vec,
    class_names: List[str],
    vector_size: int = 100,
) -> Dict:
    """Abordagem Word2Vec + Regressão Logística."""
    start_time = time.time()

    X_train_vec = np.array(
        [document_vector(tokens, w2v_model, vector_size) for tokens in tokenized_texts_train]
    )
    X_test_vec = np.array(
        [document_vector(tokens, w2v_model, vector_size) for tokens in tokenized_texts_test]
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    elapsed = time.time() - start_time
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_word2vec.png")
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_names,
        "Matriz de Confusão - Word2Vec + Regressão Logística",
        cm_path,
    )

    return {
        "name": "Word2Vec + Regressão Logística",
        "classification_report": report,
        "time_seconds": elapsed,
        "y_pred": y_pred,
        "confusion_matrix_path": cm_path,
    }


def format_report(report_dict: Dict, time_seconds: float) -> str:
    """Constrói texto legível a partir do classification_report."""
    lines = []
    lines.append(f"Tempo de execução: {time_seconds:.2f} segundos\n")
    lines.append("Métricas por classe (precision, recall, f1-score, support):\n")

    for label, metrics in report_dict.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        f1 = metrics.get("f1-score", 0.0)
        support = metrics.get("support", 0)
        lines.append(
            f"Classe {label:>15}: "
            f"precision={precision:.3f}, recall={recall:.3f}, "
            f"f1-score={f1:.3f}, support={support}"
        )

    if "accuracy" in report_dict:
        lines.append(f"\nAcurácia geral: {report_dict['accuracy']:.3f}")

    if "macro avg" in report_dict:
        macro = report_dict["macro avg"]
        lines.append(
            "Média macro - "
            f"precision={macro['precision']:.3f}, "
            f"recall={macro['recall']:.3f}, "
            f"f1-score={macro['f1-score']:.3f}"
        )

    if "weighted avg" in report_dict:
        wavg = report_dict["weighted avg"]
        lines.append(
            "Média ponderada - "
            f"precision={wavg['precision']:.3f}, "
            f"recall={wavg['recall']:.3f}, "
            f"f1-score={wavg['f1-score']:.3f}"
        )

    return "\n".join(lines)


def generate_conclusion(deep_results: Dict, w2v_results: Dict) -> str:
    """Gera conclusão automática com base nos resultados reais."""
    deep_report = deep_results["classification_report"]
    w2v_report = w2v_results["classification_report"]

    deep_f1 = deep_report["macro avg"]["f1-score"]
    w2v_f1 = w2v_report["macro avg"]["f1-score"]
    deep_acc = deep_report["accuracy"]
    w2v_acc = w2v_report["accuracy"]
    deep_time = deep_results["time_seconds"]
    w2v_time = w2v_results["time_seconds"]

    if w2v_f1 > deep_f1:
        winner = w2v_results["name"]
        loser = deep_results["name"]
    else:
        winner = deep_results["name"]
        loser = w2v_results["name"]

    return f"""
CONCLUSÃO

A abordagem com melhor desempenho neste experimento foi: {winner}.

Resultados comparativos:
- {deep_results['name']}: acurácia={deep_acc:.3f}, F1 macro={deep_f1:.3f}, tempo={deep_time:.2f}s
- {w2v_results['name']}: acurácia={w2v_acc:.3f}, F1 macro={w2v_f1:.3f}, tempo={w2v_time:.2f}s

Vantagens do Deep Learning (Word2Vec pré-treinado + LSTM):
- Captura dependências sequenciais e contexto entre palavras.
- Permite fine-tuning dos embeddings durante o treinamento da rede.
- Pode escalar melhor em cenários com mais dados e textos mais complexos.

Desvantagens do Deep Learning:
- Maior custo computacional e tempo de treinamento ({deep_time:.2f}s neste experimento).
- Exige mais ajuste de hiperparâmetros (épocas, dropout, arquitetura).
- Menor interpretabilidade dos resultados.

Vantagens do Word2Vec + Regressão Logística:
- Implementação mais simples e treinamento muito mais rápido ({w2v_time:.2f}s).
- Excelente desempenho em classificação de notícias com vocabulário estável.
- Maior interpretabilidade do classificador linear.

Desvantagens do Word2Vec + Regressão Logística:
- A média dos vetores pode perder informação de ordem das palavras.
- Depende da qualidade do Word2Vec treinado no corpus.
- Pode ser limitado em tarefas que exigem compreensão profunda de contexto.

Neste dataset BBC News, {winner} apresentou melhor equilíbrio entre desempenho e eficiência,
enquanto {loser} ficou atrás nas métricas de classificação obtidas.
""".strip()


def main():
    set_seeds()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(os.path.dirname(__file__), "bbc-text.csv")
    print(f"Lendo dataset em: {csv_path}")

    df = load_dataset(csv_path)

    print("Pré-processando textos...")
    cleaned_texts, tokenized_texts = preprocess_texts(df["text"])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["category"])
    class_names = list(label_encoder.classes_)

    X_train_texts, X_test_texts, y_train, y_test, tokenized_train, tokenized_test = train_test_split(
        cleaned_texts,
        y,
        tokenized_texts,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Treinando Word2Vec (Gensim) no conjunto de treino...")
    w2v_model = train_word2vec_model(tokenized_train)

    print("Treinando modelo de Deep Learning com embeddings Word2Vec pré-treinados...")
    deep_results = build_and_train_deep_model(
        X_train_texts=X_train_texts,
        y_train=y_train,
        X_test_texts=X_test_texts,
        y_test=y_test,
        w2v_model=w2v_model,
        class_names=class_names,
    )

    print("Treinando modelo Word2Vec + Regressão Logística...")
    w2v_results = train_word2vec_classifier(
        tokenized_texts_train=tokenized_train,
        tokenized_texts_test=tokenized_test,
        y_train=y_train,
        y_test=y_test,
        w2v_model=w2v_model,
        class_names=class_names,
    )

    report_lines = []
    report_lines.append("========== RELATÓRIO COMPARATIVO ==========\n")
    report_lines.append(f">>> {deep_results['name']}\n")
    report_lines.append(format_report(deep_results["classification_report"], deep_results["time_seconds"]))
    report_lines.append(f"\nMatriz de confusão salva em: {deep_results['confusion_matrix_path']}\n")
    report_lines.append("\n-------------------------------------------\n")
    report_lines.append(f">>> {w2v_results['name']}\n")
    report_lines.append(format_report(w2v_results["classification_report"], w2v_results["time_seconds"]))
    report_lines.append(f"\nMatriz de confusão salva em: {w2v_results['confusion_matrix_path']}\n")
    report_lines.append("\n========== CONCLUSÃO ==========\n")
    report_lines.append(generate_conclusion(deep_results, w2v_results))

    final_report = "\n".join(report_lines)
    print(f"\n{final_report}")

    report_path = os.path.join(OUTPUT_DIR, "relatorio_comparativo.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(final_report)

    print(f"\nRelatório completo salvo em: {report_path}")


if __name__ == "__main__":
    main()
