"""
System Detekcji Anomalii w Transakcjach Bankowych
================================================

Ten skrypt integruje podstawowe oraz zaawansowane algorytmy detekcji anomalii
do wykrywania podejrzanych transakcji bankowych.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import os
import time
import datetime as dt
import joblib
import json

# Import zaawansowanych algorytmów (upewnij się, że plik jest w tym samym katalogu)
from advanced_anomaly_detection import (
    train_and_evaluate_advanced_models,
    visualize_advanced_results,
    HBOS,
    EnsembleAnomalyDetector
)

# Konfiguracja stylów wizualizacji
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        print("Informacja: Nie można ustawić stylu seaborn-whitegrid, używam domyślnego.")

sns.set_palette("Set2")


def generate_transaction_data(n_samples=10000, fraud_ratio=0.01):
    """
    Generuje syntetyczne dane transakcji bankowych z podanym odsetkiem transakcji fraudowych.
    
    Parameters:
    -----------
    n_samples : int, default=10000
        Liczba transakcji do wygenerowania
    fraud_ratio : float, default=0.01
        Proporcja transakcji fraudowych
        
    Returns:
    --------
    pd.DataFrame
        DataFrame zawierający dane transakcji
    """
    print(f"Generowanie {n_samples} transakcji (w tym {fraud_ratio*100}% fraudowych)...")
    
    # Liczba transakcji fraudowych
    n_frauds = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_frauds
    
    # Generowanie normalnych transakcji
    normal_data = {
        'transaction_id': [f'T{i:08d}' for i in range(n_normal)],
        'customer_id': np.random.randint(1000, 9999, size=n_normal),
        'timestamp': [int(time.time() - np.random.randint(0, 60*60*24*30)) for _ in range(n_normal)],  # Ostatnie 30 dni
        'amount': np.random.lognormal(4, 0.8, n_normal),  # Większość transakcji w zakresie kilkuset złotych
        'merchant_category': np.random.choice(['grocery', 'restaurant', 'entertainment', 'transport', 'online_retail', 'utilities'], n_normal, p=[0.3, 0.2, 0.1, 0.15, 0.15, 0.1]),
        'location_country': np.random.choice(['Poland', 'Germany', 'UK', 'France', 'Spain'], n_normal, p=[0.8, 0.05, 0.05, 0.05, 0.05]),
        'card_present': np.random.choice([True, False], n_normal, p=[0.7, 0.3]),
        'is_fraud': np.zeros(n_normal, dtype=int)
    }
    
    # Generowanie fraudowych transakcji
    fraud_data = {
        'transaction_id': [f'T{i+n_normal:08d}' for i in range(n_frauds)],
        'customer_id': np.random.randint(1000, 9999, size=n_frauds),
        'timestamp': [int(time.time() - np.random.randint(0, 60*60*24*30)) for _ in range(n_frauds)],
        'amount': np.random.lognormal(6, 1.2, n_frauds),  # Wyższe kwoty dla fraudów
        'merchant_category': np.random.choice(['entertainment', 'online_retail', 'other'], n_frauds, p=[0.3, 0.5, 0.2]),
        'location_country': np.random.choice(['USA', 'Russia', 'China', 'Poland', 'Nigeria'], n_frauds, p=[0.2, 0.2, 0.2, 0.2, 0.2]),
        'card_present': np.random.choice([True, False], n_frauds, p=[0.1, 0.9]),  # Fraudy częściej online
        'is_fraud': np.ones(n_frauds, dtype=int)
    }
    
    # Łączenie danych
    all_data = {}
    for key in normal_data:
        all_data[key] = np.concatenate([normal_data[key], fraud_data[key]])
    
    # Tworzenie DataFrame i losowe sortowanie
    df = pd.DataFrame(all_data)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Dodanie dodatkowych kolumn
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Dodanie kolumny reprezentującej historyczne wydatki
    customer_spending = {}
    for cid in df['customer_id'].unique():
        customer_spending[cid] = np.random.lognormal(5, 0.5)  # Średnie miesięczne wydatki
    
    df['customer_avg_monthly_spending'] = df['customer_id'].map(customer_spending)
    df['amount_to_avg_ratio'] = df['amount'] / df['customer_avg_monthly_spending']
    
    # Dodanie cech specyficznych dla zaawansowanych algorytmów
    df['transaction_velocity'] = np.random.randint(1, 10, size=len(df))  # Symulacja prędkości transakcji (liczba transakcji w ostatnim czasie)
    df['days_since_last_transaction'] = np.random.randint(0, 30, size=len(df))  # Symulacja dni od ostatniej transakcji
    
    print(f"Wygenerowano dane: {df.shape[0]} transakcji, {df['is_fraud'].sum()} fraudowych ({df['is_fraud'].mean()*100:.2f}%)")
    return df


def explore_data(df):
    """
    Przeprowadza eksploracyjną analizę danych transakcji.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame zawierający dane transakcji
        
    Returns:
    --------
    dict
        Słownik zawierający statystyki i wykresy
    """
    print("Rozpoczynam eksploracyjną analizę danych...")
    
    # Podstawowe informacje o danych
    print("\nPodstawowe informacje:")
    print(f"Liczba transakcji: {df.shape[0]}")
    print(f"Liczba fraudów: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    
    # Statystyki dla poszczególnych kolumn
    print("\nStatystyki kwot transakcji:")
    fraud_amounts = df[df['is_fraud'] == 1]['amount']
    normal_amounts = df[df['is_fraud'] == 0]['amount']
    
    print(f"Średnia kwota (normalne): {normal_amounts.mean():.2f} PLN")
    print(f"Średnia kwota (fraudy): {fraud_amounts.mean():.2f} PLN")
    print(f"Mediana kwoty (normalne): {normal_amounts.median():.2f} PLN")
    print(f"Mediana kwoty (fraudy): {fraud_amounts.median():.2f} PLN")
    
    # Tworzenie wykresów
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram kwot transakcji
    ax = axes[0, 0]
    bins = np.logspace(1, 5, 50)
    ax.hist(normal_amounts, bins=bins, alpha=0.7, label='Normalne')
    ax.hist(fraud_amounts, bins=bins, alpha=0.7, label='Fraudy')
    ax.set_xscale('log')
    ax.set_xlabel('Kwota (PLN)')
    ax.set_ylabel('Liczba transakcji')
    ax.set_title('Rozkład kwot transakcji (skala logarytmiczna)')
    ax.legend()
    
    # Fraudy według kategorii merchanta
    ax = axes[0, 1]
    fraud_by_category = df.groupby('merchant_category')['is_fraud'].mean() * 100
    fraud_by_category.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_xlabel('Kategoria merchanta')
    ax.set_ylabel('Procent fraudów (%)')
    ax.set_title('Procent fraudów według kategorii merchanta')
    
    # Fraudy według pory dnia
    ax = axes[1, 0]
    fraud_by_hour = df.groupby('hour_of_day')['is_fraud'].mean() * 100
    fraud_by_hour.plot(kind='line', marker='o', ax=ax)
    ax.set_xlabel('Godzina dnia')
    ax.set_ylabel('Procent fraudów (%)')
    ax.set_title('Procent fraudów według godziny dnia')
    ax.set_xticks(range(0, 24, 2))
    
    # Fraudy według kraju
    ax = axes[1, 1]
    fraud_by_country = df.groupby('location_country')['is_fraud'].mean() * 100
    fraud_by_country.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_xlabel('Kraj')
    ax.set_ylabel('Procent fraudów (%)')
    ax.set_title('Procent fraudów według kraju')
    
    plt.tight_layout()
    plt.savefig('eda_plots.png', dpi=300)
    
    # Korelacja cech
    numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Macierz korelacji cech')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    
    print("Zakończono eksploracyjną analizę danych.")
    
    return {
        'fraud_ratio': df['is_fraud'].mean(),
        'avg_normal_amount': normal_amounts.mean(),
        'avg_fraud_amount': fraud_amounts.mean(),
        'corr_matrix': corr_matrix
    }


def prepare_data(df):
    """
    Przygotowuje dane do modelowania.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame zawierający dane transakcji
        
    Returns:
    --------
    tuple
        (X_train_unsupervised, X_train_unsupervised_scaled, X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test, features)
    """
    print("Przygotowywanie danych do modelowania...")
    
    # Wybór cech do modelu
    features = [
        'amount', 'hour_of_day', 'day_of_week', 'weekend',
        'customer_avg_monthly_spending', 'amount_to_avg_ratio',
        'transaction_velocity', 'days_since_last_transaction'
    ]
    
    # One-hot encoding dla zmiennych kategorycznych
    categorical_features = ['merchant_category', 'location_country', 'card_present']
    
    # Dodanie zmiennych kategorycznych po one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    # Aktualizacja listy cech
    features.extend([col for col in df_encoded.columns if any(cat in col for cat in categorical_features)])
    
    # Podział na zbiory treningowy i testowy
    X = df_encoded[features]
    y = df_encoded['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Dodatkowo przygotujemy dane do treningu nienadzorowanego (używamy tylko normalnych transakcji)
    X_train_unsupervised = X_train[y_train == 0]
    
    # Skalowanie danych
    scaler = StandardScaler()
    X_train_unsupervised_scaled = scaler.fit_transform(X_train_unsupervised)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Wymiar zbioru treningowego (nienadzorowany): {X_train_unsupervised.shape}")
    print(f"Wymiar zbioru treningowego (nadzorowany): {X_train.shape}")
    print(f"Wymiar zbioru testowego: {X_test.shape}")
    
    return X_train_unsupervised, X_train_unsupervised_scaled, X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test, features


def train_and_evaluate_models(X_train_unsupervised_scaled, X_train, y_train, X_test_scaled, y_test):
    """
    Trenuje i ewaluuje różne modele detekcji anomalii.
    
    Parameters:
    -----------
    X_train_unsupervised_scaled : np.ndarray
        Przeskalowane cechy zbioru treningowego dla metod nienadzorowanych (tylko normalne transakcje)
    X_train : pd.DataFrame
        Cechy zbioru treningowego
    y_train : pd.Series
        Etykiety zbioru treningowego
    X_test_scaled : np.ndarray
        Przeskalowane cechy zbioru testowego
    y_test : pd.Series
        Etykiety zbioru testowego
        
    Returns:
    --------
    dict
        Słownik zawierający wytrenowane modele i ich wyniki
    """
    print("Trenowanie i ewaluacja podstawowych modeli detekcji anomalii...")
    
    results = {}
    
    # 1. Isolation Forest
    print("\nTrenowanie modelu Isolation Forest...")
    start_time = time.time()
    if_model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.01,  # Oczekiwany procent anomalii
        random_state=42
    )
    if_model.fit(X_train_unsupervised_scaled)
    
    # Predykcja (1 dla inlierów, -1 dla outlierów - konwertujemy do 0/1 gdzie 1 to anomalia)
    if_preds = (if_model.predict(X_test_scaled) == -1).astype(int)
    if_scores = -if_model.decision_function(X_test_scaled)  # Wyższe wartości = większe prawdopodobieństwo anomalii
    print(f"Czas trenowania Isolation Forest: {time.time() - start_time:.2f}s")
    
    # 2. Local Outlier Factor
    print("Trenowanie modelu Local Outlier Factor...")
    start_time = time.time()
    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.01,
        novelty=True
    )
    lof_model.fit(X_train_unsupervised_scaled)
    
    lof_preds = (lof_model.predict(X_test_scaled) == -1).astype(int)
    lof_scores = -lof_model.decision_function(X_test_scaled)
    print(f"Czas trenowania Local Outlier Factor: {time.time() - start_time:.2f}s")
    
    # 3. One-Class SVM
    print("Trenowanie modelu One-Class SVM...")
    start_time = time.time()
    ocsvm_model = OneClassSVM(
        kernel='rbf',
        nu=0.01,  # Górna granica frakcji outlierów
        gamma='scale'
    )
    ocsvm_model.fit(X_train_unsupervised_scaled)
    
    ocsvm_preds = (ocsvm_model.predict(X_test_scaled) == -1).astype(int)
    ocsvm_scores = -ocsvm_model.decision_function(X_test_scaled)
    print(f"Czas trenowania One-Class SVM: {time.time() - start_time:.2f}s")
    
    # Zapisywanie wyników
    models = {
        'Isolation Forest': (if_model, if_preds, if_scores),
        'Local Outlier Factor': (lof_model, lof_preds, lof_scores),
        'One-Class SVM': (ocsvm_model, ocsvm_preds, ocsvm_scores)
    }
    
    for name, (model, preds, scores) in models.items():
        # Raport klasyfikacji
        report = classification_report(y_test, preds, output_dict=True)
        
        # Macierz pomyłek
        cm = confusion_matrix(y_test, preds)
        
        # ROC i AUC
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_score = auc(fpr, tpr)
        
        # Precision-Recall
        precision, recall, _ = precision_recall_curve(y_test, scores)
        
        results[name] = {
            'model': model,
            'predictions': preds,
            'scores': scores,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc': {'fpr': fpr, 'tpr': tpr, 'auc': auc_score},
            'pr': {'precision': precision, 'recall': recall}
        }
        
        print(f"\nWyniki dla modelu {name}:")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Precision (fraud): {report.get('1', {}).get('precision', 0):.4f}")
        print(f"Recall (fraud): {report.get('1', {}).get('recall', 0):.4f}")
        print(f"F1-score (fraud): {report.get('1', {}).get('f1-score', 0):.4f}")
        print(f"AUC: {auc_score:.4f}")
    
    return results


def visualize_results(basic_results, advanced_results, X_test, X_test_scaled, y_test, features):
    """
    Wizualizuje wyniki wszystkich modeli.
    
    Parameters:
    -----------
    basic_results : dict
        Słownik zawierający wyniki podstawowych modeli
    advanced_results : dict
        Słownik zawierający wyniki zaawansowanych modeli
    X_test : pd.DataFrame
        Cechy zbioru testowego
    X_test_scaled : np.ndarray
        Przeskalowane cechy zbioru testowego
    y_test : pd.Series
        Etykiety zbioru testowego
    features : list
        Lista nazw cech
    """
    print("Wizualizacja wyników modeli...")
    
    # Połącz wyniki z obu kategorii modeli dla wspólnej wizualizacji
    all_results = {**basic_results, **advanced_results}
    
    # 1. Krzywe ROC
    plt.figure(figsize=(12, 10))
    
    # Pętla przez wszystkie modele
    for name, result in all_results.items():
        fpr = result['roc']['fpr']
        tpr = result['roc']['tpr']
        auc_score = result['roc']['auc']
        
        # Używaj różnych stylów linii dla różnych kategorii modeli
        linestyle = '--' if name in advanced_results else '-'
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linestyle=linestyle)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (1 - Specyficzność)')
    plt.ylabel('True Positive Rate (Czułość)')
    plt.title('Krzywe ROC dla wszystkich modeli detekcji anomalii')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('all_roc_curves.png', dpi=300)
    
    # 2. Krzywe Precision-Recall
    plt.figure(figsize=(12, 10))
    
    for name, result in all_results.items():
        precision = result['pr']['precision']
        recall = result['pr']['recall']
        
        # Używaj różnych stylów linii dla różnych kategorii modeli
        linestyle = '--' if name in advanced_results else '-'
        plt.plot(recall, precision, label=name, linestyle=linestyle)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Krzywe Precision-Recall dla wszystkich modeli detekcji anomalii')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('all_pr_curves.png', dpi=300)
    
    # 3. Porównanie wyników wszystkich modeli
    results_df = []
    
    for name, result in all_results.items():
        report = result['classification_report']
        auc_score = result['roc']['auc']
        
        results_df.append({
            'Model': name,
            'Kategoria': 'Zaawansowany' if name in advanced_results else 'Podstawowy',
            'Accuracy': report['accuracy'],
            'Precision': report.get('1', {}).get('precision', 0),
            'Recall': report.get('1', {}).get('recall', 0),
            'F1': report.get('1', {}).get('f1-score', 0),
            'AUC': auc_score
        })
    
    results_table = pd.DataFrame(results_df)
    results_table.set_index(['Kategoria', 'Model'], inplace=True)
    
    # Formatowanie do wyświetlenia
    display_table = results_table.copy()
    for col in display_table.columns:
        display_table[col] = display_table[col].map(lambda x: f"{x:.4f}")
    
    print("\nPorównanie wszystkich modeli:")
    print(display_table)
    
    # Zapisanie tabeli do CSV
    results_table.to_csv('model_comparison.csv')
    
    # 4. Wizualizacja porównania metryk dla wszystkich modeli
    plt.figure(figsize=(15, 10))
    
    metrics = ['AUC', 'F1', 'Precision', 'Recall']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        
        # Przygotuj dane do wykresu
        plot_data = pd.DataFrame({
            'Model': [r['Model'] for r in results_df],
            'Wartość': [r[metric] for r in results_df],
            'Kategoria': [r['Kategoria'] for r in results_df]
        })
        
        # Tworzenie wykresu słupkowego
        sns.barplot(x='Model', y='Wartość', hue='Kategoria', data=plot_data)
        plt.title(f'Porównanie {metric}')
        plt.ylim([0, 1])
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    plt.savefig('model_metrics_comparison.png', dpi=300)
    
    # 5. Wizualizacja najlepszych modeli na danych
    # Wybór najlepszego modelu podstawowego
    best_basic_model = max(basic_results.items(), key=lambda x: x[1]['roc']['auc'])[0]
    # Wybór najlepszego modelu zaawansowanego
    if advanced_results:  # Sprawdzamy, czy mamy zaawansowane modele
        best_advanced_model = max(advanced_results.items(), key=lambda x: x[1]['roc']['auc'])[0]
    
    plt.figure(figsize=(14, 7))
    
    # Wybieramy dwie cechy do wizualizacji
    feature1 = 'amount'
    feature2 = 'amount_to_avg_ratio'
    feature1_idx = features.index(feature1)
    feature2_idx = features.index(feature2)
    
    # Wizualizacja dla najlepszego modelu podstawowego
    plt.subplot(1, 2, 1)
    best_basic_preds = basic_results[best_basic_model]['predictions']
    
    plt.scatter(X_test[feature1][y_test == 0], X_test[feature2][y_test == 0],
               c='blue', label='Prawdziwe normalne', alpha=0.5, s=10)
    plt.scatter(X_test[feature1][y_test == 1], X_test[feature2][y_test == 1],
               c='red', label='Prawdziwe fraudy', alpha=0.8, s=30)
    plt.scatter(X_test[feature1][best_basic_preds == 1], X_test[feature2][best_basic_preds == 1],
               facecolors='none', edgecolors='green', label='Wykryte anomalie', s=80)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f'Najlepszy model podstawowy: {best_basic_model}')
    plt.legend()
    plt.grid(True)
    
    # Wizualizacja dla najlepszego modelu zaawansowanego (jeśli istnieje)
    if advanced_results:
        plt.subplot(1, 2, 2)
        best_advanced_preds = advanced_results[best_advanced_model]['predictions']
        
        plt.scatter(X_test[feature1][y_test == 0], X_test[feature2][y_test == 0],
                  c='blue', label='Prawdziwe normalne', alpha=0.5, s=10)
        plt.scatter(X_test[feature1][y_test == 1], X_test[feature2][y_test == 1],
                  c='red', label='Prawdziwe fraudy', alpha=0.8, s=30)
        plt.scatter(X_test[feature1][best_advanced_preds == 1], X_test[feature2][best_advanced_preds == 1],
                  facecolors='none', edgecolors='green', label='Wykryte anomalie', s=80)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'Najlepszy model zaawansowany: {best_advanced_model}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('best_models_comparison.png', dpi=300)
    
    print("Wizualizacja zakończona. Zapisano wykresy.")
    
    return results_table


def main():
    """
    Główna funkcja uruchamiająca cały proces analizy.
    """
    print("=== ROZSZERZONY SYSTEM DETEKCJI ANOMALII W TRANSAKCJACH BANKOWYCH ===")
    print(f"Data i czas uruchomienia: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Ustawienie parametrów
    n_samples = 10000
    fraud_ratio = 0.01
    print(f"\nParametry: liczba próbek = {n_samples}, odsetek fraudów = {fraud_ratio*100}%\n")
    
    # 2. Generowanie danych
    df = generate_transaction_data(n_samples=n_samples, fraud_ratio=fraud_ratio)
    
    # Zapisanie wygenerowanych danych do pliku CSV
    # Upewnij się, że katalog istnieje i zapisz dane do pliku
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/transactions.csv', index=False)
    print(f"Dane zapisane do pliku: data/transactions.csv")
    
    # 3. Eksploracyjna analiza danych
    eda_results = explore_data(df)
    
    # 4. Przygotowanie danych
    X_train_unsupervised, X_train_unsupervised_scaled, X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test, features = prepare_data(df)
    
    # 5. Trenowanie i ewaluacja podstawowych modeli
    print("\n=== PODSTAWOWE MODELE DETEKCJI ANOMALII ===\n")
    basic_model_results = train_and_evaluate_models(X_train_unsupervised_scaled, X_train, y_train, X_test_scaled, y_test)
    
    # 6. Trenowanie i ewaluacja zaawansowanych modeli
    print("\n=== ZAAWANSOWANE MODELE DETEKCJI ANOMALII ===\n")
    advanced_model_results = train_and_evaluate_advanced_models(X_train_unsupervised_scaled, X_test_scaled, y_test)
    
    # 7. Wizualizacja i porównanie wszystkich wyników
    print("\n=== PORÓWNANIE WSZYSTKICH MODELI ===\n")
    results_table = visualize_results(basic_model_results, advanced_model_results, X_test, X_test_scaled, y_test, features)
    
    # 8. Podsumowanie
    print("\n=== PODSUMOWANIE WYNIKÓW ===")
    
    # Znajdź najlepszy model ze wszystkich
    all_results = {**basic_model_results, **advanced_model_results}
    best_model = max(all_results.items(), key=lambda x: x[1]['roc']['auc'])
    best_model_name = best_model[0]
    best_model_category = "zaawansowany" if best_model_name in advanced_model_results else "podstawowy"
    best_model_results = best_model[1]
    best_report = best_model_results['classification_report']
    best_auc = best_model_results['roc']['auc']
    
    print(f"\nNajlepszy model: {best_model_name} ({best_model_category})")
    print(f"AUC: {best_auc:.4f}")
    print(f"Accuracy: {best_report['accuracy']:.4f}")
    print(f"Precision (fraud): {best_report.get('1', {}).get('precision', 0):.4f}")
    print(f"Recall (fraud): {best_report.get('1', {}).get('recall', 0):.4f}")
    print(f"F1-score (fraud): {best_report.get('1', {}).get('f1-score', 0):.4f}")
    
    # 9. Zapisanie modeli
    
    # Zapisz najlepszy model
    joblib.dump(best_model_results['model'], f'best_model_{best_model_name.replace(" ", "_").lower()}.pkl')
    print(f"\nNajlepszy model zapisany jako: best_model_{best_model_name.replace(' ', '_').lower()}.pkl")
    
    # Zapisanie wszystkich wyników do pliku
    
    # Przygotowanie wyników do zapisu - poprawka dla serializacji do JSON
    serializable_results = {}
    for name, result in all_results.items():
        serializable_results[name] = {}
        
        # Kopiujemy tylko te elementy, które można serializować
        for key, value in result.items():
            if key == 'model':
                # Pomijamy obiekty modeli, których nie można serializować
                continue
            elif key == 'classification_report':
                # Raporty klasyfikacji można bezpośrednio serializować
                serializable_results[name][key] = value
            elif key == 'confusion_matrix':
                # Konwertujemy macierz pomyłek na zwykłą listę
                serializable_results[name][key] = value.tolist()
            elif key == 'predictions' or key == 'scores':
                # Konwertujemy arrays numpy do list
                serializable_results[name][key] = value.tolist()
            elif key == 'roc':
                # Obsługujemy strukturę ROC
                serializable_results[name][key] = {
                    'fpr': value['fpr'].tolist(),
                    'tpr': value['tpr'].tolist(),
                    'auc': float(value['auc'])
                }
            elif key == 'pr':
                # Obsługujemy strukturę Precision-Recall
                serializable_results[name][key] = {
                    'precision': value['precision'].tolist(),
                    'recall': value['recall'].tolist()
                }
    
    # Zapis do pliku JSON
    try:
        with open('model_results.json', 'w') as f:
            json.dump(serializable_results, f)
        print("Wyniki modeli zapisane jako: model_results.json")
    except Exception as e:
        print(f"Błąd podczas zapisywania wyników do JSON: {e}")
        print("Pomijanie zapisu wyników do JSON.")
    
    print("\nProjekt zakończony pomyślnie!")
    print(f"Czas zakończenia: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
