"""
System Detekcji Anomalii w Transakcjach Bankowych

Projekt demonstruje zastosowanie uczenia maszynowego do wykrywania podejrzanych
transakcji bankowych z wykorzystaniem algorytmów detekcji anomalii.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import datetime as dt
import time

# Ustawienie stylu wykresów
# Ustawienie stylu zgodnego z nowszymi wersjami matplotlib
plt.style.use('seaborn-v0_8-whitegrid')  # Zaktualizowana nazwa stylu
sns.set_palette("Set2")

# Ustawienie losowości dla powtarzalności wyników
np.random.seed(42)

# 1. Generowanie syntetycznych danych transakcji
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
    
    print(f"Wygenerowano dane: {df.shape[0]} transakcji, {df['is_fraud'].sum()} fraudowych ({df['is_fraud'].mean()*100:.2f}%)")
    return df

# 2. Eksploracyjna analiza danych (EDA)
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
    corr_matrix = df[['amount', 'hour_of_day', 'day_of_week', 'weekend', 
                     'customer_avg_monthly_spending', 'amount_to_avg_ratio', 
                     'is_fraud']].corr()
    
    plt.figure(figsize=(10, 8))
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

# 3. Przygotowanie danych
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
        (X_train_unsupervised, y_train_unsupervised, X_train, y_train, X_test, y_test)
    """
    print("Przygotowywanie danych do modelowania...")
    
    # Wybór cech do modelu
    features = [
        'amount', 'hour_of_day', 'day_of_week', 'weekend',
        'customer_avg_monthly_spending', 'amount_to_avg_ratio'
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
    y_train_unsupervised = y_train[y_train == 0]
    
    print(f"Wymiar zbioru treningowego (nienadzorowany): {X_train_unsupervised.shape}")
    print(f"Wymiar zbioru treningowego (nadzorowany): {X_train.shape}")
    print(f"Wymiar zbioru testowego: {X_test.shape}")
    
    return X_train_unsupervised, y_train_unsupervised, X_train, y_train, X_test, y_test

# 4. Implementacja algorytmów detekcji anomalii
def train_and_evaluate_models(X_train_unsupervised, X_train, y_train, X_test, y_test):
    """
    Trenuje i ewaluuje różne modele detekcji anomalii.
    
    Parameters:
    -----------
    X_train_unsupervised : pd.DataFrame
        Cechy zbioru treningowego dla metod nienadzorowanych (tylko normalne transakcje)
    X_train : pd.DataFrame
        Cechy zbioru treningowego
    y_train : pd.Series
        Etykiety zbioru treningowego
    X_test : pd.DataFrame
        Cechy zbioru testowego
    y_test : pd.Series
        Etykiety zbioru testowego
        
    Returns:
    --------
    dict
        Słownik zawierający wytrenowane modele i ich wyniki
    """
    print("Trenowanie i ewaluacja modeli detekcji anomalii...")
    
    results = {}
    
    # Skalowanie danych
    scaler = StandardScaler()
    X_train_unsupervised_scaled = scaler.fit_transform(X_train_unsupervised)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Isolation Forest
    print("\nTrenowanie modelu Isolation Forest...")
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
    
    # 2. Local Outlier Factor
    print("Trenowanie modelu Local Outlier Factor...")
    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.01,
        novelty=True
    )
    lof_model.fit(X_train_unsupervised_scaled)
    
    lof_preds = (lof_model.predict(X_test_scaled) == -1).astype(int)
    lof_scores = -lof_model.decision_function(X_test_scaled)
    
    # 3. One-Class SVM
    print("Trenowanie modelu One-Class SVM...")
    ocsvm_model = OneClassSVM(
        kernel='rbf',
        nu=0.01,  # Górna granica frakcji outlierów
        gamma='scale'
    )
    ocsvm_model.fit(X_train_unsupervised_scaled)
    
    ocsvm_preds = (ocsvm_model.predict(X_test_scaled) == -1).astype(int)
    ocsvm_scores = -ocsvm_model.decision_function(X_test_scaled)
    
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

# 5. Wizualizacja wyników
def visualize_results(results, X_test, y_test):
    """
    Wizualizuje wyniki modeli.
    
    Parameters:
    -----------
    results : dict
        Słownik zawierający wyniki modeli
    X_test : pd.DataFrame
        Cechy zbioru testowego
    y_test : pd.Series
        Etykiety zbioru testowego
    """
    print("Wizualizacja wyników modeli...")
    
    # 1. Krzywe ROC
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr = result['roc']['fpr']
        tpr = result['roc']['tpr']
        auc_score = result['roc']['auc']
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Krzywe ROC dla modeli detekcji anomalii')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curves.png', dpi=300)
    
    # 2. Krzywe Precision-Recall
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        precision = result['pr']['precision']
        recall = result['pr']['recall']
        
        plt.plot(recall, precision, label=name)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Krzywe Precision-Recall dla modeli detekcji anomalii')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('pr_curves.png', dpi=300)
    
    # 3. Macierze pomyłek
    plt.figure(figsize=(15, 5))
    
    for i, (name, result) in enumerate(results.items()):
        cm = result['confusion_matrix']
        
        plt.subplot(1, 3, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        plt.title(f'Macierz pomyłek: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    
    # 4. Przykłady wykrytych anomalii
    plt.figure(figsize=(12, 10))
    
    # Wybierzmy dwie cechy do wizualizacji
    feature1 = 'amount'
    feature2 = 'amount_to_avg_ratio'
    
    for i, (name, result) in enumerate(results.items()):
        predictions = result['predictions']
        
        plt.subplot(2, 2, i+1)
        
        # Prawdziwe klasy
        plt.scatter(X_test[feature1][y_test == 0], X_test[feature2][y_test == 0], 
                   c='blue', label='Prawdziwe normalne', alpha=0.5, s=10)
        plt.scatter(X_test[feature1][y_test == 1], X_test[feature2][y_test == 1], 
                   c='red', label='Prawdziwe fraudy', alpha=0.8, s=30)
        
        # Dodanie predykcji
        plt.scatter(X_test[feature1][predictions == 1], X_test[feature2][predictions == 1], 
                   facecolors='none', edgecolors='green', label='Wykryte anomalie', s=80)
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.title(f'Wykryte anomalie: {name}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection.png', dpi=300)
    
    print("Wizualizacja zakończona. Zapisano wykresy.")

# 6. Główna funkcja
def main():
    """
    Główna funkcja uruchamiająca cały proces analizy.
    """
    print("=== SYSTEM DETEKCJI ANOMALII W TRANSAKCJACH BANKOWYCH ===")
    
    # 1. Generowanie danych
    df = generate_transaction_data(n_samples=10000, fraud_ratio=0.01)
    
    # 2. Eksploracyjna analiza danych
    eda_results = explore_data(df)
    
    # 3. Przygotowanie danych
    X_train_unsupervised, y_train_unsupervised, X_train, y_train, X_test, y_test = prepare_data(df)
    
    # 4. Trenowanie i ewaluacja modeli
    model_results = train_and_evaluate_models(X_train_unsupervised, X_train, y_train, X_test, y_test)
    
    # 5. Wizualizacja wyników
    visualize_results(model_results, X_test, y_test)
    
    # 6. Podsumowanie
    print("\n=== PODSUMOWANIE WYNIKÓW ===")
    best_model = max(model_results.items(), key=lambda x: x[1]['roc']['auc'])
    print(f"\nNajlepszy model: {best_model[0]}")
    print(f"AUC: {best_model[1]['roc']['auc']:.4f}")
    print(f"Precision (fraud): {best_model[1]['classification_report'].get('1', {}).get('precision', 0):.4f}")
    print(f"Recall (fraud): {best_model[1]['classification_report'].get('1', {}).get('recall', 0):.4f}")
    
    print("\nProjekt zakończony pomyślnie!")

if __name__ == "__main__":
    main()
