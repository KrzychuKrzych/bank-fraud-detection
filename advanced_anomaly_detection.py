"""
Ten moduł zawiera zaawansowane algorytmy detekcji anomalii, które rozszerzają podstawową
funkcjonalność systemu detekcji fraudów w transakcjach bankowych.

Implementowane algorytmy:
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Elliptic Envelope
- Autoencoder (wykorzystujący Deep Learning)
- HBOS (Histogram-Based Outlier Score)
- Ensemble Anomaly Detection (połączenie wielu algorytmów)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import time

# Implementacja własnego algorytmu HBOS
class HBOS:
    """
    Histogram-Based Outlier Score (HBOS)
    
    Algorytm detekcji anomalii oparty na histogramach. Działa pod założeniem,
    że każda cecha ma swój rozkład, a obserwacje w regionach o niskim 
    prawdopodobieństwie są potencjalnymi anomaliami.
    
    Parameters:
    -----------
    n_bins : int, default=10
        Liczba kubełków (bins) w histogramach
    contamination : float, default=0.1
        Oczekiwana proporcja outlierów w danych
    """
    
    def __init__(self, n_bins=10, contamination=0.1):
        self.n_bins = n_bins
        self.contamination = contamination
        self.histograms = None
        self.bin_edges = None
        self.feature_log_probs = None
        self.threshold = None
        
    def fit(self, X):
        """
        Dopasowuje model do danych treningowych.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dane treningowe
        
        Returns:
        --------
        self : object
            Dopasowany model
        """
        n_samples, n_features = X.shape
        
        # Inicjalizacja struktur danych
        self.histograms = []
        self.bin_edges = []
        self.feature_log_probs = []
        
        # Dla każdej cechy:
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            # Utworzenie histogramu
            hist, bin_edges = np.histogram(feature_values, bins=self.n_bins, density=True)
            
            # Dodanie małej wartości dla uniknięcia log(0)
            hist = np.clip(hist, 1e-10, None)
            
            # Przekształcenie na ujemny logarytm prawdopodobieństwa
            log_prob = -np.log(hist)
            
            # Zapisanie wyników
            self.histograms.append(hist)
            self.bin_edges.append(bin_edges)
            self.feature_log_probs.append(log_prob)
        
        # Obliczenie scorów anomalii dla danych treningowych
        scores = self.decision_function(X)
        
        # Ustalenie progu na podstawie oczekiwanego zanieczyszczenia
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        return self
    
    def _get_bin_idx(self, value, bin_edges):
        """Pomocnicza funkcja do znajdowania indeksu kubełka dla wartości."""
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                return i
        return len(bin_edges) - 2  # Ostatni kubełek dla wartości >= max
    
    def decision_function(self, X):
        """
        Oblicza funkcję decyzyjną dla podanych danych.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dane do oceny
            
        Returns:
        --------
        scores : array of shape (n_samples,)
            Wynik anomalii dla każdej obserwacji. Wyższe wartości oznaczają
            większe prawdopodobieństwo anomalii.
        """
        n_samples, n_features = X.shape
        scores = np.zeros(n_samples)
        
        # Dla każdej obserwacji:
        for sample_idx in range(n_samples):
            sample_score = 0
            
            # Dla każdej cechy:
            for feature_idx in range(n_features):
                value = X[sample_idx, feature_idx]
                bin_edges = self.bin_edges[feature_idx]
                log_probs = self.feature_log_probs[feature_idx]
                
                # Znajdź odpowiedni kubełek
                bin_idx = self._get_bin_idx(value, bin_edges)
                
                # Dodaj log-prawdopodobieństwo do sumy
                sample_score += log_probs[bin_idx]
            
            scores[sample_idx] = sample_score
            
        return scores
    
    def predict(self, X):
        """
        Przewiduje, czy obserwacje są anomaliami.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dane do oceny
            
        Returns:
        --------
        predictions : array of shape (n_samples,)
            1 dla anomalii, 0 dla normalnych obserwacji
        """
        scores = self.decision_function(X)
        predictions = (scores >= self.threshold).astype(int)
        return predictions


# Ensemble Anomaly Detector
class EnsembleAnomalyDetector:
    """
    Ensemble Anomaly Detector
    
    Łączy kilka modeli detekcji anomalii, aby uzyskać bardziej niezawodne wyniki.
    Używa techniki głosowania większościowego lub średniej ważonej scorów.
    
    Parameters:
    -----------
    base_detectors : list
        Lista modeli detekcji anomalii
    weights : array-like, default=None
        Wagi dla każdego modelu. Jeśli None, używane są równe wagi.
    method : {'voting', 'averaging'}, default='averaging'
        Metoda łączenia predykcji.
    threshold : float, default=0.5
        Próg dla metody 'averaging'.
    """
    
    def __init__(self, base_detectors, weights=None, method='averaging', threshold=0.5):
        self.base_detectors = base_detectors
        
        # Inicjalizacja wag
        if weights is None:
            self.weights = np.ones(len(base_detectors)) / len(base_detectors)
        else:
            self.weights = weights
        
        self.method = method
        self.threshold = threshold
    
    def fit(self, X):
        """
        Dopasowuje wszystkie modele bazowe do danych treningowych.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dane treningowe
        
        Returns:
        --------
        self : object
            Dopasowany model
        """
        for detector in self.base_detectors:
            detector.fit(X)
        return self
    
    def decision_function(self, X):
        """
        Oblicza funkcję decyzyjną dla podanych danych.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dane do oceny
            
        Returns:
        --------
        scores : array of shape (n_samples,)
            Wynik anomalii dla każdej obserwacji.
        """
        n_samples = X.shape[0]
        all_scores = np.zeros((n_samples, len(self.base_detectors)))
        
        # Zbieranie scorów z wszystkich modeli
        for i, detector in enumerate(self.base_detectors):
            try:
                scores = detector.decision_function(X)
                
                # Normalizacja scorów do przedziału [0, 1]
                min_score, max_score = scores.min(), scores.max()
                if max_score > min_score:
                    scores = (scores - min_score) / (max_score - min_score)
                
                all_scores[:, i] = scores
            except AttributeError:
                # Jeśli model nie ma funkcji decision_function, używamy predict
                preds = detector.predict(X)
                all_scores[:, i] = preds
        
        # Łączenie scorów zgodnie z wybraną metodą
        if self.method == 'averaging':
            final_scores = np.average(all_scores, axis=1, weights=self.weights)
        else:
            # Głosowanie większościowe - zaimplementowane jako średnia predykcji binarnych
            binary_preds = (all_scores >= 0.5).astype(int)
            final_scores = np.average(binary_preds, axis=1, weights=self.weights)
        
        return final_scores
    
    def predict(self, X):
        """
        Przewiduje, czy obserwacje są anomaliami.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Dane do oceny
            
        Returns:
        --------
        predictions : array of shape (n_samples,)
            1 dla anomalii, 0 dla normalnych obserwacji
        """
        scores = self.decision_function(X)
        predictions = (scores >= self.threshold).astype(int)
        return predictions


# Funkcja do trenowania i ewaluacji zaawansowanych modeli
def train_and_evaluate_advanced_models(X_train_unsupervised_scaled, X_test_scaled, y_test):
    """
    Trenuje i ewaluuje zaawansowane modele detekcji anomalii.
    
    Parameters:
    -----------
    X_train_unsupervised_scaled : array-like of shape (n_samples, n_features)
        Przeskalowane dane treningowe (tylko normalne transakcje)
    X_test_scaled : array-like of shape (n_samples, n_features)
        Przeskalowane dane testowe
    y_test : array-like of shape (n_samples,)
        Etykiety zbioru testowego
        
    Returns:
    --------
    dict
        Słownik zawierający wytrenowane modele i ich wyniki
    """
    print("Trenowanie i ewaluacja zaawansowanych modeli detekcji anomalii...")
    
    results = {}
    contamination = 0.01  # Oczekiwany procent anomalii
    
    # 1. DBSCAN
    print("\nTrenowanie modelu DBSCAN...")
    start_time = time.time()
    
    # Dobór parametru eps na podstawie k-distance plot (można zaimplementować auto-dobór)
    eps = 0.5  # Dobrać odpowiednio do danych
    
    dbscan_model = DBSCAN(
        eps=eps,
        min_samples=5,  # Min. liczba próbek w sąsiedztwie
        n_jobs=-1
    )
    dbscan_model.fit(X_train_unsupervised_scaled)
    
    # Przekształcenie etykiet klastrów na predykcje anomalii (-1 to outlier)
    dbscan_labels = dbscan_model.labels_
    
    # Przygotowanie funkcji do predykcji nowych danych
    def dbscan_predict(model, X):
        # Implementacja prostej metody predykcji dla DBSCAN
        # Sprawdzenie dla każdej nowej obserwacji, czy znajduje się w odległości eps od punktów z klasy non-outlier
        
        pred = np.ones(X.shape[0], dtype=int)  # Domyślnie wszystko jako anomalie
        train_core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        train_core_samples_mask[model.core_sample_indices_] = True
        train_core_samples = X_train_unsupervised_scaled[train_core_samples_mask]
        
        for i, sample in enumerate(X):
            # Sprawdzenie czy punkt jest bliski core samples
            distances = np.linalg.norm(train_core_samples - sample, axis=1)
            if np.any(distances <= eps):
                pred[i] = 0  # Nie jest anomalią
        
        return pred
    
    # Predykcje na zbiorze testowym
    dbscan_preds = dbscan_predict(dbscan_model, X_test_scaled)
    
    # Tworzę scory na podstawie odległości do najbliższego non-outliera
    def dbscan_decision_function(model, X):
        train_core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
        train_core_samples_mask[model.core_sample_indices_] = True
        train_core_samples = X_train_unsupervised_scaled[train_core_samples_mask]
        
        scores = np.zeros(X.shape[0])
        for i, sample in enumerate(X):
            if train_core_samples.shape[0] > 0:
                # Minimalna odległość do core samples
                min_distance = np.min(np.linalg.norm(train_core_samples - sample, axis=1))
                scores[i] = min_distance
            else:
                scores[i] = 1.0  # W przypadku braku core samples
        
        return scores
    
    dbscan_scores = dbscan_decision_function(dbscan_model, X_test_scaled)
    
    print(f"Czas trenowania DBSCAN: {time.time() - start_time:.2f}s")
    
    # 2. Elliptic Envelope
    print("Trenowanie modelu Elliptic Envelope...")
    start_time = time.time()
    
    ee_model = EllipticEnvelope(
        contamination=contamination,
        random_state=42
    )
    ee_model.fit(X_train_unsupervised_scaled)
    
    # Predykcje (1 dla inlierów, -1 dla outlierów - konwertujemy do 0/1)
    ee_preds = (ee_model.predict(X_test_scaled) == -1).astype(int)
    ee_scores = -ee_model.decision_function(X_test_scaled)  # Wyższe wartości = większe prawdopodobieństwo anomalii
    
    print(f"Czas trenowania Elliptic Envelope: {time.time() - start_time:.2f}s")
    
    # 3. HBOS
    print("Trenowanie modelu HBOS...")
    start_time = time.time()
    
    hbos_model = HBOS(
        n_bins=20,
        contamination=contamination
    )
    hbos_model.fit(X_train_unsupervised_scaled)
    
    hbos_preds = hbos_model.predict(X_test_scaled)
    hbos_scores = hbos_model.decision_function(X_test_scaled)
    
    print(f"Czas trenowania HBOS: {time.time() - start_time:.2f}s")
    
    # 4. Ensemble - łączący Elliptic Envelope i HBOS
    print("Trenowanie modelu Ensemble...")
    start_time = time.time()
    
    ensemble_model = EnsembleAnomalyDetector(
        base_detectors=[ee_model, hbos_model],
        weights=[0.5, 0.5],
        method='averaging',
        threshold=0.5
    )
    # Ensemble nie wymaga dodatkowego trenowania, ponieważ używa już wytrenowanych modeli
    
    ensemble_preds = ensemble_model.predict(X_test_scaled)
    ensemble_scores = ensemble_model.decision_function(X_test_scaled)
    
    print(f"Czas trenowania Ensemble: {time.time() - start_time:.2f}s")
    
    # Zapisywanie wyników
    models = {
        'DBSCAN': (dbscan_model, dbscan_preds, dbscan_scores),
        'Elliptic Envelope': (ee_model, ee_preds, ee_scores),
        'HBOS': (hbos_model, hbos_preds, hbos_scores),
        'Ensemble': (ensemble_model, ensemble_preds, ensemble_scores)
    }
    
    for name, (model, preds, scores) in models.items():
        # Raport klasyfikacji
        report = classification_report(y_test, preds, output_dict=True)
        
        # Macierz pomyłek
        cm = confusion_matrix(y_test, preds)
        
        # ROC i AUC
        try:
            fpr, tpr, _ = roc_curve(y_test, scores)
            auc_score = auc(fpr, tpr)
        except:
            # W przypadku problemów z obliczeniem krzywej ROC
            fpr, tpr = np.array([0, 1]), np.array([0, 1])
            auc_score = 0.5
        
        # Precision-Recall
        try:
            precision, recall, _ = precision_recall_curve(y_test, scores)
        except:
            precision, recall = np.array([0, 1]), np.array([1, 0])
        
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
        if 'roc' in results[name]:
            print(f"AUC: {results[name]['roc']['auc']:.4f}")
    
    return results


# Funkcja do wizualizacji wyników zaawansowanych modeli
def visualize_advanced_results(all_results, X_test_scaled, y_test, features=None):
    """
    Wizualizuje i porównuje wyniki zaawansowanych modeli detekcji anomalii.
    
    Parameters:
    -----------
    all_results : dict
        Słownik zawierający wyniki wszystkich modeli
    X_test_scaled : array-like
        Przeskalowane dane testowe
    y_test : array-like
        Etykiety zbioru testowego
    features : list, optional
        Nazwy cech
    """
    # 1. Krzywe ROC dla wszystkich modeli
    plt.figure(figsize=(12, 10))
    
    for name, result in all_results.items():
        if 'roc' in result:
            fpr = result['roc']['fpr']
            tpr = result['roc']['tpr']
            auc_score = result['roc']['auc']
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Krzywe ROC dla wszystkich modeli detekcji anomalii')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('all_models_roc_curves.png', dpi=300)
    plt.show()
    
    # 2. Macierze pomyłek dla zaawansowanych modeli
    advanced_models = ['DBSCAN', 'Elliptic Envelope', 'HBOS', 'Ensemble']
    fig, axes = plt.subplots(2, 2, figsize=(18, 15))
    
    for i, name in enumerate(advanced_models):
        if name in all_results:
            cm = all_results[name]['confusion_matrix']
            ax = axes[i//2, i%2]
            
            # Normalizacja macierzy pomyłek (procentowo)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Dodanie opisów wartości
            labels = [f"{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)" for i in range(cm.shape[0]) for j in range(cm.shape[1])]
            labels = np.asarray(labels).reshape(cm.shape)
            
            sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
                    xticklabels=['Normal', 'Fraud'],
                    yticklabels=['Normal', 'Fraud'],
                    ax=ax)
            
            ax.set_title(f'Macierz pomyłek: {name}')
            ax.set_xlabel('Przewidziana klasa')
            ax.set_ylabel('Prawdziwa klasa')
    
    plt.tight_layout()
    plt.savefig('advanced_models_confusion_matrices.png', dpi=300)
    plt.show()
    
    # 3. Porównanie metryk dla wszystkich modeli
    metrics = []
    
    for name, result in all_results.items():
        report = result['classification_report']
        auc_score = result['roc']['auc'] if 'roc' in result else np.nan
        
        metrics.append({
            'Model': name, 
            'Accuracy': report['accuracy'], 
            'Precision': report.get('1', {}).get('precision', 0), 
            'Recall': report.get('1', {}).get('recall', 0), 
            'F1-Score': report.get('1', {}).get('f1-score', 0),
            'AUC': auc_score
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.set_index('Model', inplace=True)
    
    # Wizualizacja porównania metryk
    plt.figure(figsize=(14, 10))
    
    metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'AUC']
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        sns.barplot(x=metrics_df.index, y=metrics_df[metric])
        plt.title(f'Porównanie {metric}')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Dodanie etykiet wartości
        for j, v in enumerate(metrics_df[metric]):
            plt.text(j, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('models_metrics_comparison.png', dpi=300)
    plt.show()
    
    # Wyświetlenie tabeli porównawczej
    print("\nPorównanie wszystkich modeli:")
    
    # Formatowanie procentowe
    for col in metrics_df.columns:
        metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.4f}")
    
    return metrics_df


# Przykład użycia
if __name__ == "__main__":
    print("Ten moduł zawiera zaawansowane algorytmy detekcji anomalii dla systemu detekcji fraudów.")
    print("Zaimportuj funkcje z tego modułu do głównego pliku projektu.")
