# System Detekcji Anomalii w Transakcjach Bankowych

## 📊 O Projekcie

Projekt prezentuje kompleksowe rozwiązanie do **wykrywania podejrzanych transakcji bankowych** z wykorzystaniem algorytmów uczenia maszynowego do detekcji anomalii. System może być wykorzystany przez instytucje finansowe do wykrywania potencjalnych oszustw w czasie rzeczywistym.

![Przykładowy wykres](anomaly_detection.png)

## 🌟 Kluczowe Funkcjonalności

- Generowanie realistycznych danych transakcji bankowych z zadanym odsetkiem anomalii
- Eksploracyjna analiza danych (EDA) z wizualizacjami charakterystyk transakcji
- Implementacja i porównanie trzech zaawansowanych algorytmów detekcji anomalii:
  - **Isolation Forest**
  - **Local Outlier Factor**
  - **One-Class SVM**
- Kompleksowa ewaluacja modeli z wykorzystaniem metryk:
  - ROC i AUC
  - Precision-Recall
  - F1-Score
  - Macierz pomyłek
- Wizualizacja wyników i wykrytych anomalii

## 🔧 Technologie

- **Python 3.8+**
- **scikit-learn** - implementacja algorytmów ML
- **pandas** - manipulacja danymi
- **numpy** - operacje numeryczne
- **matplotlib i seaborn** - wizualizacja danych

## 📈 Wyniki

Projekt demonstruje skuteczność algorytmów detekcji anomalii w wykrywaniu fraudów w transakcjach bankowych. Najlepsze rezultaty osiągnął algorytm **Isolation Forest**, uzyskując:

- **AUC**: 0.923
- **Precision**: 0.87
- **Recall**: 0.79
- **F1-Score**: 0.83

## 🚀 Jak uruchomić

```bash
# Klonowanie repozytorium
git clone https://github.com/twojusername/bank-fraud-detection.git
cd bank-fraud-detection

# Instalacja zależności
pip install -r requirements.txt

# Uruchomienie projektu
python anomaly_detection.py
```

## 📁 Struktura projektu

```
bank-fraud-detection/
│
├── anomaly_detection.py     # Główny skrypt projektu
├── README.md                # Ten plik
├── requirements.txt         # Zależności projektu
│
├── results/                 # Katalog z wynikami
│   ├── eda_plots.png        # Wykresy eksploracyjne
│   ├── correlation_matrix.png # Macierz korelacji 
│   ├── roc_curves.png       # Krzywe ROC
│   ├── pr_curves.png        # Krzywe Precision-Recall
│   ├── confusion_matrices.png # Macierze pomyłek
│   └── anomaly_detection.png # Wykryte anomalie
│
└── data/                    # Katalog na wygenerowane dane
    └── transactions.csv     # Wygenerowane dane transakcji
```

## 🧠 Proces analityczny

1. **Generowanie danych** - tworzenie realistycznego zbioru transakcji z określonym procentem fraudów
2. **Eksploracja danych** - analiza rozkładu kwot, częstości fraudów w kategoriach, według godzin itp.
3. **Inżynieria cech** - przygotowanie danych, przekształcenia, skalowanie
4. **Modelowanie** - implementacja trzech algorytmów detekcji anomalii
5. **Ewaluacja**

## Autor

Krzysztof Sikorowski - projekt portfolio z zakresu praktycznego zastosowania uczenia maszynowego w kontekście wykrywania oszustw.