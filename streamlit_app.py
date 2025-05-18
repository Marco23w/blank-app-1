import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Configurazione per visualizzazione più chiara
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

# Creiamo un dataset semplificato per l'assicurazione sanitaria
# Useremo solo due variabili: età e stato di fumatore
np.random.seed(42)  # Per riproducibilità

# Generiamo dataset di esempio semplificato
n_samples = 10  # Piccolo numero di campioni per la comprensione
ages = np.random.randint(20, 65, n_samples)  # Età tra 20 e 65 anni
smoker = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 0=non fumatore, 1=fumatore

# Generiamo i costi assicurativi con una formula semplice
# Base: 5000 + 50 per anno di età + 8000 se fumatore + rumore casuale
base_cost = 5000
age_factor = 50
smoker_factor = 8000
noise = np.random.normal(0, 1000, n_samples)  # Rumore casuale

insurance_cost = base_cost + age_factor * ages + smoker_factor * smoker + noise

# Creiamo il dataframe
data = pd.DataFrame({
    'age': ages,
    'smoker': smoker,
    'insurance_cost': insurance_cost
})

print("Dataset di esempio:")
print(data)

# Prepariamo i dati per il modello
X = data[['age', 'smoker']]
y = data['insurance_cost']

# 1. SPIEGAZIONE DEL GRADIENT BOOSTING: SIMULAZIONE PASSO-PASSO

# Passo 1: Inizializzare il modello con la media target (primo stimatore)
initial_prediction = np.mean(y)
print(f"\nPasso 1: Inizializzazione con valore medio: {initial_prediction:.2f}")

# Calcolo dei residui iniziali
residuals = y - initial_prediction
print("\nResidui iniziali:")
for i in range(n_samples):
    print(f"  Campione {i+1}: Valore reale = {y.iloc[i]:.2f}, Predizione = {initial_prediction:.2f}, Residuo = {residuals.iloc[i]:.2f}")

# Passo 2: Addestriamo il primo albero di decisione sui residui
tree_depth = 1  # Albero molto semplice per comprensibilità
learning_rate = 0.1  # Learning rate basso per contributi graduali

tree1 = DecisionTreeRegressor(max_depth=tree_depth)
tree1.fit(X, residuals)

# Predicizione del primo albero
tree1_predictions = tree1.predict(X)
print("\nPasso 2: Predizioni del primo albero sui residui:")
for i in range(n_samples):
    print(f"  Campione {i+1}: Residuo = {residuals.iloc[i]:.2f}, Predizione albero = {tree1_predictions[i]:.2f}")

# Aggiorniamo la predizione con il contributo del primo albero
predictions_step2 = initial_prediction + learning_rate * tree1_predictions

# Calcoliamo i nuovi residui
residuals2 = y - predictions_step2
print("\nNuovi residui dopo il primo albero:")
for i in range(n_samples):
    print(f"  Campione {i+1}: Valore reale = {y.iloc[i]:.2f}, Predizione aggiornata = {predictions_step2[i]:.2f}, Residuo = {residuals2.iloc[i]:.2f}")

# Passo 3: Addestriamo il secondo albero sui nuovi residui
tree2 = DecisionTreeRegressor(max_depth=tree_depth)
tree2.fit(X, residuals2)

# Predizioni del secondo albero
tree2_predictions = tree2.predict(X)
print("\nPasso 3: Predizioni del secondo albero sui nuovi residui:")
for i in range(n_samples):
    print(f"  Campione {i+1}: Residuo = {residuals2.iloc[i]:.2f}, Predizione albero = {tree2_predictions[i]:.2f}")

# Aggiorniamo la predizione con il contributo del secondo albero
predictions_step3 = predictions_step2 + learning_rate * tree2_predictions

# Calcoliamo i nuovi residui
residuals3 = y - predictions_step3
print("\nNuovi residui dopo il secondo albero:")
for i in range(n_samples):
    print(f"  Campione {i+1}: Valore reale = {y.iloc[i]:.2f}, Predizione aggiornata = {predictions_step3[i]:.2f}, Residuo = {residuals3.iloc[i]:.2f}")

# Passo 4: Addestriamo il terzo albero sui nuovi residui
tree3 = DecisionTreeRegressor(max_depth=tree_depth)
tree3.fit(X, residuals3)

# Predizioni del terzo albero
tree3_predictions = tree3.predict(X)
print("\nPasso 4: Predizioni del terzo albero sui nuovi residui:")
for i in range(n_samples):
    print(f"  Campione {i+1}: Residuo = {residuals3.iloc[i]:.2f}, Predizione albero = {tree3_predictions[i]:.2f}")

# Aggiorniamo la predizione con il contributo del terzo albero
predictions_step4 = predictions_step3 + learning_rate * tree3_predictions

# Calcoliamo l'errore ad ogni passaggio
mse_initial = mean_squared_error(y, np.ones(n_samples) * initial_prediction)
mse_tree1 = mean_squared_error(y, predictions_step2)
mse_tree2 = mean_squared_error(y, predictions_step3)
mse_tree3 = mean_squared_error(y, predictions_step4)

print("\nErrore quadratico medio (MSE) ad ogni iterazione:")
print(f"  Iniziale (solo media): {mse_initial:.2f}")
print(f"  Dopo il primo albero: {mse_tree1:.2f}")
print(f"  Dopo il secondo albero: {mse_tree2:.2f}")
print(f"  Dopo il terzo albero: {mse_tree3:.2f}")

# 2. VISUALIZZAZIONE GRAFICA DEL PROCESSO

# Visualizza l'evoluzione delle previsioni per un campione specifico
sample_idx = 0  # Scegliamo il primo campione per semplicità

real_value = y.iloc[sample_idx]
predictions = [
    initial_prediction,
    predictions_step2[sample_idx],
    predictions_step3[sample_idx],
    predictions_step4[sample_idx]
]

plt.figure(figsize=(10, 6))
plt.plot([0, 1, 2, 3], [real_value] * 4, 'r-', label='Valore reale')
plt.plot([0, 1, 2, 3], predictions, 'bo-', label='Previsioni')
plt.xlabel('Numero di iterazioni')
plt.ylabel('Costo assicurativo')
plt.title('Evoluzione della previsione attraverso le iterazioni')
plt.legend()
plt.xticks([0, 1, 2, 3], ['Iniziale', 'Albero 1', 'Albero 2', 'Albero 3'])
plt.grid(True)
plt.savefig('gbr_iterazioni.png')

# Confronto tra predizioni finali e valori reali
plt.figure(figsize=(10, 6))
plt.scatter(range(n_samples), y, color='red', label='Valori reali')
plt.scatter(range(n_samples), predictions_step4, color='blue', label='Predizioni dopo 3 alberi')
plt.xlabel('Indice campione')
plt.ylabel('Costo assicurativo')
plt.title('Confronto tra valori reali e predizioni finali')
plt.legend()
plt.grid(True)
plt.savefig('gbr_confronto.png')

# 3. IMPLEMENTAZIONE CON SCIKIT-LEARN PER CONFRONTO

# Inizializziamo un modello GBR con 3 estimatori
gbr_model = GradientBoostingRegressor(n_estimators=3, learning_rate=learning_rate, max_depth=tree_depth, random_state=42)
gbr_model.fit(X, y)

# Predizioni con il modello completo
sklearn_predictions = gbr_model.predict(X)

print("\nConfrontiamo le nostre previsioni manuali con quelle di scikit-learn:")
for i in range(n_samples):
    print(f"  Campione {i+1}: Manuale = {predictions_step4[i]:.2f}, Scikit-learn = {sklearn_predictions[i]:.2f}")

# 4. CONSIDERAZIONI FINALI

print("\nRIDUZIONE DELL'ERRORE:")
print(f"Totale riduzione MSE: {mse_initial - mse_tree3:.2f} ({(1 - mse_tree3/mse_initial)*100:.2f}%)")

# Creiamo un riassunto delle feature importances dal modello scikit-learn
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': gbr_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nIMPORTANZA DELLE FEATURE:")
print(feature_importance)

# Visualizziamo l'importanza delle feature
plt.figure(figsize=(8, 5))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Importanza delle Feature nel Gradient Boosting')
plt.xlabel('Feature')
plt.ylabel('Importanza')
plt.tight_layout()
plt.savefig('gbr_feature_importance.png')

# 5. CONCLUSIONE E FORMULE CHIAVE

print("\nFORMULE CHIAVE DEL GRADIENT BOOSTING:")
print("1. Previsione iniziale F₀(x) = media di y")
print("2. Per ogni passo m = 1, 2, ..., M:")
print("   a. Calcolare i residui: r_i = y_i - F_{m-1}(x_i)")
print("   b. Addestrare un albero decisionale h_m(x) sui residui")
print("   c. Aggiornare il modello: F_m(x) = F_{m-1}(x) + η × h_m(x)")
print("      dove η è il learning rate (nel nostro esempio: 0.1)")
print("\nNel nostro caso, dopo 3 iterazioni:")
print(f"F_3(x) = {initial_prediction:.2f} + {learning_rate} × [tree1(x) + tree2(x) + tree3(x)]")

# Salva le immagini generate
print("\nVisualizzazioni salvate come:")
print("- gbr_iterazioni.png: Evoluzione delle previsioni")
print("- gbr_confronto.png: Confronto tra valori reali e predetti")
print("- gbr_feature_importance.png: Importanza delle feature")
