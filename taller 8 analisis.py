import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import unidecode
import seaborn as sns


data = pd.read_csv("https://raw.githubusercontent.com/NicolayB/taller-8/refs/heads/main/productividad.csv", delimiter=";")
print(data)
print("\n")

print(data.isnull().sum())
print("\n")

print(data.dtypes)
print("\n")

cols_to_drop = ['quarter', 'date','targeted_productivity']
ndata = data.drop(columns=cols_to_drop)
print(ndata.head())

# Definir las columnas que se utilizarán para identificar duplicados
columnas = ['department',
    'day',
    'smv',
    'wip',
    'over_time',
    'incentive',
    'idle_time',
    'idle_men',
    'no_of_style_change',
    'no_of_workers',
    'actual_productivity',
    'team'
    ]

# Identificar duplicados
duplicados_lat_lon = ndata[ndata.duplicated(subset=columnas, keep=False)]

# Contar el número de duplicados
num_duplicados_lat_lon = duplicados_lat_lon.shape[0]

# Mostrar el número de registros duplicados y los registros duplicados
print(f"Número de registros duplicados: {num_duplicados_lat_lon}")
print(duplicados_lat_lon)

ndata = ndata.drop_duplicates(subset=columnas, keep='first')
print("\n")
print(ndata['department'].unique())

ndata['department'] = ndata['department'].str.lower()
ndata['department'] = ndata['department'].apply(unidecode.unidecode)
ndata['department'] = ndata['department'].str.strip()
print(ndata['department'].unique())
print("\n")

print(ndata.dtypes)
print("\n")

ndata['no_of_workers'] = ndata['no_of_workers'].astype(int)

ndata['actual_productivity'] = pd.to_numeric(ndata['actual_productivity'], errors = 'coerce')
print(ndata.dtypes)
print("\n")

print(ndata.isnull().sum())
print("\n")

ndata.loc[ndata['department'] == 'finishing', 'wip'] = ndata.loc[ndata['department'] == 'finishing', 'wip'].fillna(0)

media_actpro = ndata.groupby(['department','no_of_workers'])['actual_productivity'].mean().reset_index()

# Mostramos el resultado
print(media_actpro.head())
print("\n")

media_actpro = ndata.groupby(['department', 'no_of_workers'])['actual_productivity'].mean().reset_index()
media_actpro.rename(columns={'actual_productivity': 'mean_productivity'}, inplace=True)
ndata = ndata.merge(media_actpro, on=['department', 'no_of_workers'], how='left')
ndata['actual_productivity'].fillna(ndata['mean_productivity'], inplace=True)
ndata.drop(columns=['mean_productivity'], inplace=True)
print(ndata.isnull().sum())
print("\n")


# Análisis de datos
# Estadísticas descriptivas
est_desc = ndata.describe()
print(est_desc)
#est_desc.to_excel("talleres/taller 8/estadisticas descriptivas.xlsx")
print("\n")

# Gráficos
# Histogramas
cols = ["wip", "incentive", "idle_time", "idle_men", "no_of_style_change"]
for col in ndata.columns:
    if col in cols:
        continue
    else:
        sns.histplot(ndata[col])
        plt.title(f"Histograma {col}")
        plt.xlabel("Valores")
        plt.ylabel("Frecuencia")
        plt.grid()
        plt.show()
# Boxplots
cols = ["day", "department", "team", "idle_men", "idle_time", "no_of_style_change"]
for col in ndata.columns:
    if col in cols:
        sns.boxplot(ndata, x=col, y="actual_productivity")
    else:
        sns.boxplot(ndata, y=col)
    plt.title(f"Boxplot {col}")
    plt.grid()
    plt.show()


dias = list(ndata["day"].unique())
# Se crean las variables dummies
cols = ndata.columns[0:2]
ndata = pd.get_dummies(ndata, prefix="", prefix_sep="", columns=cols, dtype=int)

# Crear matriz de variables y productividad actual
X = ndata.drop("actual_productivity", axis=1)
Y = ndata["actual_productivity"]


# Correlación entre variables
corr = ndata[ndata.columns[0:12]].corr()
sns.heatmap(corr, cmap="Blues", annot=True)
plt.title("Mapa de correlación entre variables")
plt.show()

# Correlación con la variable de interés
correlacion = pd.DataFrame(X.corrwith(Y))
sns.heatmap(correlacion, cmap="Blues", annot=True)
plt.title("Mapa de correlación con variable de interés")
plt.show()

# Diagramas de regresión para las variables que más nos interesan
cols = ["idle_time", "idle_men", "over_time", "no_of_style_change", "no_of_workers"]
cols.extend(dias)
print(cols)
sns.pairplot(ndata, x_vars=cols[0:5], y_vars="actual_productivity", height=5, kind="reg")
sns.pairplot(ndata, x_vars=cols[5:], y_vars="actual_productivity", height=5, kind="reg")
plt.show()


