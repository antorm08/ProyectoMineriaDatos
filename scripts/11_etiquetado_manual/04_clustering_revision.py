"""Analiza coherencia de etiquetas generadas mediante clustering."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "manual_500" / "desarrollo_80_etiquetado.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "11_etiquetado_manual"


def analizar(input_file, report_dir, n_clusters, max_features, random_state):
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file).fillna("")
    df = df[(df["texto_modelo"].astype(str).str.strip() != "") & (df["sentimiento_final"].astype(str).str.strip() != "")].copy()

    vectorizador = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True)
    x = vectorizador.fit_transform(df["texto_modelo"].astype(str))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["cluster"] = kmeans.fit_predict(x)

    tabla = pd.crosstab(df["cluster"], df["sentimiento_final"])
    tabla_pct = tabla.div(tabla.sum(axis=1), axis=0).round(4)
    tabla.to_csv(report_dir / "clustering_vs_etiquetas_cantidad.csv", encoding="utf-8-sig")
    tabla_pct.to_csv(report_dir / "clustering_vs_etiquetas_porcentaje.csv", encoding="utf-8-sig")

    resumen = []
    for cluster, fila in tabla.iterrows():
        clase_top = fila.idxmax()
        cantidad_top = int(fila.max())
        total = int(fila.sum())
        resumen.append({
            "cluster": cluster,
            "clase_mayoritaria": clase_top,
            "cantidad_cluster": total,
            "cantidad_mayoritaria": cantidad_top,
            "pureza_cluster": round(cantidad_top / total, 4) if total else 0,
        })
    pd.DataFrame(resumen).to_csv(report_dir / "resumen_pureza_clusters.csv", index=False, encoding="utf-8-sig")

    metricas = pd.DataFrame([
        ("adjusted_rand_index", round(adjusted_rand_score(df["sentimiento_final"], df["cluster"]), 4)),
        ("normalized_mutual_info", round(normalized_mutual_info_score(df["sentimiento_final"], df["cluster"]), 4)),
        ("inertia", round(float(kmeans.inertia_), 4)),
    ], columns=["metrica", "valor"])
    metricas.to_csv(report_dir / "metricas_clustering.csv", index=False, encoding="utf-8-sig")
    df.to_csv(report_dir / "desarrollo_80_etiquetado_con_clusters.csv", index=False, encoding="utf-8-sig")

    print("Clustering generado. Revisa reports/11_etiquetado_manual/.")
    print(metricas.to_string(index=False))


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Clustering de revision sobre etiquetas generadas.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    analizar(args.input, args.report_dir, args.n_clusters, args.max_features, args.random_state)


if __name__ == "__main__":
    main()
