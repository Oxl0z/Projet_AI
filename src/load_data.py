# src/load_data.py

import os
import io
import zipfile
import requests
import pandas as pd


def load_and_prepare_data(url, data_dir="data"):
    """
    Télécharge, extrait et prépare le dataset Sentiment140.
    Retourne un DataFrame pandas avec les colonnes 'sentiment' et 'text'.
    """

    # Crée le répertoire de données s'il n'existe pas
    os.makedirs(data_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "sentiment140.zip")
    csv_path = os.path.join(data_dir, "training.1600000.processed.noemoticon.csv")

    # Télécharger et extraire si le CSV n'existe pas
    if not os.path.exists(csv_path):
        print("Téléchargement du jeu de données...")
        response = requests.get(url)
        response.raise_for_status()  # Vérifie que la requête a réussi

        # Extraire le contenu du zip en mémoire
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(data_dir)

        print("Téléchargement et extraction terminés.")

    # Définir les colonnes et charger les données
    cols = ["sentiment", "id", "date", "query", "user", "text"]
    df = pd.read_csv(
        csv_path,
        header=None,
        names=cols,
        encoding="latin-1"  # Encodage spécifique à ce dataset
    )

    # Ne garder que les colonnes pertinentes
    df = df[["sentiment", "text"]]

    # Mapper les sentiments : 0 -> négatif, 4 -> positif (1)
    df["sentiment"] = df["sentiment"].replace({4: 1})

    print("Préparation des données terminée.")
    return df


if __name__ == "__main__":
    # URL directe vers le fichier zip du dataset
    dataset_url = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    # Pour des raisons de performance, on prend un échantillon de 50k tweets
    data_df = load_and_prepare_data(dataset_url).sample(n=50000, random_state=42)

    # Sauvegarder l’échantillon
    output_path = os.path.join("data", "raw_tweets.csv")
    data_df.to_csv(output_path, index=False)

    print(f"Échantillon de données sauvegardé dans {output_path}")
