"""Baixa e prepara o dataset público usado pelo laboratório.

Fonte original: UCI Machine Learning Repository — Diabetes 130-US Hospitals.
"""

from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

DATA_URL = "https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "diabetes_dataset.zip"
CSV_PATH = RAW_DIR / "diabetic_data.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists():
        print(f"Dataset já disponível em: {CSV_PATH}")
        return

    print("Baixando dataset público do UCI...")
    with urlopen(DATA_URL, timeout=60) as response:
        ZIP_PATH.write_bytes(response.read())

    print("Extraindo arquivo CSV...")
    with ZipFile(ZIP_PATH) as archive:
        members = [name for name in archive.namelist() if name.endswith("diabetic_data.csv")]
        if not members:
            raise RuntimeError("O arquivo diabetic_data.csv não foi encontrado no pacote do UCI.")
        with archive.open(members[0]) as source, CSV_PATH.open("wb") as target:
            target.write(source.read())

    ZIP_PATH.unlink(missing_ok=True)
    print(f"Dataset pronto em: {CSV_PATH}")


if __name__ == "__main__":
    main()
