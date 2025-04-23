import os
import argparse
import subprocess
from pathlib import Path
import sys

# python .\convert_yolo_to_ncnn.py --weights .\yolov5\runs\train\exp5\weights\best.pt

def run(cmd):
    """Exécute la commande shell et affiche la sortie en cas d'erreur."""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de la commande: {cmd}\n{e.stderr}")
        sys.exit(1)

def main(weights_path):
    weights_path = os.path.abspath(weights_path)
    
    # Vérification de l'existence du fichier de weights
    if not os.path.exists(weights_path):
        print(f"Le fichier de weights n'existe pas : {weights_path}")
        sys.exit(1)
    
    root_dir = Path(__file__).parent.resolve()
    yolov5_dir = root_dir / "yolov5"
    output_dir = root_dir / "model" / "ncnn_best"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Définition des chemins pour l'ONNX et sa version simplifiée
    onnx_path = Path(weights_path).with_suffix(".onnx")
    onnx_simplified = onnx_path.with_name(onnx_path.stem + "-sim.onnx")
    param_path = output_dir / "best.param"
    bin_path = output_dir / "best.bin"

    # Étape 1 : Exporter en ONNX
    cmd_export = f"python {yolov5_dir}/export.py --weights {weights_path} --img 640 --batch 1 --include onnx"
    run(cmd_export)

    # Vérifier que le fichier ONNX a bien été créé
    if not onnx_path.exists():
        print(f"Erreur : le fichier ONNX n'a pas été généré : {onnx_path}")
        sys.exit(1)

    # Étape 2 : Simplifier le modèle ONNX
    cmd_simplify = f"python -m onnxsim {onnx_path} {onnx_simplified}"
    run(cmd_simplify)

    # Vérifier que le fichier simplifié a bien été créé
    if not onnx_simplified.exists():
        print(f"Erreur : le fichier ONNX simplifié n'a pas été généré : {onnx_simplified}")
        sys.exit(1)

    # Étape 3 : Convertir ONNX -> NCNN
    cmd_convert = f"onnx2ncnn {onnx_simplified} {param_path} {bin_path}"
    run(cmd_convert)

    print(f"\n✅ Conversion terminée. Fichiers NCNN sauvegardés dans : {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir un modèle YOLOv5 (.pt) vers NCNN via ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Chemin vers le fichier .pt")
    args = parser.parse_args()
    main(args.weights)
