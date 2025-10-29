"""
Split Dataset

Divide un dataset en conjuntos de entrenamiento, validación y prueba.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split


def find_dataset_files(data_dir: Path) -> List[str]:
    """
    Encontrar todos los grafos en el directorio.
    
    Args:
        data_dir: Directorio con grafos
        
    Returns:
        Lista de IDs de grafos
    """
    graph_files = list(data_dir.glob("graph_*.pkl"))
    graph_ids = [f.stem.replace("graph_", "") for f in graph_files]
    logger.info(f"Encontrados {len(graph_ids)} grafos")
    return graph_ids


def load_labels(data_dir: Path, graph_ids: List[str]) -> Dict[str, bool]:
    """
    Cargar etiquetas de grafos.
    
    Args:
        data_dir: Directorio con grafos
        graph_ids: IDs de grafos
        
    Returns:
        Diccionario {graph_id: is_attack}
    """
    labels = {}
    for graph_id in graph_ids:
        label_file = data_dir / f"label_{graph_id}.json"
        if label_file.exists():
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                labels[graph_id] = label_data.get('is_attack', False)
        else:
            # Asumir que es benigno si no hay etiqueta
            labels[graph_id] = False
    
    return labels


def split_dataset(
    graph_ids: List[str],
    labels: Dict[str, bool],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Dividir dataset en train/val/test.
    
    Args:
        graph_ids: IDs de grafos
        labels: Etiquetas de grafos
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para prueba
        stratify: Si usar estratificación por clase
        random_state: Semilla aleatoria
        
    Returns:
        Tupla (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # Preparar datos para split
    graph_ids = list(graph_ids)
    y = [labels[gid] for gid in graph_ids]
    
    # Verificar distribución
    n_attack = sum(y)
    n_benign = len(y) - n_attack
    logger.info(f"Dataset: {len(y)} grafos ({n_attack} ataques, {n_benign} benignos)")
    
    # Split estratificado
    if stratify and n_attack > 0 and n_benign > 0:
        # Train + (Val + Test)
        train_ids, temp_ids, train_y, temp_y = train_test_split(
            graph_ids,
            y,
            train_size=train_ratio,
            stratify=y,
            random_state=random_state
        )
        
        # Val + Test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1 - val_size),
            stratify=temp_y,
            random_state=random_state
        )
    else:
        # Split sin estratificación
        logger.warning("No se puede estratificar (clase única), usando split aleatorio")
        train_ids, temp_ids = train_test_split(
            graph_ids,
            train_size=train_ratio,
            random_state=random_state
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=(1 - val_size),
            random_state=random_state
        )
    
    logger.info(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    return train_ids, val_ids, test_ids


def copy_files(
    graph_ids: List[str],
    src_dir: Path,
    dst_dir: Path,
    split_name: str
) -> None:
    """
    Copiar archivos de grafos a directorio de destino.
    
    Args:
        graph_ids: IDs de grafos a copiar
        src_dir: Directorio origen
        dst_dir: Directorio destino
        split_name: Nombre del split (train/val/test)
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for graph_id in graph_ids:
        # Copiar graph
        src_graph = src_dir / f"graph_{graph_id}.pkl"
        dst_graph = dst_dir / f"graph_{graph_id}.pkl"
        shutil.copy2(src_graph, dst_graph)
        
        # Copiar features
        src_features = src_dir / f"features_{graph_id}.npy"
        dst_features = dst_dir / f"features_{graph_id}.npy"
        shutil.copy2(src_features, dst_features)
        
        # Copiar label
        src_label = src_dir / f"label_{graph_id}.json"
        dst_label = dst_dir / f"label_{graph_id}.json"
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
    
    logger.info(f"✓ Copiados {len(graph_ids)} grafos a {split_name}/")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Dividir dataset en train/val/test"
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Directorio con grafos'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Directorio de salida (default: input/../split)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Proporción para entrenamiento (default: 0.7)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Proporción para validación (default: 0.15)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Proporción para prueba (default: 0.15)'
    )
    
    parser.add_argument(
        '--no-stratify',
        action='store_true',
        help='No usar estratificación por clase'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validar proporciones
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        logger.error(f"Las proporciones deben sumar 1.0 (actual: {total})")
        return
    
    # Directorio de salida
    if args.output is None:
        args.output = args.input.parent / f"{args.input.name}_split"
    
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Proporciones: {args.train_ratio:.1%} / {args.val_ratio:.1%} / {args.test_ratio:.1%}")
    
    # Encontrar grafos
    graph_ids = find_dataset_files(args.input)
    if not graph_ids:
        logger.error("No se encontraron grafos")
        return
    
    # Cargar etiquetas
    labels = load_labels(args.input, graph_ids)
    
    # Dividir dataset
    train_ids, val_ids, test_ids = split_dataset(
        graph_ids=graph_ids,
        labels=labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=not args.no_stratify,
        random_state=args.seed
    )
    
    # Copiar archivos
    copy_files(train_ids, args.input, args.output / "train", "train")
    copy_files(val_ids, args.input, args.output / "val", "val")
    copy_files(test_ids, args.input, args.output / "test", "test")
    
    # Guardar metadata
    metadata = {
        'total_graphs': len(graph_ids),
        'train_graphs': len(train_ids),
        'val_graphs': len(val_ids),
        'test_graphs': len(test_ids),
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'stratified': not args.no_stratify,
        'random_seed': args.seed,
        'attack_count': sum(labels.values()),
        'benign_count': len(labels) - sum(labels.values()),
    }
    
    with open(args.output / "split_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset dividido exitosamente!")
    print(f"\n📊 Estadísticas:")
    print(f"   Total: {len(graph_ids)} grafos")
    print(f"   Train: {len(train_ids)} grafos ({len(train_ids)/len(graph_ids):.1%})")
    print(f"   Val:   {len(val_ids)} grafos ({len(val_ids)/len(graph_ids):.1%})")
    print(f"   Test:  {len(test_ids)} grafos ({len(test_ids)/len(graph_ids):.1%})")
    print(f"\n   Ataques: {sum(labels.values())} ({sum(labels.values())/len(labels):.1%})")
    print(f"   Benignos: {len(labels) - sum(labels.values())} ({(len(labels) - sum(labels.values()))/len(labels):.1%})")
    print(f"\n📁 Dataset guardado en: {args.output}")
    print(f"\n📝 Próximo paso:")
    print(f"   Entrenar modelo: python scripts/train_detector.py --data {args.output}")


if __name__ == "__main__":
    main()
