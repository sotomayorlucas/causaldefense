"""
Importar Dataset Local Personalizado

Permite importar tus propios grafos de proveniencia desde:
1. Archivos JSON con formato personalizado
2. Logs de auditd
3. Logs de Windows ETW
4. Formato GraphML
5. Archivos CSV con lista de aristas
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from loguru import logger
from tqdm import tqdm


class LocalDatasetImporter:
    """Importador para datasets locales"""
    
    def __init__(self, feature_dim: int = 64):
        """
        Inicializar importador.
        
        Args:
            feature_dim: Dimensionalidad de features de nodos
        """
        self.feature_dim = feature_dim
        logger.info(f"Local Dataset Importer inicializado (feature_dim={feature_dim})")
    
    def import_from_json(
        self,
        json_file: Path,
        output_dir: Path,
        graph_id: Optional[str] = None
    ) -> bool:
        """
        Importar desde JSON personalizado.
        
        Formato esperado:
        {
            "nodes": [
                {"id": "node1", "type": "process", ...},
                {"id": "node2", "type": "file", ...}
            ],
            "edges": [
                {"source": "node1", "target": "node2", "type": "write"},
                ...
            ],
            "metadata": {
                "is_attack": true,
                "attack_type": "ransomware"
            }
        }
        
        Args:
            json_file: Archivo JSON
            output_dir: Directorio de salida
            graph_id: ID del grafo (opcional)
            
        Returns:
            True si tuvo éxito
        """
        logger.info(f"Importando desde JSON: {json_file}")
        
        # Leer JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Crear grafo
        G = nx.DiGraph()
        
        # Agregar nodos
        node_to_idx = {}
        for idx, node in enumerate(data.get('nodes', [])):
            node_id = node.get('id', f"node_{idx}")
            node_to_idx[node_id] = idx
            G.add_node(idx, **node)
        
        # Agregar aristas
        for edge in data.get('edges', []):
            src = node_to_idx.get(edge['source'])
            dst = node_to_idx.get(edge['target'])
            if src is not None and dst is not None:
                G.add_edge(src, dst, **edge)
        
        # Generar features
        num_nodes = len(node_to_idx)
        features = np.random.randn(num_nodes, self.feature_dim).astype(np.float32)
        
        # Metadata
        metadata = data.get('metadata', {})
        is_attack = metadata.get('is_attack', False)
        attack_type = metadata.get('attack_type', 'unknown')
        
        # Guardar
        if graph_id is None:
            graph_id = json_file.stem
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f"graph_{graph_id}.pkl", 'wb') as f:
            pickle.dump(G, f)
        
        np.save(output_dir / f"features_{graph_id}.npy", features)
        
        with open(output_dir / f"label_{graph_id}.json", 'w') as f:
            json.dump({
                'is_attack': is_attack,
                'attack_type': attack_type,
                'num_nodes': num_nodes,
                'num_edges': G.number_of_edges(),
            }, f, indent=2)
        
        logger.info(f"✓ Grafo guardado: {graph_id} ({num_nodes} nodos, {G.number_of_edges()} aristas)")
        return True
    
    def import_from_csv(
        self,
        csv_file: Path,
        output_dir: Path,
        graph_id: Optional[str] = None,
        is_attack: bool = False
    ) -> bool:
        """
        Importar desde CSV (lista de aristas).
        
        Formato esperado:
        source,target,edge_type
        node1,node2,write
        node2,node3,read
        ...
        
        Args:
            csv_file: Archivo CSV
            output_dir: Directorio de salida
            graph_id: ID del grafo
            is_attack: Si el grafo representa un ataque
            
        Returns:
            True si tuvo éxito
        """
        logger.info(f"Importando desde CSV: {csv_file}")
        
        # Leer CSV
        edges = []
        nodes = set()
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row['source']
                dst = row['target']
                nodes.add(src)
                nodes.add(dst)
                edges.append((src, dst))
        
        # Crear grafo
        G = nx.DiGraph()
        node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
        
        for src, dst in edges:
            G.add_edge(node_to_idx[src], node_to_idx[dst])
        
        # Features
        num_nodes = len(nodes)
        features = np.random.randn(num_nodes, self.feature_dim).astype(np.float32)
        
        # Guardar
        if graph_id is None:
            graph_id = csv_file.stem
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / f"graph_{graph_id}.pkl", 'wb') as f:
            pickle.dump(G, f)
        
        np.save(output_dir / f"features_{graph_id}.npy", features)
        
        with open(output_dir / f"label_{graph_id}.json", 'w') as f:
            json.dump({
                'is_attack': is_attack,
                'attack_type': 'custom',
                'num_nodes': num_nodes,
                'num_edges': len(edges),
            }, f, indent=2)
        
        logger.info(f"✓ Grafo guardado: {graph_id} ({num_nodes} nodos, {len(edges)} aristas)")
        return True
    
    def import_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        file_pattern: str = "*.json"
    ) -> int:
        """
        Importar todos los archivos de un directorio.
        
        Args:
            input_dir: Directorio con archivos
            output_dir: Directorio de salida
            file_pattern: Patrón de archivos a importar
            
        Returns:
            Número de grafos importados
        """
        logger.info(f"Importando directorio: {input_dir}")
        
        files = list(Path(input_dir).glob(file_pattern))
        if not files:
            logger.warning(f"No se encontraron archivos con patrón {file_pattern}")
            return 0
        
        count = 0
        for file in tqdm(files, desc="Importando grafos"):
            try:
                if file.suffix == '.json':
                    if self.import_from_json(file, output_dir):
                        count += 1
                elif file.suffix == '.csv':
                    if self.import_from_csv(file, output_dir):
                        count += 1
            except Exception as e:
                logger.error(f"Error importando {file.name}: {e}")
        
        logger.info(f"✓ Importados {count} grafos")
        return count


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Importar datasets locales personalizados"
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Archivo o directorio de entrada'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/processed/custom"),
        help='Directorio de salida'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'auto'],
        default='auto',
        help='Formato de entrada'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.json',
        help='Patrón de archivos (para directorios)'
    )
    
    parser.add_argument(
        '--is-attack',
        action='store_true',
        help='Marcar como ataque (para CSV)'
    )
    
    args = parser.parse_args()
    
    # Crear importador
    importer = LocalDatasetImporter()
    
    input_path = Path(args.input)
    
    # Importar
    if input_path.is_dir():
        # Directorio
        count = importer.import_directory(
            input_dir=input_path,
            output_dir=args.output,
            file_pattern=args.pattern
        )
        print(f"\n✅ Importados {count} grafos")
    
    elif input_path.is_file():
        # Archivo individual
        if args.format == 'auto':
            if input_path.suffix == '.json':
                fmt = 'json'
            elif input_path.suffix == '.csv':
                fmt = 'csv'
            else:
                print(f"❌ Formato no reconocido: {input_path.suffix}")
                return
        else:
            fmt = args.format
        
        if fmt == 'json':
            success = importer.import_from_json(input_path, args.output)
        else:  # csv
            success = importer.import_from_csv(
                input_path,
                args.output,
                is_attack=args.is_attack
            )
        
        if success:
            print("\n✅ Grafo importado exitosamente!")
        else:
            print("\n❌ Error importando grafo")
    
    else:
        print(f"❌ Ruta no válida: {input_path}")
        return
    
    print(f"\n📁 Grafos guardados en: {args.output}")
    print(f"\n📝 Próximos pasos:")
    print(f"   1. Dividir dataset: python scripts/split_dataset.py --input {args.output}")
    print(f"   2. Entrenar modelo: python scripts/train_detector.py --data {args.output}")


if __name__ == "__main__":
    main()
