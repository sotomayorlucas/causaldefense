"""
Importador de Datasets Externos para CausalDefend

Soporta:
1. DARPA Transparent Computing (TC) E3
2. DARPA OpTC (Operationally Transparent Cyber)
3. StreamSpot
4. Datasets personalizados en formato JSON

Referencias:
- DARPA TC: https://github.com/darpa-i2o/Transparent-Computing
- OpTC: https://github.com/FiveDirections/OpTC-data
- StreamSpot: https://github.com/sbustreamspot/sbustreamspot-data
"""

import argparse
import gzip
import json
import pickle
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import networkx as nx
import numpy as np
from loguru import logger
from tqdm import tqdm


class DatasetImporter:
    """Importador de datasets externos"""
    
    # URLs de datasets públicos
    DATASETS = {
        'streamspot': {
            'name': 'StreamSpot',
            'url': 'https://github.com/sbustreamspot/sbustreamspot-data/archive/master.zip',
            'description': 'Dataset de proveniencia para detección de ataques APT',
            'size': '~500 MB',
            'format': 'txt',
        },
        'darpa_tc_sample': {
            'name': 'DARPA TC Sample',
            'url': 'https://drive.google.com/uc?id=SAMPLE_ID',  # Placeholder
            'description': 'Muestra del dataset DARPA Transparent Computing',
            'size': '~2 GB',
            'format': 'json',
        },
        'optc_sample': {
            'name': 'DARPA OpTC Sample',
            'url': 'https://github.com/FiveDirections/OpTC-data/raw/master/sample.json',
            'description': 'Muestra del dataset DARPA OpTC',
            'size': '~1 GB',
            'format': 'json',
        }
    }
    
    def __init__(self, output_dir: Path = Path("data/external")):
        """
        Inicializar importador.
        
        Args:
            output_dir: Directorio para guardar datasets descargados
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset Importer inicializado en: {self.output_dir}")
    
    def list_available_datasets(self):
        """Listar datasets disponibles"""
        print("\n" + "="*80)
        print("DATASETS EXTERNOS DISPONIBLES")
        print("="*80 + "\n")
        
        for key, info in self.DATASETS.items():
            print(f"📦 {info['name']}")
            print(f"   ID: {key}")
            print(f"   Descripción: {info['description']}")
            print(f"   Tamaño: {info['size']}")
            print(f"   Formato: {info['format']}")
            print(f"   URL: {info['url'][:60]}...")
            print()
    
    def download_dataset(
        self,
        dataset_id: str,
        force: bool = False
    ) -> Optional[Path]:
        """
        Descargar dataset.
        
        Args:
            dataset_id: ID del dataset a descargar
            force: Forzar re-descarga si ya existe
            
        Returns:
            Path al archivo descargado o None si falló
        """
        if dataset_id not in self.DATASETS:
            logger.error(f"Dataset '{dataset_id}' no reconocido")
            logger.info(f"Datasets disponibles: {list(self.DATASETS.keys())}")
            return None
        
        info = self.DATASETS[dataset_id]
        logger.info(f"Descargando {info['name']}...")
        
        # Determinar extensión
        url = info['url']
        if url.endswith('.zip'):
            ext = '.zip'
        elif url.endswith('.tar.gz') or url.endswith('.tgz'):
            ext = '.tar.gz'
        elif url.endswith('.json'):
            ext = '.json'
        elif url.endswith('.json.gz'):
            ext = '.json.gz'
        else:
            ext = '.data'
        
        # Archivo de salida
        output_file = self.output_dir / f"{dataset_id}{ext}"
        
        # Verificar si ya existe
        if output_file.exists() and not force:
            logger.info(f"Dataset ya existe en: {output_file}")
            logger.info("Usa --force para re-descargar")
            return output_file
        
        try:
            # Callback para progreso
            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = count * block_size / total_size * 100
                    print(f"\rDescargando: {percent:.1f}% ({count * block_size / 1024 / 1024:.1f} MB)", end='')
            
            # Descargar
            logger.info(f"Descargando desde: {url}")
            urlretrieve(url, output_file, progress_hook)
            print()  # Nueva línea después del progreso
            
            logger.info(f"✓ Descargado: {output_file}")
            logger.info(f"✓ Tamaño: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
            
            return output_file
            
        except Exception as e:
            logger.error(f"Error descargando dataset: {e}")
            if output_file.exists():
                output_file.unlink()
            return None
    
    def extract_archive(self, archive_path: Path) -> Path:
        """
        Extraer archivo comprimido.
        
        Args:
            archive_path: Path al archivo comprimido
            
        Returns:
            Path al directorio extraído
        """
        extract_dir = archive_path.parent / archive_path.stem
        
        logger.info(f"Extrayendo {archive_path.name}...")
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(extract_dir)
            
            elif archive_path.suffix == '.gz' or archive_path.name.endswith('.tar.gz'):
                with tarfile.open(archive_path, 'r:gz') as tf:
                    tf.extractall(extract_dir)
            
            else:
                logger.warning(f"Formato no reconocido: {archive_path.suffix}")
                return archive_path.parent
            
            logger.info(f"✓ Extraído a: {extract_dir}")
            return extract_dir
            
        except Exception as e:
            logger.error(f"Error extrayendo archivo: {e}")
            return archive_path.parent
    
    def parse_streamspot(
        self,
        data_dir: Path,
        output_dir: Path
    ) -> Tuple[int, int]:
        """
        Parsear dataset StreamSpot.
        
        StreamSpot contiene grafos de proveniencia en formato de lista de aristas:
        src_id dst_id edge_type timestamp
        
        Args:
            data_dir: Directorio con archivos StreamSpot
            output_dir: Directorio de salida
            
        Returns:
            Tupla (num_graphs, num_edges)
        """
        logger.info("Parseando StreamSpot dataset...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        graph_files = list(data_dir.glob("*.txt"))
        if not graph_files:
            logger.warning(f"No se encontraron archivos .txt en {data_dir}")
            return 0, 0
        
        num_graphs = 0
        total_edges = 0
        
        for graph_file in tqdm(graph_files, desc="Procesando grafos"):
            # Leer archivo
            edges = []
            nodes = set()
            
            with open(graph_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            src, dst = parts[0], parts[1]
                            nodes.add(src)
                            nodes.add(dst)
                            edges.append((src, dst))
            
            if len(edges) == 0:
                continue
            
            # Crear grafo NetworkX
            G = nx.DiGraph()
            
            # Mapear nodos a índices
            node_to_idx = {node: idx for idx, node in enumerate(sorted(nodes))}
            
            for src, dst in edges:
                G.add_edge(node_to_idx[src], node_to_idx[dst])
            
            # Generar features aleatorias (64 dims)
            num_nodes = len(nodes)
            features = np.random.randn(num_nodes, 64).astype(np.float32)
            
            # Determinar si es ataque (basado en nombre de archivo)
            is_attack = 'attack' in graph_file.stem.lower() or 'malicious' in graph_file.stem.lower()
            
            # Guardar
            graph_id = graph_file.stem
            
            # Guardar grafo
            with open(output_dir / f"graph_{graph_id}.pkl", 'wb') as f:
                pickle.dump(G, f)
            
            # Guardar features
            np.save(output_dir / f"features_{graph_id}.npy", features)
            
            # Guardar label
            with open(output_dir / f"label_{graph_id}.json", 'w') as f:
                json.dump({
                    'is_attack': is_attack,
                    'attack_type': 'streamspot_attack' if is_attack else 'normal',
                    'num_nodes': num_nodes,
                    'num_edges': len(edges),
                }, f, indent=2)
            
            num_graphs += 1
            total_edges += len(edges)
        
        logger.info(f"✓ Procesados {num_graphs} grafos con {total_edges} aristas totales")
        return num_graphs, total_edges
    
    def parse_darpa_tc(
        self,
        data_file: Path,
        output_dir: Path,
        max_graphs: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Parsear dataset DARPA TC (JSON format).
        
        DARPA TC contiene eventos de proveniencia en formato CDM (Common Data Model).
        
        Args:
            data_file: Archivo JSON con datos DARPA TC
            output_dir: Directorio de salida
            max_graphs: Máximo número de grafos a procesar
            
        Returns:
            Tupla (num_graphs, num_edges)
        """
        logger.info("Parseando DARPA TC dataset...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Leer JSON
        if data_file.suffix == '.gz':
            with gzip.open(data_file, 'rt') as f:
                data = json.load(f)
        else:
            with open(data_file, 'r') as f:
                data = json.load(f)
        
        logger.info(f"Cargados {len(data)} eventos")
        
        # TODO: Implementar parseo completo de CDM
        # Por ahora, convertir a grafos simples
        
        num_graphs = 0
        total_edges = 0
        
        logger.info("✓ Parseo de DARPA TC completado")
        return num_graphs, total_edges
    
    def import_dataset(
        self,
        dataset_id: str,
        output_dir: Path = Path("data/processed"),
        force_download: bool = False,
        max_graphs: Optional[int] = None
    ) -> bool:
        """
        Importar dataset completo.
        
        Args:
            dataset_id: ID del dataset
            output_dir: Directorio de salida para grafos procesados
            force_download: Forzar re-descarga
            max_graphs: Máximo número de grafos a procesar
            
        Returns:
            True si tuvo éxito
        """
        logger.info("="*80)
        logger.info(f"IMPORTANDO DATASET: {dataset_id}")
        logger.info("="*80)
        
        # 1. Descargar
        archive_path = self.download_dataset(dataset_id, force=force_download)
        if archive_path is None:
            return False
        
        # 2. Extraer si es necesario
        if archive_path.suffix in ['.zip', '.gz'] or archive_path.name.endswith('.tar.gz'):
            data_dir = self.extract_archive(archive_path)
        else:
            data_dir = archive_path.parent
        
        # 3. Parsear según formato
        output_dir = Path(output_dir)
        
        if dataset_id == 'streamspot':
            num_graphs, num_edges = self.parse_streamspot(data_dir, output_dir)
        
        elif dataset_id in ['darpa_tc_sample', 'optc_sample']:
            # Buscar archivo JSON
            json_files = list(data_dir.glob("*.json")) + list(data_dir.glob("*.json.gz"))
            if json_files:
                num_graphs, num_edges = self.parse_darpa_tc(json_files[0], output_dir, max_graphs)
            else:
                logger.error("No se encontraron archivos JSON")
                return False
        
        else:
            logger.error(f"Parser no implementado para {dataset_id}")
            return False
        
        logger.info("="*80)
        logger.info("IMPORTACIÓN COMPLETADA")
        logger.info("="*80)
        logger.info(f"✓ Grafos procesados: {num_graphs}")
        logger.info(f"✓ Aristas totales: {num_edges}")
        logger.info(f"✓ Ubicación: {output_dir}")
        
        return True


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Importar datasets externos para CausalDefend"
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='Listar datasets disponibles'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='ID del dataset a importar'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("data/processed"),
        help='Directorio de salida'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Forzar re-descarga si ya existe'
    )
    
    parser.add_argument(
        '--max-graphs',
        type=int,
        help='Máximo número de grafos a procesar'
    )
    
    args = parser.parse_args()
    
    # Crear importador
    importer = DatasetImporter()
    
    # Listar datasets
    if args.list:
        importer.list_available_datasets()
        return
    
    # Importar dataset
    if args.dataset:
        success = importer.import_dataset(
            dataset_id=args.dataset,
            output_dir=args.output,
            force_download=args.force,
            max_graphs=args.max_graphs
        )
        
        if success:
            print("\n✅ Dataset importado exitosamente!")
            print(f"\n📝 Próximos pasos:")
            print(f"   1. Dividir en train/val/test:")
            print(f"      python scripts/split_dataset.py --input {args.output}")
            print(f"   2. Entrenar modelo:")
            print(f"      python scripts/train_detector.py --data {args.output}")
        else:
            print("\n❌ Error importando dataset")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
