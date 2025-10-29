"""
Dashboard Interactivo de Detección APT

Muestra resultados de detección en tiempo real con:
- Visualización de scores
- Estadísticas de rendimiento
- Análisis de confianza
- Recomendaciones de respuesta
"""

import sys
from pathlib import Path
import json
from typing import Dict, List

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_banner():
    """Banner del dashboard"""
    print("\n" + "="*80)
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║              🛡️  CAUSALDEFEND APT DETECTION DASHBOARD  🛡️           ║
    ║                                                                   ║
    ║          Sistema de Detección de Amenazas Persistentes           ║
    ║               Basado en Redes Neuronales Causales                ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    print("="*80)


def load_results():
    """Cargar resultados de evaluación"""
    results_file = Path("models/evaluation_results.json")
    
    if not results_file.exists():
        print("\n⚠️  No se encontraron resultados de evaluación.")
        print("💡 Ejecuta primero: python examples/test_detector_advanced.py")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results


def print_progress_bar(value: float, max_value: float = 100, width: int = 40):
    """Imprimir barra de progreso"""
    percentage = min(100, (value / max_value) * 100)
    filled = int(width * percentage / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percentage:.1f}%"


def show_performance_metrics(results: Dict):
    """Mostrar métricas de rendimiento"""
    print("\n" + "─"*80)
    print("📊 MÉTRICAS DE RENDIMIENTO DEL MODELO")
    print("─"*80 + "\n")
    
    metrics = [
        ("Accuracy", results['accuracy'] * 100, "Precisión general del modelo"),
        ("Precision", results['precision'] * 100, "Fiabilidad de detecciones positivas"),
        ("Recall", results['recall'] * 100, "Cobertura de ataques reales"),
        ("F1 Score", results['f1'] * 100, "Balance entre Precision y Recall"),
    ]
    
    for name, value, description in metrics:
        bar = print_progress_bar(value)
        print(f"{name:12s} {bar}  {value:5.1f}%")
        print(f"{'':12s} └─ {description}")
        print()


def show_confusion_matrix(results: Dict):
    """Mostrar matriz de confusión mejorada"""
    cm = results['confusion_matrix']
    tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']
    total = tp + tn + fp + fn
    
    print("\n" + "─"*80)
    print("📈 MATRIZ DE CONFUSIÓN")
    print("─"*80 + "\n")
    
    # Matriz visual
    print("                    PREDICCIÓN DEL MODELO")
    print("                 ┌─────────────┬─────────────┐")
    print(f"                 │   NORMAL    │   ATAQUE    │")
    print("     ┌───────────┼─────────────┼─────────────┤")
    print(f"     │  NORMAL   │  TN: {tn:3d}    │  FP: {fp:3d}    │  ← Verdaderos negativos / Falsos positivos")
    print(f" R   │           │  {tn/total*100:5.1f}%     │  {fp/total*100:5.1f}%     │")
    print("  E  ├───────────┼─────────────┼─────────────┤")
    print(f"   A │  ATAQUE   │  FN: {fn:3d}    │  TP: {tp:3d}    │  ← Falsos negativos / Verdaderos positivos")
    print(f"    L│           │  {fn/total*100:5.1f}%     │  {tp/total*100:5.1f}%     │")
    print("     └───────────┴─────────────┴─────────────┘")
    
    print(f"\n  📊 Interpretación:")
    print(f"     • TN ({tn}): Tráfico normal correctamente clasificado")
    print(f"     • TP ({tp}): Ataques correctamente detectados")
    print(f"     • FP ({fp}): Falsos positivos (falsa alarma) {'⚠️  CRÍTICO' if fp > 0 else '✓ Excelente'}")
    print(f"     • FN ({fn}): Ataques no detectados {'❌ PELIGROSO' if fn > 0 else '✓ Perfecto'}")


def show_threat_intelligence(results: Dict):
    """Mostrar inteligencia de amenazas"""
    cm = results['confusion_matrix']
    tp, fn = cm['tp'], cm['fn']
    
    print("\n" + "─"*80)
    print("🎯 INTELIGENCIA DE AMENAZAS")
    print("─"*80 + "\n")
    
    total_threats = tp + fn
    detected_rate = (tp / total_threats * 100) if total_threats > 0 else 0
    missed_rate = (fn / total_threats * 100) if total_threats > 0 else 0
    
    print(f"  Total de amenazas analizadas: {total_threats}")
    print(f"  ✅ Detectadas: {tp} ({detected_rate:.1f}%)")
    print(f"  ❌ No detectadas: {fn} ({missed_rate:.1f}%)")
    
    print(f"\n  🔍 Análisis de Riesgo:")
    
    if detected_rate >= 95:
        risk_level = "🟢 BAJO"
        recommendation = "Excelente cobertura. Mantener monitoreo continuo."
    elif detected_rate >= 85:
        risk_level = "🟡 MEDIO"
        recommendation = "Buena cobertura. Considerar ajuste de threshold."
    elif detected_rate >= 70:
        risk_level = "🟠 ALTO"
        recommendation = "Cobertura aceptable. Revisar ataques no detectados."
    else:
        risk_level = "🔴 CRÍTICO"
        recommendation = "Cobertura insuficiente. Re-entrenar modelo urgente."
    
    print(f"     Nivel de Riesgo: {risk_level}")
    print(f"     Recomendación: {recommendation}")


def show_detection_threshold(results: Dict):
    """Mostrar información del threshold"""
    threshold = results['threshold']
    
    print("\n" + "─"*80)
    print("⚖️  THRESHOLD DE DETECCIÓN")
    print("─"*80 + "\n")
    
    print(f"  Threshold actual: {threshold:.4f}")
    print(f"  ├─ Scores > {threshold:.4f} → Clasificados como ATAQUE 🚨")
    print(f"  └─ Scores ≤ {threshold:.4f} → Clasificados como NORMAL ✓")
    
    print(f"\n  📋 Ajuste del Threshold:")
    print(f"     • Aumentar → Menos falsos positivos, pero más ataques perdidos")
    print(f"     • Disminuir → Más detecciones, pero más falsas alarmas")
    print(f"     • Óptimo → Balance entre precision y recall (F1 Score máximo)")


def show_recommendations():
    """Mostrar recomendaciones operacionales"""
    print("\n" + "─"*80)
    print("💡 RECOMENDACIONES OPERACIONALES")
    print("─"*80 + "\n")
    
    recommendations = [
        ("🔄 Actualización de Modelo", [
            "Re-entrenar con datos recientes cada 2-4 semanas",
            "Incorporar nuevos patrones de ataque APT",
            "Validar con threat intelligence feeds actuales",
        ]),
        ("📊 Monitoreo Continuo", [
            "Revisar dashboards cada 4-8 horas",
            "Configurar alertas para detecciones de alta confianza",
            "Mantener log de falsos positivos para ajustes",
        ]),
        ("🛡️ Respuesta a Incidentes", [
            "Investigar inmediatamente scores > threshold * 2",
            "Cuarentena automática para scores críticos",
            "Análisis causal para entender cadenas de ataque",
        ]),
        ("🔬 Mejora Continua", [
            "Etiquetar detecciones manualmente para feedback",
            "Analizar ataques no detectados (falsos negativos)",
            "Ajustar features basado en nuevas técnicas APT",
        ]),
    ]
    
    for title, items in recommendations:
        print(f"  {title}:")
        for item in items:
            print(f"     • {item}")
        print()


def show_system_status():
    """Mostrar estado del sistema"""
    print("\n" + "─"*80)
    print("⚙️  ESTADO DEL SISTEMA")
    print("─"*80 + "\n")
    
    # Verificar archivos críticos
    checks = [
        ("Modelo Entrenado", Path("models/detector.ckpt").exists()),
        ("CI Tester", Path("models/ci_tester.ckpt").exists()),
        ("Dataset de Test", Path("data/processed/test").exists()),
        ("Resultados de Evaluación", Path("models/evaluation_results.json").exists()),
    ]
    
    print("  Componentes del Sistema:")
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"     {icon} {name:30s} {'OK' if status else 'FALTANTE'}")
    
    all_ok = all(status for _, status in checks)
    
    print(f"\n  Estado General: {'🟢 OPERACIONAL' if all_ok else '🔴 INCOMPLETO'}")


def main():
    """Dashboard principal"""
    print_banner()
    
    # Verificar sistema
    show_system_status()
    
    # Cargar resultados
    results = load_results()
    
    if results is None:
        print("\n" + "="*80)
        return
    
    # Mostrar secciones
    show_performance_metrics(results)
    show_confusion_matrix(results)
    show_threat_intelligence(results)
    show_detection_threshold(results)
    show_recommendations()
    
    # Footer
    print("\n" + "="*80)
    print(" "*25 + "FIN DEL REPORTE")
    print("="*80)
    
    print("\n📁 Archivos Generados:")
    print("   • models/evaluation_results.json - Métricas detalladas")
    print("   • models/detector.ckpt - Modelo entrenado")
    
    print("\n🚀 Próximas Acciones:")
    print("   1. Integrar con SIEM para alertas en tiempo real")
    print("   2. Configurar pipeline de detección automática")
    print("   3. Implementar respuesta automática a amenazas")
    print("   4. Generar reportes ejecutivos periódicos")
    print()


if __name__ == "__main__":
    main()
