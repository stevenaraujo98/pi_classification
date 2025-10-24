# %%
# %pip install requests tqdm sentence-transformers scikit-learn pandas
# %pip install sentence-transformers
# %pip install tf-keras
# Opcional si activas clasificador cero-shot:
# %pip install transformers torch

# %% [markdown]
# Evaluación de Patentabilidad: Novedad, Nivel Inventivo y Aplicación Industrial
# 
# Resumen:
# - Busca arte previo en:
#   - OpenAlex (literatura científica)
#   - PatentsView (patentes USA)
# - Calcula similitud semántica con embeddings multilingües
# - Emite reporte con:
#   1) Novedad (score y referencias más similares)
#   2) Nivel inventivo (índice de obviedad y análisis de clustering)
#   3) Aplicación industrial (CPC sugerido + clasificador por etiquetas opcional)
# 
# Avisos:
# - Respeta Términos de Uso de las APIs.
# - Ajusta umbrales según tu dominio.
# - Este script es un punto de partida. Valida resultados con expertos en PI.
# 
# Uso:
# python evaluar_patentabilidad.py --text "Descripción del proyecto..." --openalex_email "tu_email@dominio.com"
# 
# Parámetros clave:
# --max_results_openalex, --max_results_patents, --top_k, --threshold_novelty, --threshold_obvious_mean, etc.
# 
# Autor: GPT-5 (Abacus.AI)

# %%
import os
import re
import json
import time
import math
import argparse
import datetime as dt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv

# %%
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit("Falta sentence-transformers. Instala con: pip install sentence-transformers") from e

# %%
from gets_api import normalize_text, search_openalex, search_patentsview

# %%
load_dotenv()

# %%
@dataclass
class Reference:
    source: str                       # "openalex" | "patentsview"
    id: str
    title: str
    abstract: str
    url: Optional[str]
    date: Optional[str]
    authors_or_assignees: Optional[str] = None
    cpc_sections: List[str] = field(default_factory=list)
    cpc_groups: List[str] = field(default_factory=list)
    score: Optional[float] = None     # Similaridad final con el proyecto


# %% [markdown]
# # Embeddings y similitud

# %%
def load_embedder(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> SentenceTransformer:
    """
    Carga un modelo de embedding de oraciones.
    Modelo multilingüe recomendado para español/inglés.
    Alternativas:
      - sentence-transformers/distiluse-base-multilingual-cased-v2
      - sentence-transformers/all-MiniLM-L6-v2 (monolingüe inglés)
    """
    return SentenceTransformer(model_name)

# %%
def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """ 
    Embedding de una lista de textos. 
    Normaliza los embeddings para facilitar el cálculo de similitud coseno.
    """
    return np.asarray(embedder.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Similitud coseno entre dos conjuntos de embeddings normalizados. """
    return cosine_similarity(a, b)

# %% [markdown]
# # Heurísticas de decisión

# %%
def evaluate_novelty(max_sim: float, threshold_novelty: float = 0.80, borderline_band: Tuple[float, float] = (0.65, 0.80)) -> Dict[str, Any]:
    """
    Evaluación de novedad basada en la similitud máxima con el corpus previo.
    Si la similitud es mas baja que el umbral -> Novedad probable.

    Novedad alta ~ menor similitud con arte previo. Aquí invertimos la lógica:
    - Si max_sim >= threshold_novelty -> Probable NO novedad (muy parecido).
    - Si entre borderline -> Revisar (posible colisión parcial).
    - Si por debajo del borde -> Novedad probable.
    """
    lower, upper = borderline_band
    if max_sim >= threshold_novelty:
        status = "No novedoso (colisión probable)"
        risk = "ALTO"
    elif max_sim >= lower:
        status = "Revisar (zonas grises, semejanza moderada)"
        risk = "MEDIO"
    else:
        status = "Novedad probable"
        risk = "BAJO"
    novelty_score = 1.0 - max_sim  # más alto = más novedoso
    return {
        "status": status,
        "risk": risk,
        "max_similarity": round(max_sim, 4),
        "novelty_score": round(novelty_score, 4),
        "thresholds": {
            "no_novedad_si_sim_ge": threshold_novelty,
            "zona_gris": borderline_band
        }
    }

# %%
def evaluate_inventive_step(
    top_k_sims: List[float],
    cpc_groups_of_top: List[List[str]],
    obvious_mean_threshold: float = 0.60,
    multi_ref_obvious_threshold: float = 0.55,
    min_refs_for_obviousness: int = 3
) -> Dict[str, Any]:
    """
    Evaluación de nivel inventivo (no-obviedad).
    Si la similitud es alta con varios documentos o si hay múltiples referencias fuertes de diferentes grupos CPC,
    puede indicar obviedad.
    

    Heurística de nivel inventivo (no-obviedad). Ideas:
    - Si el promedio de top_k es alto, sugiere obviedad (mucha proximidad a varios documentos).
    - Si existen >= min_refs referencias con sim >= multi_ref_obvious_threshold provenientes de grupos CPC diferentes,
      puede indicar combinación obvia (riesgo para nivel inventivo).
    - Resultado: Alto/Medio/Bajo riesgo de falta de nivel inventivo.
    """
    if not top_k_sims:
        return {
            "status": "Insuficiente evidencia (pocas coincidencias)",
            "risk": "DESCONOCIDO",
            "mean_topk_similarity": None,
            "explanation": "No se hallaron suficientes vecinos para evaluar."
        }

    mean_sim = float(np.mean(top_k_sims))
    high_cluster = mean_sim >= obvious_mean_threshold # si el promedio de la similitud de los top-k mayor al umbral

    # Diversidad CPC entre los top-k
    flat_groups = [g for groups in cpc_groups_of_top for g in (groups or [])]
    unique_groups = set(flat_groups)
    strong_refs = sum(1 for s in top_k_sims if s >= multi_ref_obvious_threshold) # suma de la cantidad de referencias fuertes (similitud mas alta que el umbral)
    diverse = len(unique_groups) >= min(3, len(top_k_sims))  # diversidad mínima, si hay al menos 3 grupos CPC diferentes entre los top-k

    if high_cluster or (strong_refs >= min_refs_for_obviousness and diverse):
        status = "Riesgo: nivel inventivo bajo (obviedad probable)"
        risk = "ALTO"
    elif mean_sim >= (obvious_mean_threshold - 0.1) or strong_refs >= (min_refs_for_obviousness - 1):
        status = "Riesgo intermedio (posible obviedad)"
        risk = "MEDIO"
    else:
        status = "Nivel inventivo probable (no-obvio)"
        risk = "BAJO"

    return {
        "status": status,
        "risk": risk,
        "mean_topk_similarity": round(mean_sim, 4),
        "strong_ref_count": strong_refs,
        "unique_cpc_groups": list(sorted(unique_groups)),
        "explanation": "Promedio de similitudes y diversidad CPC considerados."
    }

# %%
CPC_SECTION_MAP = {
    "A": "Necesidades humanas",
    "B": "Técnicas industriales; transporte",
    "C": "Química; metalurgia",
    "D": "Textiles; papel",
    "E": "Construcciones fijas",
    "F": "Ingeniería mecánica; iluminación; calefacción; armas; voladura",
    "G": "Física (incluye computación)",
    "H": "Electricidad",
    "Y": "Tecnologías emergentes o transversales"
}

# %%
INDUSTRIAL_KEYWORDS = [
    "proceso", "procedimiento", "método", "fabricación", "producción",
    "dispositivo", "sistema", "aparato", "planta", "línea de", "automatización",
    "ensamblaje", "tratamiento", "reactor", "módulo", "unidad", "industrial", "piloto"
]

DEFAULT_SEMANTIC_LABELS = [
    "biotecnología", "química", "farmacéutica", "software", "IA/ML", "robótica",
    "energía", "mecánica", "eléctrica", "electrónica", "telecomunicaciones",
    "agricultura", "materiales", "médico", "construcción", "transporte",
    "minería", "textil", "alimentos"
]

# %%
def classify_industrial_app(
    project_text: str,
    nearest_patents: List[Reference],
    semantic_labels: Optional[List[str]] = None,
    min_keyword_hits: int = 1
) -> Dict[str, Any]:
    """
    Clasificador de aplicabilidad industrial:
    - Evidencia textual: presencia de términos industriales en el proyecto.
    - Sugerencia CPC: mayorías por sección/grupo en patentes más cercanas.
    - Etiquetas semánticas: sugerencias simples por similitud de palabras clave (opcional).

    nearest_patents: patentes con similitud más alta
    semantic_labels: etiquetas semánticas para sugerencias simples

    Nota: Puede ampliarse con cero-shot NLI (xnli) si se requiere.
    """
    text = project_text.lower()
    keyword_hits = [kw for kw in INDUSTRIAL_KEYWORDS if kw in text] # búsqueda simple de palabras clave de industria en el texto del proyecto
    evidence = len(keyword_hits) >= min_keyword_hits

    # Mayoría CPC por secciones y grupos
    section_counts: Dict[str, int] = {}
    group_counts: Dict[str, int] = {}
    for ref in nearest_patents:
        for s in (ref.cpc_sections or []):
            section_counts[s] = section_counts.get(s, 0) + 1
        for g in (ref.cpc_groups or []):
            group_counts[g] = group_counts.get(g, 0) + 1

    print("section_counts", section_counts)
    print("group_counts", group_counts)
    top_sections = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:5] # Top 5 secciones CPC
    top_groups = sorted(group_counts.items(), key=lambda x: x[1], reverse=True)[:10] # Top 10 grupos CPC

    # Etiquetas sugeridas de semántica simples: conteo de palabras
    sem_suggestions: List[Tuple[str, int]] = []
    if semantic_labels:
        for lab in semantic_labels:
            score = 0
            # Conteo simple de tokens coincidentes
            ###################################################
            for token in re.findall(r"[A-Za-zÁ-Úá-úñÑ0-9\-/]+", lab.lower()):
                if token and token in text:
                    score += 1
            ###################################################
            if score > 0:
                sem_suggestions.append((lab, score))
        sem_suggestions.sort(key=lambda x: x[1], reverse=True)
        sem_suggestions = sem_suggestions[:5]

    status = "Aplicación industrial probable" if evidence or top_sections else "Revisar (evidencia limitada)"
    return {
        "status": status,
        "evidence_keywords": keyword_hits,
        ###################################################
        "top_cpc_sections": [(s, c, CPC_SECTION_MAP.get(s, "")) for s, c in top_sections],
        ###################################################
        "top_cpc_groups": top_groups,
        "semantic_labels_suggested": sem_suggestions
    }

# %% [markdown]
# # Pipeline principal

# %%
def build_corpus_strings(refs: List[Reference]) -> List[str]:
    """
    Para embeddings: concatenar título + abstract de cada referencia.
    """
    corpus = []
    for r in refs:
        t = normalize_text(r.title)
        a = normalize_text(r.abstract)
        if t and a:
            corpus.append(f"{t}. {a}")
        else:
            corpus.append(t or a)
    return corpus

# %%
def evaluate_project(
    project_text: tuple[str, str],
    max_results_openalex: int = 50,
    max_results_patents: int = 100,
    embedder_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    top_k: int = 10,
    novelty_threshold: float = 0.80,
    borderline_band: Tuple[float, float] = (0.65, 0.80),
    obvious_mean_threshold: float = 0.60,
    multi_ref_obvious_threshold: float = 0.55,
    min_refs_for_obviousness: int = 3,
    semantic_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Orquestación completa: búsqueda, embeddings, similitudes y reporte.
    """
    project_title_text = normalize_text(project_text[0])
    project_abstract_text = normalize_text(project_text[1])
    project_completo_text = f"{project_title_text}{'' if project_title_text[-1] == '.' else '.'} {project_abstract_text}"

    if not project_text:
        raise ValueError("El texto del proyecto está vacío.")
    
    print("project_title_text", project_title_text)
    print("project_abstract_text", project_abstract_text)
    print("project_completo_text", project_completo_text)

    print("1) Buscando en OpenAlex...")
    oa_refs = search_openalex(
        query=project_title_text,
        max_results=max_results_openalex,
        per_page=50,
        from_year=None,
        to_year=None,
        lang_filter=None
    )

    print(f"   -> {len(oa_refs)} resultados de OpenAlex")


    print("2) Buscando en PatentsView...")
    pv_refs = search_patentsview(
        query_text=(project_title_text, project_abstract_text),
        max_results=max_results_patents,
        per_page=100
    )
    print(f"   -> {len(pv_refs)} resultados de PatentsView")

    
    # Construir corpus
    all_refs = oa_refs + pv_refs
    corpus = build_corpus_strings(all_refs)
    # ← VALIDACIÓN PARA CORPUS VACÍO
    if not corpus or len(corpus) == 0:
        print("⚠️  Advertencia: No se encontraron referencias en las APIs")
        return {
            "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            "input_project_excerpt": project_completo_text[:600] + ("..." if len(project_completo_text) > 600 else ""),
            "error": "No se encontraron referencias en OpenAlex ni PatentsView",
            "parameters": {
                "max_results_openalex": max_results_openalex,
                "max_results_patents": max_results_patents,
                "embedder_name": embedder_name,
                "top_k": top_k,
                "novelty_threshold": novelty_threshold,
                "borderline_band": borderline_band,
                "obvious_mean_threshold": obvious_mean_threshold,
                "multi_ref_obvious_threshold": multi_ref_obvious_threshold,
                "min_refs_for_obviousness": min_refs_for_obviousness
            },
            "modules": {
                "1_novedad": {
                    "status": "Sin datos para evaluar",
                    "risk": "DESCONOCIDO",
                    "max_similarity": 0.0,
                    "novelty_score": 1.0,
                    "note": "No se encontraron referencias para comparar"
                },
                "2_nivel_inventivo": {
                    "status": "Sin datos para evaluar",
                    "risk": "DESCONOCIDO",
                    "mean_topk_similarity": None,
                    "explanation": "No se encontraron referencias para comparar"
                },
                "3_aplicacion_industrial": {
                    "status": "Evaluación basada solo en palabras clave",
                    "evidence_keywords": [kw for kw in INDUSTRIAL_KEYWORDS if kw in project_completo_text.lower()],
                    "top_cpc_sections": [],
                    "top_cpc_groups": [],
                    "semantic_labels_suggested": []
                }
            },
            "top_references": []
        }

    # Embeddings
    print("3) Generando embeddings...")
    embedder = load_embedder(embedder_name)
    print("   Modelo de embedding cargado:", embedder_name)
    print("Tamaño del proyecto:", len(project_completo_text))
    proj_vec = embed_texts(embedder, [project_completo_text]) # Embedding del proyecto
    print("   Embedding del proyecto generado.")
    print("Tamaño del corpus:", len(corpus))
    corpus_vecs = embed_texts(embedder, corpus) # Embeddings del corpus
    print("   Embeddings del corpus generados.")

    # Similitudes
    sims = cosine_sim(proj_vec, corpus_vecs).flatten() # Similitudes coseno
    for ref, s in zip(all_refs, sims):
        ref.score = float(s)

    # Ordenar por similitud
    sorted_refs = sorted(all_refs, key=lambda r: r.score or 0.0, reverse=True) # Más similar primero, orden descendente
    max_sim = float(max(sims)) if len(sims) else 0.0 # Similitud máxima

    # 1) Novedad
    novelty = evaluate_novelty(
        max_sim=max_sim,
        threshold_novelty=novelty_threshold,
        borderline_band=borderline_band
    )

    # 2) Nivel inventivo
    top_k_refs = sorted_refs[:min(top_k, len(sorted_refs))]
    top_k_sims = [r.score or 0.0 for r in top_k_refs]
    top_k_cpc_groups = [r.cpc_groups for r in top_k_refs]
    inventive = evaluate_inventive_step(
        top_k_sims=top_k_sims,
        cpc_groups_of_top=top_k_cpc_groups,
        obvious_mean_threshold=obvious_mean_threshold,
        multi_ref_obvious_threshold=multi_ref_obvious_threshold,
        min_refs_for_obviousness=min_refs_for_obviousness
    )

    # 3) Aplicación industrial
    nearest_patents = [r for r in top_k_refs if r.source == "patentsview"]
    industrial = classify_industrial_app(
        project_text=project_completo_text,
        nearest_patents=nearest_patents,
        semantic_labels=semantic_labels or DEFAULT_SEMANTIC_LABELS
    )

    # Resumen de referencias principales (mezcla de artículos/patentes)
    def ref_to_dict(r: Reference) -> Dict[str, Any]:
        return {
            "source": r.source,
            "id": r.id,
            "title": r.title,
            "date": r.date,
            "url": r.url,
            "score": round(r.score or 0.0, 4),
            "cpc_sections": r.cpc_sections,
            "cpc_groups": r.cpc_groups,
            "by": r.authors_or_assignees,
            "abstract": r.abstract
        }

    top_refs_out = [ref_to_dict(r) for r in top_k_refs]

    # Reporte final
    report = {
        "timestamp": dt.datetime.utcnow().isoformat() + "Z",
        "input_project_excerpt": project_completo_text[:600] + ("..." if len(project_completo_text) > 600 else ""),
        "parameters": {
            "max_results_openalex": max_results_openalex,
            "max_results_patents": max_results_patents,
            "embedder_name": embedder_name,
            "top_k": top_k,
            "novelty_threshold": novelty_threshold,
            "borderline_band": borderline_band,
            "obvious_mean_threshold": obvious_mean_threshold,
            "multi_ref_obvious_threshold": multi_ref_obvious_threshold,
            "min_refs_for_obviousness": min_refs_for_obviousness
        },
        "modules": {
            "1_novedad": novelty,
            "2_nivel_inventivo": inventive,
            "3_aplicacion_industrial": industrial
        },
        "top_references": top_refs_out
    }

    return report


# %%
def report_to_markdown(report: Dict[str, Any]) -> str:
    """
    Convierte el reporte a Markdown legible.
    """
    md = []
    md.append(f"# Reporte de Patentabilidad")
    md.append(f"- Fecha (UTC): {report['timestamp']}")
    md.append("")
    md.append("## Resumen del Proyecto")
    md.append(f"{report['input_project_excerpt']}")
    md.append("")
    md.append("## Parámetros")
    for k, v in report.get("parameters", {}).items():
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## 1) Novedad")
    nov = report["modules"]["1_novedad"]
    md.append(f"- Estado: {nov['status']}")
    md.append(f"- Riesgo: {nov['risk']}")
    md.append(f"- Máxima similitud: {nov['max_similarity']}")
    md.append(f"- Puntaje de novedad (1 - max_sim): {nov['novelty_score']}")
    md.append("")
    md.append("## 2) Nivel Inventivo")
    inv = report["modules"]["2_nivel_inventivo"]
    md.append(f"- Estado: {inv['status']}")
    md.append(f"- Riesgo: {inv['risk']}")
    md.append(f"- Promedio similitud top-k: {inv.get('mean_topk_similarity')}")
    md.append(f"- Nº refs fuertes: {inv.get('strong_ref_count')}")
    md.append(f"- CPC grupos únicos: {', '.join(inv.get('unique_cpc_groups', []))}")
    md.append(f"- Nota: {inv.get('explanation')}")
    md.append("")
    md.append("## 3) Aplicación Industrial")
    ind = report["modules"]["3_aplicacion_industrial"]
    md.append(f"- Estado: {ind['status']}")
    kws = ind.get("evidence_keywords", [])
    md.append(f"- Palabras clave detectadas: {', '.join(kws) if kws else 'Ninguna'}")
    md.append("- CPC secciones sugeridas:")
    for s, c, label in ind.get("top_cpc_sections", []):
        md.append(f"  - {s} ({label}): {c}")
    md.append("- CPC grupos sugeridos:")
    for g, c in ind.get("top_cpc_groups", []):
        md.append(f"  - {g}: {c}")
    labs = ind.get("semantic_labels_suggested", [])
    if labs:
        md.append("- Etiquetas semánticas sugeridas:")
        for lab, sc in labs:
            md.append(f"  - {lab} (score:{sc})")
    md.append("")
    md.append("## Top referencias similares")
    for r in report.get("top_references", []):
        title = r.get("title") or "(sin título)"
        url = r.get("url") or ""
        source = r.get("source")
        score = r.get("score")
        date = r.get("date") or ""
        who = r.get("by") or ""
        md.append(f"- [{source}] {title} ({date}) | sim={score} | {who} | {url}")
    md.append("")
    return "\n".join(md)

# %%
def save_report_files(report: Dict[str, Any], out_prefix: str = "reporte_patentabilidad") -> Tuple[str, str]:
    """
    Guarda JSON y Markdown. Retorna rutas.
    """
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = f"result/{out_prefix}_{ts}.json"
    md_path = f"result/{out_prefix}_{ts}.md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_to_markdown(report))
    return json_path, md_path

# %%
title = "Detecting and Characterizing Group Interactions Using 3D Spatial Data to Enhance Human-Robot Engagement"
abstract = "As robotic systems become increasingly integrated into human environments, it is critical to develop advanced methods that enable them to interpret and respond to complex social dynamics. This work combines a YOLOv8-based human pose estimation approach with 3D Mean Shift clustering for the detection and analysis of behavioral characteristics in social groups, using 3D point clouds generated by the Intel® RealSense™D435i as a cost-effective alternative to LiDAR systems. Our proposed method achieves 97% accuracy in classifying social group geometric configurations (L, C, and I patterns) and demonstrates the value of depth information by reaching 50% precision in 3D group detection using adaptive clustering, significantly outperforming standard 2D approaches. Validation was conducted with 12 participants across 8 experimental scenarios, demonstrating robust estimation of body orientation (40° error), a key indicator for interaction analysis, while head direction estimation presented greater variability (70° error), both measured relative to the depth plane and compared against OptiTrack ground truth data. The framework processes 120 samples at 2–6m distances, achieving 70% torso orientation accuracy at 5m and identifying triadic L-shaped groups with F1-score=0.91. These results enable autonomous robots to quantify group centroids, analyze interaction patterns, and navigate dynamically using real-time convex hull approximations. The integration of accessible 3D perception with efficient processing could enhance human-robot interactions, demonstrating its feasibility in applications such as social robotics, healthcare, care environments, and service industries, where social adaptability and collaborative decision-making are essential."
text = (title, abstract)

# %%
# text = "Nuestro proyecto propone un método de fabricación de..."
input_file = "./data/...."

labels_file = ""
max_results_openalex = 50
max_results_patents = 100
embedder_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
top_k = 10
threshold_novelty = 0.8
borderline_low = 0.5
borderline_high = 0.8
borderline_band = (borderline_low, borderline_high)
obvious_mean_threshold = 0.6
multi_ref_obvious_threshold = 0.55
min_refs_for_obviousness = 3

# %%
if text:
    project_text = text
elif input_file and os.path.exists(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        project_text = f.read()
else:
    raise SystemExit("Proporciona --text o --input_file con el contenido del proyecto.")

# etiquetas semánticas para el analisis de aplicación industrial
semantic_labels = None
if labels_file and os.path.exists(labels_file):
    with open(labels_file, "r", encoding="utf-8") as f:
        semantic_labels = [line.strip() for line in f if line.strip()]

# %%
report = evaluate_project(
    project_text=project_text,
    max_results_openalex=max_results_openalex,
    max_results_patents=max_results_patents,
    embedder_name=embedder_name,
    top_k=top_k,
    novelty_threshold=threshold_novelty,
    borderline_band=(borderline_low, borderline_high),
    obvious_mean_threshold=obvious_mean_threshold,
    multi_ref_obvious_threshold=multi_ref_obvious_threshold,
    min_refs_for_obviousness=min_refs_for_obviousness,
    semantic_labels=semantic_labels
)

# %%
report.keys()

# %%
report["modules"]

# %%
report["top_references"]

# %%
json_path, md_path = save_report_files(report)
print(f"Reporte guardado en:\n- {json_path}\n- {md_path}")


