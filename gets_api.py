from typing import List, Dict, Any, Optional, Tuple
import requests
from dataclasses import dataclass, field
import os
import time
from dotenv import load_dotenv
import re
import json

load_dotenv()
OPENALEX_BASE = os.getenv("OPENALEX_BASE")
PATENTSVIEW_BASE = os.getenv("PATENTSVIEW_BASE")
PATENTSVIEW_KEY = os.getenv("PATENTSVIEW_KEY")

@dataclass
class Reference:
    source: str                       # "openalex" | "patentsview"
    id: str
    title: str
    abstract: str
    url: Optional[str]
    date: Optional[str]
    authors_or_assignees: Optional[str] = None
    ###################################################
    cpc_sections: List[str] = field(default_factory=list) # Secciones CPC asociadas
    cpc_groups: List[str] = field(default_factory=list) # Grupos CPC asociados
    ###################################################
    score: Optional[float] = None     # Similaridad final con el proyecto
    doi: Optional[str] = None
    keywords: Optional[str] = None
    concept_names: Optional[str] = None

def normalize_text(s: str) -> str:
    """
    Normaliza texto: elimina saltos de línea y espacios extra.
    """
    if not s:
        return ""
    s = s.replace("\n", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def abstract_from_openalex_inv_idx(inv_idx: Dict[str, List[int]]) -> str:
    """
    Reconstruye abstract desde el inverted_index de OpenAlex.
    """
    if not inv_idx:
        return ""
    # La longitud es 1 + max(pos) en todos los términos
    max_pos = max(max(positions) for positions in inv_idx.values()) # se queda el máximo
    length = max_pos + 1

    out = [""] * length # toda la longitud con strings vacíos
    for word, positions in inv_idx.items():
        for p in positions: # recorrer cada posición en la que se encuentra la palabra
            out[p] = word
    return " ".join(out).strip()

def search_openalex(
    query: str,
    max_results: int = 50,
    per_page: int = 25,
    from_year: Optional[int] = None,
    to_year: Optional[int] = None,
    lang_filter: Optional[str] = None,  # ejemplo: "es|en"
    sleep_s: float = 0.5,
) -> List[Reference]:
    """
    Busca en OpenAlex trabajos relevantes. Usa `search=`.
    Reconstruye abstracts y devuelve referencias normalizadas.
    """
    results: List[Reference] = []
    page = 1
    fetched = 0

    while fetched < max_results:
        params = {
            "search": query,
            # "filter": "title_and_abstract.search:" + query, # no carga titulo y abstract
            "per_page": min(per_page, max_results - fetched),
            "page": page,
            "sort": "relevance_score:desc",
        }
        
        filters = []
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        if to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")
        if lang_filter:
            filters.append(f"language.search:{lang_filter}")
        if filters:
            params["filter"] = ",".join(filters)

        print("params", params)

        resp = requests.get(f"{OPENALEX_BASE}/works", params=params, timeout=60)
        if resp.status_code != 200:
            # backoff simple
            time.sleep(2)
            resp = requests.get(f"{OPENALEX_BASE}/works", params=params, timeout=60)
            if resp.status_code != 200:
                break

        data = resp.json()
        print("data", data)

        works = data.get("results", [])
        if not works:
            break

        for w in works:
            wid = w.get("id") or ""
            title = normalize_text(w.get("title", ""))
            inv_idx = w.get("abstract_inverted_index")
            abstract = abstract_from_openalex_inv_idx(inv_idx) if inv_idx else normalize_text(w.get("abstract", ""))
            
            url = None
            # priorizar doi o url principal
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            if doi:
                url = f"https://doi.org/{doi}"
            else:
                # Manejo seguro de primary_location anidado
                primary_location = w.get("primary_location") or {}
                source = primary_location.get("source") or {}
                
                # Intentar obtener URL de diferentes fuentes
                url = (primary_location.get("landing_page_url") or 
                       primary_location.get("pdf_url") or 
                       source.get("host_organization_name"))
            date = w.get("publication_date") or str(w.get("publication_year") or "")

            authorships = w.get("authorships", [])
            authors_info = []
            for a in authorships:
                if a.get("author"):
                    author_data = a.get("author", {})
                    name = author_data.get("display_name", "")
                    orcid = author_data.get("orcid", "")
                    
                    # Obtener afiliación de forma segura
                    affiliations = a.get("affiliations", [])
                    affiliation = ""
                    if affiliations and len(affiliations) > 0:
                        affiliation = affiliations[0].get("raw_affiliation_string", "")
                    
                    # Crear string del autor con la información disponible
                    author_str = name
                    if orcid:
                        author_str += f" (ORCID: {orcid})"
                    if affiliation:
                        author_str += f" - {affiliation}"
                    
                    authors_info.append(author_str)

            authors = ", ".join(authors_info)

            keywords_list = w.get("keywords", [])
            keywords = ", ".join([k.get("display_name", "") for k in keywords_list])

            concepts_list = w.get("concepts", [])
            concept_names = ", ".join([c.get("display_name", "") for c in concepts_list])
            
            results.append(Reference(
                source="openalex",
                id=wid,
                title=title,
                abstract=abstract,
                doi=doi,
                url=url,
                date=date,
                authors_or_assignees=authors or None,
                keywords=keywords or None,
                concept_names=concept_names or None
            ))

            fetched += 1
            if fetched >= max_results:
                break

        page += 1
        time.sleep(sleep_s)

    return results

def search_patentsview(
    query_text: tuple[str, str],
    max_results: int = 100,
    per_page: int = 100,
    sleep_s: float = 0.8
) -> List[Reference]:
    """
    Busca patentes en PatentsView por título y abstract usando búsqueda de texto (_text_any).
    Devuelve referencias con CPC cuando estén disponibles.
    """
    fields = [
        "patent_id", "patent_title", "patent_date", "patent_abstract", "assignees"
    ]
    q = {
        "_or": [
            {"_text_any": {"patent_title": query_text[0]}},
            {"_text_any": {"patent_abstract": query_text[1]}}
        ]
    }
    results: List[Reference] = []
    page = 1
    fetched = 0

    while fetched < max_results:
        params = {
            "q": json.dumps(q),
            "f": json.dumps(fields),
            "o": json.dumps({
                "page": page,
                "per_page": min(per_page, max_results - fetched)
            })
        }

        print("URL", PATENTSVIEW_BASE)
        print("params", params)
        resp = requests.get(PATENTSVIEW_BASE, params=params, timeout=90, headers={"x-api-key": PATENTSVIEW_KEY})
        if resp.status_code != 200:
            time.sleep(2)
            resp = requests.get(PATENTSVIEW_BASE, params=params, timeout=90, headers={"x-api-key": PATENTSVIEW_KEY})
            if resp.status_code != 200:
                break

        print("resp", resp)

        data = resp.json()
        patents = data.get("patents", [])
        if not patents:
            break

        for p in patents:
            pn = p.get("patent_number") or p.get("patent_id") or ""
            title = normalize_text(p.get("patent_title", ""))
            abstract = normalize_text(p.get("patent_abstract", ""))
            date = p.get("patent_date")
            url = f"https://patents.google.com/patent/US{pn}" if pn else None

            # Procesar assignees
            assignees = p.get("assignees") or []
            assg_names = []
            for a in assignees:
                org = a.get("assignee_organization", "")
                first = a.get("assignee_first_name", "")
                last = a.get("assignee_last_name", "")
                if org:
                    assg_names.append(org)
                elif first or last:
                    assg_names.append(f"{first} {last}".strip())
            
            # Procesar inventors
            inventors = p.get("inventors") or []
            inventor_names = []
            for inv in inventors:
                first = inv.get("inventor_name_first", "")
                last = inv.get("inventor_name_last", "")
                if first or last:
                    inventor_names.append(f"{first} {last}".strip())
            
            # Combinar assignees e inventors
            all_names = []
            if assg_names:
                all_names.extend([f"Assignee: {name}" for name in assg_names])
            if inventor_names:
                all_names.extend([f"Inventor: {name}" for name in inventor_names])
            
            authors_or_assignees = ", ".join([x for x in all_names if x]) or None


            cpcs = p.get("cpcs") or []
            cpc_sections = list({c.get("cpc_section_id") for c in cpcs if c.get("cpc_section_id")})
            cpc_groups = list({c.get("cpc_group_id") for c in cpcs if c.get("cpc_group_id")})

            results.append(Reference(
                source="patentsview",
                id=pn,
                title=title,
                abstract=abstract,
                url=url,
                date=date,
                authors_or_assignees=authors_or_assignees,
                cpc_sections=cpc_sections,
                cpc_groups=cpc_groups
            ))
            fetched += 1
            if fetched >= max_results:
                break

        page += 1
        time.sleep(sleep_s)

    return results

