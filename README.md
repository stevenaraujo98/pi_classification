# Propiedad intelectual
Recuperación de Información, Análisis y Extracción de Características, y Predicción Final

Se necesita: Título, resumen y objetivos del proyecto.

### Estructura
- data
- 1_preprocessing.ipynb


"Objeto del proyecto" puede ser tomado de "Productos esperados" y "Objetivos del proyecto" para determinar si es "Producto nuevo", "Procedimiento nuevo", "Mejora de producto", "Mejora de proceso" u "otro".
La sección de financiamiento externo, puede ser tomada de los "Datos generales del proyecto", "Financiamiento" (nombre de la empresa) y si existe convenio/contrato se puede brindar los documentos desde la sección de evidencias.
En la sección de "Unidades que participan" se puede brindar la información de "Instituciones colaboradoras".
En la sección de "Información de mercado" se podría responder el sector de la industria con la información "Obj. Socio Económico Frascati".


La idea seria buscar proyectos similares:
## Pasos
1. Módulo de Búsqueda y Novedad:  
Verificar si ninguna divulgación previa, en cualquier parte del mundo, describe exactamente lo que se reclama en la solicitud. Si la invención no ha sido divulgada públicamente en el estado de la técnica antes de la fecha de presentación, se considera nueva.  
Posibles paginas: Wipo, Google Patents, Scopus, Scival, Lens.  
Con que buscar: descripcion, nombre del proyecto, a nivel mundial. 
Una invención es nueva cuando no ha sido divulgada públicamente en ningún lugar del mundo antes de la fecha de presentación o de prioridad de la solicitud. No debe existir en publicaciones, patentes previas, conferencias, redes, o uso público. 
Ejemplos:
✅ Nuevo compuesto bioactivo aislado de un microorganismo del suelo ecuatoriano con efecto antifúngico sobre Sigatoka negra.
❌ Un fertilizante que ya está publicado en una revista científica, aunque se fabrique por primera vez en Ecuador, no es nuevo.


2. Módulo de Nivel Inventivo:  
Comparar la invención reclamada con el estado de la técnica más cercano y determinar si, para un experto en la materia, la solución propuesta resulta evidente o no a partir de los conocimientos previos. Si la solución no es obvia y resuelve un problema técnico de manera no anticipada por la tecnología existente, se puede considerar que existe un nivel inventivo.  
Con que buscar: metodo y resultado, que problema resuelve.  
Una invención tiene nivel inventivo si no resulta obvia para una persona con conocimientos normales en el campo técnico correspondiente. Debe implicar un avance o solución técnica no evidente a partir de lo que ya existe.
Ejemplos:
✅ Una formulación innovadora que combina un extracto vegetal y un probiótico de manera sinérgica para controlar enfermedades en camarones, mostrando un mecanismo técnico inesperado.
❌ Cambiar el color, tamaño o una proporción menor de ingredientes en una fórmula conocida no implica nivel inventivo.


3. Módulo de Aplicación Industrial:  
Verificar si el desarrollo puede ser fabricado o utilizado en algún tipo de industria o actividad práctica. Si la invención es susceptible de ser producida o utilizada en cualquier sector económico, cumple con este requisito. Describa cómo el desarrollo puede ser aplicado en la industria o en centros de investigación.
Con que buscar: que se pueda aplicar y no debe estar publicado.  
La invención debe ser susceptible de producción o utilización en cualquier tipo de industria , incluidas las agrícolas, pesqueras, alimentarias o biotecnológicas. Debe poder fabricarse o aplicarse de forma práctica.
Ejemplos:
✅ Un bioinsumo que puede producirse mediante fermentación y aplicarse en plantaciones de banano para reducir enfermedades fúngicas.
❌ Una teoría científica o un descubrimiento natural sin aplicación práctica no es patentable.



### Paginas donde buscan
#### Documentos cientificos 
- Web of Science (WoS) 
- `Scopus (Elsevier)` 
- PubMed Central (PMC) 
- Redalyc (revistas científicas latinoamericanas de acceso libre) 
- SciELO (producción científica iberoamericana) 
- `SpringerLink` 
- ProQuest 
- Google Scholar https://scholar.google.com/?hl=es 
- Dialnet https://dialnet.unirioja.es/ 
- Eric https://eric.ed.gov/ 

##### Se posee
- [OpenAlex](https://docs.openalex.org/api-entities/works/get-a-single-work):
    - Microsoft Academic Graph (MAG): Cuando Microsoft descontinuó este proyecto en 2021, OpenAlex utilizó su conjunto de datos final como una de sus - fuentes principales para continuar y construir sobre él.
    - Crossref: Una organización de registro de DOI (identificador de objeto digital) que proporciona datos sobre millones de trabajos académicos.
    - ORCID: Permite la identificación única de autores de investigaciones.
    - DataCite: Una organización que proporciona identificadores para conjuntos de datos de investigación y otros recursos.
    - DOAJ (Directory of Open Access Journals): Un directorio que indexa revistas académicas de acceso abierto.
    - Unpaywall: Una base de datos que rastrea las versiones de acceso abierto de los artículos.
    - PubMed y PubMed Central: Bases de datos de literatura biomédica y de ciencias de la vida.
    - Repositorios institucionales y temáticos: Reúne datos de repositorios como arXiv, Zenodo y muchos otros.
    - ISSN International Centre: Proporciona identificadores para publicaciones seriadas como revistas.
    - Rastreo web: También utiliza el rastreo web para encontrar y recopilar información de páginas de revistas y otros sitios.
- [Scopus - Elsevier](https://dev.elsevier.com/api_docs.html)
- [SpringerLink](https://dev.springernature.com/docs/api-endpoints/open-access/?source=data-solutions)


#### Patentes 
- Lens.org  https://www.lens.org/lens/ (aqui si pide el check que no eres un robot) 
- Google Patents  https://patents.google.com/ 
- Inpi Angentina  https://www.argentina.gob.ar/inpi 
- Inapi Chile https://www.inapi.cl/ 
- Wipo Patentscope  https://www.wipo.int/es/web/patentscope 
- Espacenet https://worldwide.espacenet.com/ 
- Invenes  https://consultas2.oepm.es/InvenesWeb/faces/busquedaInternet.jsp 
- Global Dossier https://www.uspto.gov/patents/basics/international-protection/global-dossier-initiative 
- Oficina Española de Patentes y Marcas (OEPM)  https://www.oepm.es/es/ 

##### Se posee
- [Patentsview](https://search.patentsview.org/docs/docs/Search%20API/SearchAPIReference#api-query-language)

## Informacion adicional
CPC significa “Cooperative Patent Classification”. Es el sistema usado por oficinas de patentes (USPTO/EPO) para clasificar patentes por tecnología.

Estructura jerárquica (de mayor a menor):
- Sección: letra grande (A–H, Y). Ej.: G = Física (incluye computación).
- Clase: 3 caracteres. Ej.: G06 (Cómputo; procesamiento de datos).
- Subclase: 4 caracteres. Ej.: G06F (Sistemas/arquitecturas de computación).
- Grupo: ej. G06F 17/00.
- Subgrupo: ej. G06F 17/30.
  
  
cpc_sections: lista de secciones (A, B, C, …, Y) que aparecen en las patentes más similares. Se usa para:
- Dar una “vista macro” del campo tecnológico.
- Alimentar el módulo de “Aplicación industrial” (sugerencias por mayoría).

cpc_groups: lista de grupos CPC (p. ej., G06F, H04L, o IDs de grupo según PatentsView) detectados en esas patentes. Se usa para:
- Afinar la caracterización tecnológica (más granular).
- En el “Nivel inventivo”, medir diversidad de grupos entre las referencias cercanas (si muchas referencias fuertes pertenecen a  distintos grupos, puede sugerir combinaciones obvias).