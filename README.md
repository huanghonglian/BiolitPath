# BiolitPath

BiolitPath (Biological Literature Path retrieval) supports biomedical literature analysis by automatically recognizing and normalizing biological entities, constructing a knowledge graph from their co-occurrence relationships, and retrieving multi-hop association paths. It helps researchers explore connections among diseases, genes, chemicals, and biological processes.

BiolitPath accepts PubMed search queries, PMID lists, and custom text in PubTator format. After preparing the input, run the complete analysis through a single entry point, `run_biolitpath.py`.

## 1. Features and workflow

```text
PubMed search query / PMID list / custom PubTator text
                          ↓
            Literature retrieval and text processing
                          ↓
               BiolitNER + GNormPlus + tmVar3
                          ↓
           Annotation integration and post-processing
                          ↓
                   Entity normalization
                          ↓
             Co-occurrence graph construction
                          ↓
                 Association path retrieval
```

### Named entity recognition

BiolitPath combines three complementary recognition modules:

- **BiolitNER**: a multi-semantic model that recognizes diseases, chemicals, cell types, Gene Ontology terms, body parts, diagnoses, treatments, foods, and locations.
- **GNormPlus**: recognizes genes/proteins, species, and cell lines, and provides normalization information where available.
- **tmVar3**: identifies genetic variants mentioned in the literature.

The annotations are integrated and post-processed, including abbreviation handling and overlap resolution, before normalization and graph construction.

### Named entity normalization

Dictionary and neural methods map recognized entities to identifiers in standard biomedical resources.

| Entity type | Resource | Entity type | Resource |
|---|---|---|---|
| Chemical | MESH; CHEBI | Body part | UBERON |
| Disease | MESH; OMIM | Treatment | UMLS (T060) |
| Cell type | Cell Ontology; Cell Taxonomy | Diagnosis | UMLS (T061) |
| Food | UMLS (T168) | Gene | NCBI Gene |
| Gene Ontology term | Gene Ontology | Species | Taxonomy |
| Location | Geoname | Cell line | RRID |

Variant normalization produces standardized representations of mutations.

### Co-occurrence graph construction

BiolitPath constructs a sentence-level co-occurrence graph with a selectable edge weight. Nodes represent normalized entities; edges represent statistical associations, not causal relationships. Entity type pairs remain restricted by `pathfinder/relation.txt`.

| Weight method | Edge weight | Filtering rule | Default threshold |
|---|---|---|---|
| `fisher` | Existing enrichment score: `-ln(raw p-value)` | `adjP < threshold` | `0.05` |
| `npmi` | Normalized pointwise mutual information | `NPMI > threshold` | `0` |
| `count` | Number of distinct sentences containing both entities | `count >= threshold` | `1` |

NPMI and count filtering do not additionally require Fisher significance. NPMI thresholds must be in `[0, 1]`; only positive NPMI edges are retained because path retrieval uses reciprocal edge weights. Count thresholds must be integers of at least `1`, and Fisher thresholds must be in `(0, 1]`. All computed statistics remain available in `co_occur_pvalue.txt`, including co-occurring pairs with Fisher p-value equal to one.

Choose the method and threshold when building the graph:

```bash
# Default Fisher graph: adjusted p-value < 0.05
python run_biolitpath.py -c test --input query --weight-method fisher

# NPMI graph: NPMI > 0.2
python run_biolitpath.py -c test --input query --weight-method npmi --threshold 0.2

# Count graph: at least 5 shared sentences
python run_biolitpath.py -c test --input query --weight-method count --threshold 5
```

To rebuild an existing case from normalized annotations, use `--from-step graph`. For example:

```bash
python run_biolitpath.py -c test --from-step graph --weight-method count --threshold 5
```

The options are also accepted by `co-occurrence.py` when run from `pathfinder/`. They apply only to graph construction; a paths-only run uses the existing graph. Rebuilding replaces the case's graph files and invalidates `graph.pkl`. The current path retrieval algorithm caps edge weights at `80` when computing path costs and scores, including count weights above `80`.

### Association path retrieval

Three query modes are supported:

1. **Single-source query**: retrieve association paths starting from a specified entity.
2. **Source-target query**: retrieve paths connecting a specified source and target.
3. **Type-guided query**: constrain paths using a sequence of entity types, such as `disease+gene+chemical`.

## 2. Installation and resources

### Python and Java environment

Activate the Python environment intended for BiolitPath and install the required dependencies. From the project root, install the multi-semantic NER dependencies:

```bash
pip install -r multi_ner/requirements.txt
```

This file is not a complete environment specification for the entire project. Literature processing, normalization, and graph analysis also require their dependencies, including Biopython, pandas, SciPy, NLTK, and NetworkX, together with the dependencies and tokenizer resources required by the models.

Check CUDA availability and Java:

```bash
python -c "import torch; print(torch.cuda.is_available())"
java -version
```

### External tools and dictionaries

Prepare GNormPlus, tmVar3, Ab3P, normalization dictionaries, and other external resources from the project resource package:

[Download the BiolitPath resource package](http://www.biomedinfo.cn/biomedinfo/download/BiolitPath.tar.gz)

The original installation instructions specify approximately 18 GB of free disk space for resources. Allow additional space for models, literature, and generated results.

Compile the GNormPlus CRF component as follows. Skip the rename command if the resource package already contains a directory named `CRF`.

```bash
cd GNormPlus
mv CRF++-0.58 CRF
cd CRF
./configure --prefix="$HOME"
make
make install
cd ../..
```

### Models

Download the models and place them in the following directories:

- [BiolitNER](https://huggingface.co/honglian/BiolitNER): `model/BiolitNER/`.
- [BiolitNEN](https://huggingface.co/honglian/BiolitNEN): `model/BiolitNEN/`.

Include the configuration, weights, tokenizer, and other required model files. The main resource layout is:

```text
BiolitPath/
├── run_biolitpath.py
├── GNormPlus/
│   ├── GNormPlus.jar
│   └── setup.txt
├── tmVar3/
│   └── tmVar.jar
├── Ab3P/
├── model/
│   ├── BiolitNER/
│   └── BiolitNEN/
├── resources/
└── case/
```

The pipeline does not install dependencies or download models automatically. All examples below are run from the project root using the activated Python environment.

## 3. Input options

The following examples use a case named `test`. Create the corresponding case directory and prepare one of the three input types.

### 3.1 PubMed search query

Save the query on the first line of `case/test/test.term.txt`:

```text
intrahepatic cholangiocarcinoma[tiab] AND FGFR2[tiab] AND BICC1[tiab]
```

Run:

```bash
python run_biolitpath.py -c test --input query
```

BiolitPath retrieves titles and abstracts from PubMed, converts them to PubTator format, and runs entity recognition, normalization, and graph construction. This command ends at `graph`; it does not run path retrieval.

### 3.2 PMID list

Save one PMID per line in `case/test/test.pmid.txt`:

```text
38331087
32934021
35564224
```

Run:

```bash
python run_biolitpath.py -c test --input pmid
```

BiolitPath retrieves the selected articles and runs through graph construction. Both query and PMID input modes require access to PubMed.

### 3.3 Custom text

Place custom text in PubTator format under `case/test/pubtator/`. Use filenames following `{case}{num}.PubTator`, for example, `test1.PubTator`:

```text
35564224|t|Example article title
35564224|a|Example abstract mentioning genes and diseases.

20417042|t|Another article title
20417042|a|Another abstract.

```

Separate articles with a blank line. Use the same article identifier for its title and abstract.

```bash
python run_biolitpath.py -c test --input text
```

In text mode, the initial `download` stage processes local text through `lit_download.py -t` rather than retrieving articles from PubMed. This processing may rewrite the input PubTator files.

## 4. Running BiolitPath

To run the complete workflow, including path retrieval, prepare the input and provide a source entity:

```bash
python run_biolitpath.py -c test --input query \
  --source disease:MESH:D018281 --max-hop 2 -k 5 -d
```

The pipeline executes all stages in order:

```text
download → ner → gnormplus → gnormplus-post → tmvar
         → ner-post → normalize → graph → paths
```

The source must be a full node identifier present in the graph generated for the case. The example illustrates identifier syntax; the node is not guaranteed to occur in every input corpus.

The ending stage is selected as follows:

- Without `--source` or an explicit `--to-step`, the pipeline ends at `graph`.
- With `--source` and no explicit `--to-step`, it ends at `paths`.
- An explicit `--to-step` takes precedence over these defaults.

## 5. Case studies and path retrieval

The following examples assume the case graphs have already been constructed. Use `--from-step paths` to query an existing graph without repeating literature processing.

### 5.1 Intrahepatic cholangiocarcinoma: single-source query

```bash
python run_biolitpath.py -c ICC --from-step paths \
  -s disease:MESH:D018281 --max-hop 2 -k 5 -d
```

### 5.2 HPV and cervical cancer: source-target query

```bash
python run_biolitpath.py -c HPV --from-step paths \
  -s species:10566 -t disease:MESH:D002583,OMIM:603956 \
  --max-hop 2 -k 5 -d
```

### 5.3 Myelodysplastic syndrome

```bash
python run_biolitpath.py -c MDS --from-step paths \
  -s disease:MESH:D009190,OMIM:614286 --max-hop 2 -k 5 -d
```

### 5.4 GO-gene association paths

```bash
# Cell proliferation
python run_biolitpath.py -c ICC --from-step paths \
  -s go:GO:0008283 -n go+gene --max-hop 1 -k 20 -d

# Cell migration
python run_biolitpath.py -c ICC --from-step paths \
  -s go:GO:0016477 -n go+gene --max-hop 1 -k 20 -d

# Epithelial-to-mesenchymal transition
python run_biolitpath.py -c ICC --from-step paths \
  -s go:GO:0001837 -n go+gene --max-hop 1 -k 20 -d
```

## 6. Output files

Results for each case are saved under `case/<case>/`:

| Directory or file | Contents |
|---|---|
| `pubtator/` | Retrieved or processed user-provided literature |
| `NERoutput/` | Multi-semantic entity recognition results |
| `gnormplus_tmp/`, `gnormplus/` | Raw and post-processed GNormPlus results |
| `tmvar3/` | Variant recognition results |
| `NERoutput_m/` | Integrated and post-processed entity annotations |
| `NENoutput/` | Normalized entity annotations |
| `pathfinder/co_occur_pvalue.txt` | Entity co-occurrence statistics and association scores |
| `pathfinder/graph.edgelist` | Graph edges and weights |
| `pathfinder/graph.nodelist` | Graph node information |
| `pathfinder/graph_config.json` | Weight method and threshold used for graph construction |
| `pathfinder/graph.pkl` | Cached NetworkX graph, including node and edge attributes |
| `pathfinder/found_paths_top<k>_<timestamp>.json` | Retrieved paths (default filename includes the requested path count and a timestamp with microseconds) |
| `logs/` | Logs for individual stages |

Path retrieval loads `graph.pkl` directly when it exists. On the first run, it loads `graph.edgelist` and `graph.nodelist` and saves the resulting graph to `graph.pkl`. Rebuilding the graph through `co-occurrence.py` removes the old cache. If you edit the text graph files manually, delete `graph.pkl` before querying to regenerate it.

Each path retrieval run uses a timestamped output filename by default. To choose a filename, pass `--output`; relative paths are resolved against the directory from which you launch the command. Parent directories are created when results are written. An explicitly selected file is overwritten if it already exists. If no paths are found, no result file is written. Repeated runs may still overwrite intermediate files.

```bash
python run_biolitpath.py -c ICC --from-step paths \
  -s disease:MESH:D018281 -k 5 --output results/ICC_paths.json
```

## 7. Parameters

| Parameter | Description | Default or requirement |
|---|---|---|
| `-h`, `--help` | Display help and exit | Optional |
| `-c`, `--case` | Case name | Required; starts with a letter, digit, or `_`; remaining characters may also include `.` and `-` |
| `--input` | Input mode: `query`, `pmid`, or `text` | Required when starting at `download` |
| `--weight-method` | Graph edge weight: `fisher`, `npmi`, or `count` | `fisher`; applies to graph construction |
| `--threshold` | Method-specific filtering threshold | Fisher: `0.05`; NPMI: `0`; count: `1` |
| `--from-step` | First stage to execute | `download` |
| `--to-step` | Last stage to execute | `paths` with a source; otherwise `graph` |
| `-s`, `--source` | Full source node identifier | Required when ending at `paths` |
| `-t`, `--target` | Full target node identifier | Optional; requires ending at `paths` |
| `-n`, `--node-types`, `--node_types` | Node type sequence, such as `go+gene` | Optional; requires ending at `paths` |
| `-k`, `--path-num` | Number of paths to retrieve | `10`; must be at least `1` |
| `--max-hop`, `--max_hop` | Maximum number of hops | `2` |
| `--min-hop`, `--min_hop` | Minimum number of hops | `1`; requires `1 ≤ min-hop ≤ max-hop` |
| `-d`, `--display` | Display path retrieval results in the console | Disabled |
| `-o`, `--output` | Path retrieval JSON output path | Timestamped filename in the case's `pathfinder/` directory; requires ending at `paths` |
| `--gnormplus-memory` | Initial and maximum Java heap for GNormPlus | `40G` |
| `--tmvar-memory` | Initial and maximum Java heap for tmVar3 | `5G` |
| `--no-cuda` | Pass `--no_cuda` to multi-semantic NER | Disabled; does not control normalization |
| `--dry-run` | Print stages, working directories, and commands | Does not execute stages, create logs, or check resources and input files |

Java heap sizes accept positive integers with an optional `K`, `M`, or `G` suffix, such as `16G` or `512M`. Each value sets both `-Xms` and `-Xmx`. Choose values appropriate for available memory and tool requirements:

```bash
python run_biolitpath.py -c test --input text \
  --gnormplus-memory 16G --tmvar-memory 5G
```

Preview the complete workflow:

```bash
python run_biolitpath.py -c test --input query \
  --source disease:MESH:D018281 --dry-run
```

A dry run still validates command-line arguments. Successful command preview does not confirm that resources or input files are ready.

## 8. Run control and troubleshooting

For a complete analysis, use the single command in Section 4. To reuse intermediate results or recover after a failure, select a continuous range with `--from-step` and `--to-step`. Both endpoints are included:

```text
download → ner → gnormplus → gnormplus-post → tmvar
         → ner-post → normalize → graph → paths
```

For example, after resolving a normalization error, rerun from normalization through graph construction:

```bash
python run_biolitpath.py -c test --from-step normalize
```

To rebuild only the graph:

```bash
python run_biolitpath.py -c test --from-step graph --to-step graph
```

Starting from an intermediate stage does not require `--input`, but its upstream results must already exist. The pipeline does not automatically run preceding stages or verify whether existing outputs correspond to the current input.

Before execution, the pipeline checks selected scripts, Java/JAR files, and some model resources. Before each stage, it checks that required input patterns match nonempty files. Missing input or a failed tool stops the workflow. These checks do not guarantee complete batches or valid file contents; individual tools may report further errors during execution.

Stage output and errors are displayed in the current terminal and saved to `case/<case>/logs/<timestamp>_<stage>.log` for troubleshooting.

Running `python run_biolitpath.py` without arguments exits because `--case` is required. A case name alone is also insufficient to start the full workflow: specify `--input` as well. To view all options:

```bash
python run_biolitpath.py --help
```
