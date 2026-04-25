# BiolitPath

BiolitPath (Biological Literature Path retrieval) is an tool designed to accelerate biomedical research by automatically extracting and standardizing biological entities (such as genes, diseases, chemicals, and cell types) from large-scale biomedical literature, constructing a knowledge graph based on their co-occurrence relationships, and enabling simple path-based retrieval.

### Install BiolitPath


```python
# Check if cuda is available
python -c "import torch;print(torch.cuda.is_available())"

# Install BioLitGraph
cd BiolitPath
pip install -r requirements.txt
```

Then, you need to download resources (e.g., external modules or dictionaries) for running BiolitPath. Note that you will need 18GB of free disk space.  
Place GNormPlus, resources, and tmVar3 under the `BiolitPath` directory. The compressed package can be downloaded from http://www.biomedinfo.cn/biomedinfo/download/BiolitPath.tar.gz.


```python
# (For Linux Users) install CRF 
cd GNormPlusJava
mv CRF++-0.58 CRF
cd CRF
./configure --prefix="$HOME"
make
make install
cd ../..

# (For Windows Users) install CRF 
cd GNormPlusJava
mv CRF++-0.58 CRF
cd ../..
```

### Running BiolitPath

In the following, we use a simple case named "**test**" as an example.

#### Step 1 Input Options

Users can input either **search query**, **PMIDs** or their own **custom text**.

- **If a Search query or PMIDs are provided**:  
  The system will retrieve the corresponding title and abstract from the PubMed database and convert them into PubTator format.

&emsp;(1)If the input is a search query. Save it to the file `./case/test/test.term.txt`. For example:

```text
intrahepatic cholangiocarcinoma[tiab] AND FGFR2[tiab] AND BICC1[tiab]  
```

&emsp;(2)If the input is a list of PMIDs. Save them to the file ./case/test/test.pmid.txt, with one PMID per line.  
&emsp;For example:

```text
38331087
32934021
35564224
```

- **If custom text is provided**:  
  The user must supply the text in PubTator format and place it in the `./case/{case_name}/pubtator` directory.  
  The filename must follow the pattern `{case_name}{num}.PubTator`, where `{case_name}` is the case name and `{num}` is a sequential number.  
  For example: `./case/test/pubtator/test1.PubTator`

```text
35564224|t|Intracellular...  
35564224|a|The study...  
  
20417042|t|Detection...  
20417042|a|Human papilloma...  
```

```python
#Search query as Input
python lit_download.py -c test -q
```


```python
#PubMed ID as Input
python lit_download.py -c test -p
```


```python
#Custom text as Input
python lit_download.py -c test -t
```

#### **Step 2 Running named entity recognition**

This stage performs comprehensive named entity recognition (NER) on the input text (in PubTator format, either retrieved from PubMed or provided by the user). The process is divided into three complementary components, each leveraging specialized models to detect distinct biomedical entity types.  
  
Download the NER model from https://huggingface.co/honglian/BiolitNER and save it to `./model/BiolitNER`.
  
**Multi-Semantic Model Recognition**  
   A advanced multi-semantic deep learning model is employed to identify a broad range of biomedical entities with rich contextual understanding. The model is capable of recognizing the following entity types:  
   - **Diseases** (e.g., cancer, diabetes, Alzheimer's disease)  
   - **Chemical** (e.g., drugs, chemicals, small molecules, metabolites)  
   - **Cell Types** (e.g., T cells, neurons, hepatocytes)  
   - **Gene Ontology Terms** (biological processes, molecular functions, cellular components)  
   - **Body Parts** (e.g., liver, heart, brain)  
   - **Diagnosis** (e.g., MRI, biopsy, blood test)  
   - **Treatment** (e.g., chemotherapy, surgery, radiotherapy)  
   - **Food** (e.g., Mediterranean diet, ketogenic diet)  
   - **Locations** (e.g., countries, regions, or places relevant to epidemiology)  



```python
python multi_ner.py -c test
```

**GnormPlus**  
   The GnormPlus tool, a state-of-the-art rule-based and machine learning hybrid system, is applied to accurately identify:  
   - **Genes/Proteins**  
   - **Species** (e.g., Homo sapiens, Mus musculus)  
   - **Cell Lines** (e.g., HeLa, HEK293)  

   GnormPlus performs context-aware disambiguation and normalization, linking entities to standard identifiers (e.g., NCBI Gene, Taxonomy IDs) where possible.


```python
case='test'
mkdir ./case/$case/gnormplus_tmp
cd GNormPlus
java -Xmx40G -Xms40G -jar GNormPlus.jar ../case/${case}/pubtator ../case/${case}/gnormplus_tmp setup.txt
python post_processing_gnormplus.py -c "${case}"
cd ..
```

**tmVar3**  
   The tmVar3 tool is used to detect **genetic variants** mentioned in the text


```python
case='test'
mkdir ./case/$case/tmvar3
cd tmVar3
java -Xmx5G -Xms5G -jar tmVar.jar ../case/$case/gnormplus ../case/$case/tmvar3 
cd ..
```

All recognized entities from the three components are integrated and annotated in the standard **PubTator format**, ensuring compatibility with downstream analysis tools. Conflicts between overlapping predictions are resolved through priority rules and confidence scoring.

**NER post-processing**


```python
cd multi_ner
python post_process_ner.py -c $case
cd ..
```

#### **Step 3 Running named entity normalization**

This stage focuses on normalizing the entities recognized in the previous NER phase, mapping them to standardized identifiers from established biomedical ontologies and databases. Overlapping or conflicting annotations are resolved based on priority rules and source reliability.

Download the NER model from https://huggingface.co/honglian/BiolitNEN and save it to `./model/BiolitNEN`.  

The normalization process covers the following entity types with their respective target resources:

| Type           | Ontology                     | Type        | Ontology        |
|:---------------|:------------------------------|:------------|:-----------------|
| Chemical       | MESH; CHEBI                  | Body part   | UBERON           |
| Disease        | MESH; OMIM                   | Treatment   | UMLS (T060)      |
| Cell type      | Cell ontology; Cell taxonomy | Diagnose    | UMLS (T061)      |
| Food           | UMLS (T168)                  | Gene        | NCBI Gene        |
| Gene ontology  | Gene ontology                | Species     | Taxonomy         |
| Location       | Geoname                      | Cell line   | RRID             |

Mutation normalization refers to representing mutations as uniformly formatted, canonical expressions rather than mapping them to entries in a specific database.


```python
python normalizer.py -c $case
```

#### Step 3 Path retrieval

- **Co-occurrence score calculation**:  
    Co-occurrence of entity pairs is assessed across the corpus using Fisher’s exact test. 
The adjusted p-value (adjP) is used to quantify statistical significance. 
Entity pairs with an $\text{adjP} < 0.05$ are considered to have a co-occurrence relationship, 
and $-\log(\text{adjP})$ is used as the co-occurrence score.

    Triples in the form of (entity1, relation, entity2) are generated. Nodes represent entities, edges denote co-occurrence relationships, and edge weights correspond to co-occurrence scores, reflecting the strength of associations between entities.


```python
cd pathfinder
python co-occurrence.py -c test
```

- **Path retrieval**:  
    A path retrieval approach is developed to identify high-confidence multi-hop association paths between entities. 

    The path retrieval module supports three flexible input modes for different application scenarios:

1. **Single-source query**:  
   Given a single source entity, the system automatically retrieves the top-*k* highest-weighted paths originating from that entity.

2. **Source–target query**:  
   When both a source entity and a target entity are specified, the system searches for the shortest association paths between them in the graph.  
   This mode is particularly useful for validating known or hypothesized indirect mechanisms of action.

3. **Type-guided path query**:  
   Given a predefined sequence of node types (e.g., *Disease → Gene → Chemical/Drug*), the system preferentially retrieves paths that conform to the specified semantic structure.  

Case Study: Path Retrieval Examples

We demonstrate the proposed KG-based path retrieval approach on three representative disease cases.

Case 1: Intrahepatic Cholangiocarcinoma


```python
python pathfinder.py -c ICC -s disease:MESH:D018281 -d --max_hop 2 -k 5
```

Case 2: Mechanistic Link Between HPV Infection and Cervical Cancer


```python
python pathfinder.py -c HPV -s species:10566 -t disease:MESH:D002583,OMIM:603956 -d --max_hop 2 -k 5
```

Case 3: Myelodysplastic Syndrome


```python
python pathfinder.py -c MDS -s disease:MESH:D009190,OMIM:614286 -d --max_hop 2 -k 5
```

GO–gene association pathways


```python
#cell proliferation
python pathfinder.py -c ICC -s go:GO:0008283 -d --max_hop 1 -k 20 -n go+gene
#cell migration
python pathfinder.py -c ICC -s go:GO:0016477 -d --max_hop 1 -k 20 -n go+gene
#epithelial to mesenchymal transition
python pathfinder.py -c ICC -s go:GO:0001837 -d --max_hop 1 -k 20 -n go+gene
```
