# BioLitGraph

BioLitGraph (Biological Literature Graph) is an tool designed to accelerate biomedical research by automatically extracting and standardizing biological entities (such as genes, proteins, diseases, drugs, and cell types) from large-scale biomedical literature, constructing a high-quality knowledge graph based on their co-occurrence relationships, and enabling simple path-based reasoning.

### Install BioLitGraph


```python
# Check if cuda is available
python -c "import torch;print(torch.cuda.is_available())"

# Install BioLitGraph
cd BioLitGraph
pip install -r requirements.txt
```


```python
curl -X POST https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/retrieve.cgi -H "Content-Type: application/x-www-form-urlencoded" -d "id=90C392ACD99A9B6769B7"
```

Then, you need to download resources (e.g., external modules or dictionaries) for running BioLitGraph. Note that you will need 15GB of free disk space. You can also download the resource file from google drive.


```python
# (For Linux Users) install CRF 
cd GNormPlusJava
tar -zxvf CRF++-0.58.tar.gz
mv CRF++-0.58 CRF
cd CRF
./configure --prefix="$HOME"
make
make install
cd ../..

# (For Windows Users) install CRF 
cd resources/GNormPlusJava
unzip -zxvf CRF++-0.58.zip
mv CRF++-0.58 CRF
cd ../..
```

### Running BioLitGraph

The minimum memory requirement for running BioLitGraph on GPU is 4GB of RAM & 5.05GB of GPU. The following command runs BioLitGraph.

#### Step 1 Input Options

Users can input either a **PMID** or their own **custom text**.

- **If a PMID is provided**:  
  The system will retrieve the corresponding title and abstract from the PubMed database using that PMID, and then convert them into PubTator format.

- **If custom text is provided**:  
  The user must supply the text in PubTator format themselves and place it in the `./case/{case_name}/pubtator` directory. The filename must strictly follow the pattern `{case_name}{num}.PubTator` (where `{case_name}` is the case name and `{num}` is a sequential number).


```python
#Search query as Input
python lit_download.py -c test -t
#PubMed ID (PMID) as Input
python lit_download.py -c test -p
```

#### **Step 2 Running named entity recognition**

This stage performs comprehensive named entity recognition (NER) on the input text (in PubTator format, either retrieved from PubMed or provided by the user). The process is divided into three complementary components, each leveraging specialized models to detect distinct biomedical entity types.

**Multi-Semantic Model Recognition**  
   A advanced multi-semantic deep learning model is employed to identify a broad range of biomedical entities with rich contextual understanding. The model is capable of recognizing the following entity types:  
   - **Diseases** (e.g., cancer, diabetes, Alzheimer's disease)  
   - **Compounds** (e.g., drugs, chemicals, small molecules, metabolites)  
   - **Cell Types** (e.g., T cells, neurons, hepatocytes)  
   - **Gene Ontology Terms** (biological processes, molecular functions, cellular components)  
   - **Body Parts** (e.g., liver, heart, brain)  
   - **Diagnostic Procedures** (e.g., MRI, biopsy, blood test)  
   - **Treatment Methods** (e.g., chemotherapy, surgery, radiotherapy)  
   - **Food** (e.g., Mediterranean diet, ketogenic diet)  
   - **Geographical Locations** (e.g., countries, regions, or places relevant to epidemiology)  

   This model excels at capturing complex semantic relationships and handling ambiguous or overlapping mentions in biomedical literature.


```python
cd multi_ner
python ner_server.py -c Medmention-diagnose
cd ..
```

**GnormPlus**  
   The GnormPlus tool, a state-of-the-art rule-based and machine learning hybrid system, is applied to accurately identify:  
   - **Genes and Proteins**  
   - **Species** (e.g., Homo sapiens, Mus musculus)  
   - **Cell Lines** (e.g., HeLa, HEK293)  

   GnormPlus performs context-aware disambiguation and normalization, linking entities to standard identifiers (e.g., NCBI Gene, Taxonomy IDs) where possible.


```python
case='NLM-Gene' 
mkdir ./case/$case/gnormplus_tmp
mkdir ./case/$case/gnormplus
cd GNorm2
java -Xmx40G -Xms40G -jar GNormPlus.jar "../case/${case}/pubtator" "../case/${case}/gnormplus_tmp" setup.txt
#python post-processing_gnormplus.py -c "${case}"
cd ..
```


```python
case='NLM-Gene' 
cd GNorm2
java -Xmx40G -Xms40G -jar GNormPlus.jar "../case/${case}/gnormplus_tmp" "../case/${case}/gnormplus" setup.txt
cd ..
```

**tmVar3**  
   The tmVar3 tool is used to detect **genetic variants** mentioned in the text, including:  
   - Single nucleotide variants (SNVs)  
   - Insertions/deletions (indels)  
   - Other sequence variations at DNA, RNA, and protein levels  


```python
case='Medmention-diagnose' 
mkdir ./case/$case/tmvar3
cd tmVar3
java -Xmx5G -Xms5G -jar tmVar.jar ../case/$case/gnormplus_tmp ../case/$case/tmvar3 
cd ..
```

All recognized entities from the three components are integrated and annotated in the standard **PubTator format**, ensuring compatibility with downstream analysis tools. Conflicts between overlapping predictions are resolved through priority rules and confidence scoring.

**NER post-processing**


```python
python post_process_ner.py -c Medmention-treatment
```

#### **Step 3 Running named entity normalization**

This stage focuses on normalizing the entities recognized in the previous NER phase, mapping them to standardized identifiers from established biomedical ontologies and databases. Normalization ensures consistency, enables interoperability, and facilitates downstream tasks such as knowledge graph construction or comparative analysis.

The normalization process covers the following entity types with their respective target resources:

- **Diseases**  
  Normalized to **MeSH** (Medical Subject Headings) and/or **OMIM** (Online Mendelian Inheritance in Man) identifiers, providing standardized disease terminology and links to genetic disorders where applicable.

- **Compounds** (drugs, chemicals, small molecules, metabolites)  
  Mapped to canonical identifiers from databases such as **PubChem**, **ChEBI**, or **DrugBank** (specific resource may depend on model configuration).

- **Cell Types**  
  Normalized using **Cell Ontology (CL)** and/or **BRENDA Tissue Ontology (BTO)**, ensuring precise representation of cell populations and tissue-derived cells.

- **Gene Ontology Terms** (biological processes, molecular functions, cellular components)  
  Directly linked to **Gene Ontology (GO)** identifiers.

- **Body Parts**  
  Normalized to concepts in **UMLS** (Unified Medical Language System), leveraging its comprehensive anatomical terminology.

- **Diagnostic Procedures**  
  Mapped to corresponding **UMLS** concepts for standardized representation of tests, imaging, and diagnostic methods.

- **Treatment Methods**  
  Normalized to **UMLS** concepts, covering interventions such as surgery, radiotherapy, and therapeutic procedures.

- **Food**  
  Linked to relevant **UMLS** concepts describing dietary patterns or nutritional regimens.

- **Geographical Locations**  
  Normalized using **GeoNames** identifiers for countries, regions, cities, or other epidemiologically relevant locations.

- **Genes and Proteins** (from GnormPlus)  
  Normalized to **NCBI Gene** (Entrez Gene) identifiers.

- **Species** (from GnormPlus)  
  Mapped to **NCBI Taxonomy** IDs.

- **Cell Lines** (from GnormPlus)  
  Normalized to **RRID** (Research Resource Identifier) where available, or standard cell line database entries.

- **Mutations** (from tmVar 3)  
  Already standardized to **HGVS** nomenclature with links to databases such as dbSNP or ClinVar (no additional ontology mapping required).

Normalization is performed using a combination of dictionary lookup, contextual disambiguation, and confidence scoring. Overlapping or conflicting annotations are resolved based on priority rules and source reliability. The final output augments the original PubTator file with standardized identifiers in the entity annotation lines, maintaining full compatibility with PubTator-centric tools and pipelines.


```python
python normalizer.py -c Medmention-treatment
```

#### Step 3 Path reasoning


```python
cd pathfinder
python co-occurrence.py -c HPV
python pathfinder.py -c HPV -s species:10566 -t disease:MESH:D002583,OMIM:603956 
```
