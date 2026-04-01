{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "37a6fc42-8db6-4013-8086-21201f6dfdd6",
   "metadata": {},
   "source": [
    "# BioLitGraph"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "64eb70fc-4828-485b-ab0f-6073f0cafcf8",
   "metadata": {},
   "source": [
    "BioLitGraph (Biological Literature Graph) is an tool designed to accelerate biomedical research by automatically extracting and standardizing biological entities (such as genes, proteins, diseases, drugs, and cell types) from large-scale biomedical literature, constructing a high-quality knowledge graph based on their co-occurrence relationships, and enabling simple path-based reasoning."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2b1cc89-e730-4719-9b5d-e0ebf4874917",
   "metadata": {},
   "source": [
    "### Install BioLitGraph"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36c4e367-ab09-464c-8500-c861286893fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check if cuda is available\n",
    "python -c \"import torch;print(torch.cuda.is_available())\"\n",
    "\n",
    "# Install BioLitGraph\n",
    "cd BioLitGraph\n",
    "pip install -r requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e027351c-4d62-40a1-a1f6-bf13ffcf745b",
   "metadata": {},
   "outputs": [],
   "source": [
    "curl -X POST https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/retrieve.cgi -H \"Content-Type: application/x-www-form-urlencoded\" -d \"id=90C392ACD99A9B6769B7\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7ac858f7-978b-4604-8ace-321d528dc55e",
   "metadata": {},
   "source": [
    "Then, you need to download resources (e.g., external modules or dictionaries) for running BioLitGraph. Note that you will need 15GB of free disk space. You can also download the resource file from google drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "641c685c-96a7-4d48-ad16-cf92a4e21efe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# (For Linux Users) install CRF \n",
    "cd GNormPlusJava\n",
    "tar -zxvf CRF++-0.58.tar.gz\n",
    "mv CRF++-0.58 CRF\n",
    "cd CRF\n",
    "./configure --prefix=\"$HOME\"\n",
    "make\n",
    "make install\n",
    "cd ../..\n",
    "\n",
    "# (For Windows Users) install CRF \n",
    "cd resources/GNormPlusJava\n",
    "unzip -zxvf CRF++-0.58.zip\n",
    "mv CRF++-0.58 CRF\n",
    "cd ../.."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "58dfdfbf-fad7-48ce-8a59-1cd939aa2ae0",
   "metadata": {},
   "source": [
    "### Running BioLitGraph"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2d302cef-3baa-48b2-90fc-5413d0757713",
   "metadata": {},
   "source": [
    "The minimum memory requirement for running BioLitGraph on GPU is 4GB of RAM & 5.05GB of GPU. The following command runs BioLitGraph."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4aafa7c9-ef2d-44a0-9005-cc1571c2c8eb",
   "metadata": {},
   "source": [
    "#### Step 1 Input Options"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ef5f555-82ea-42b3-8905-e07baa90ecfc",
   "metadata": {},
   "source": [
    "Users can input either a **PMID** or their own **custom text**.\n",
    "\n",
    "- **If a PMID is provided**:  \n",
    "  The system will retrieve the corresponding title and abstract from the PubMed database using that PMID, and then convert them into PubTator format.\n",
    "\n",
    "- **If custom text is provided**:  \n",
    "  The user must supply the text in PubTator format themselves and place it in the `./case/{case_name}/pubtator` directory. The filename must strictly follow the pattern `{case_name}{num}.PubTator` (where `{case_name}` is the case name and `{num}` is a sequential number)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e5d7329b-b5f0-411a-bcdb-576dea7d7e81",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Search query as Input\n",
    "python lit_download.py -c test -t\n",
    "#PubMed ID (PMID) as Input\n",
    "python lit_download.py -c test -p"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2e935223-314f-447a-a201-bb74603d1f61",
   "metadata": {},
   "source": [
    "#### **Step 2 Running named entity recognition**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86101bfe-04e7-4be2-806a-52e284f5d925",
   "metadata": {},
   "source": [
    "This stage performs comprehensive named entity recognition (NER) on the input text (in PubTator format, either retrieved from PubMed or provided by the user). The process is divided into three complementary components, each leveraging specialized models to detect distinct biomedical entity types.\n",
    "\n",
    "**Multi-Semantic Model Recognition**  \n",
    "   A advanced multi-semantic deep learning model is employed to identify a broad range of biomedical entities with rich contextual understanding. The model is capable of recognizing the following entity types:  \n",
    "   - **Diseases** (e.g., cancer, diabetes, Alzheimer's disease)  \n",
    "   - **Compounds** (e.g., drugs, chemicals, small molecules, metabolites)  \n",
    "   - **Cell Types** (e.g., T cells, neurons, hepatocytes)  \n",
    "   - **Gene Ontology Terms** (biological processes, molecular functions, cellular components)  \n",
    "   - **Body Parts** (e.g., liver, heart, brain)  \n",
    "   - **Diagnostic Procedures** (e.g., MRI, biopsy, blood test)  \n",
    "   - **Treatment Methods** (e.g., chemotherapy, surgery, radiotherapy)  \n",
    "   - **Food** (e.g., Mediterranean diet, ketogenic diet)  \n",
    "   - **Geographical Locations** (e.g., countries, regions, or places relevant to epidemiology)  \n",
    "\n",
    "   This model excels at capturing complex semantic relationships and handling ambiguous or overlapping mentions in biomedical literature."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "14e6b013-68c5-4090-b624-81df3549cf47",
   "metadata": {},
   "outputs": [],
   "source": [
    "cd multi_ner\n",
    "python ner_server.py -c Medmention-diagnose\n",
    "cd .."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "30fbea22-9896-462f-86f5-56dcd00604e6",
   "metadata": {},
   "source": [
    "**GnormPlus**  \n",
    "   The GnormPlus tool, a state-of-the-art rule-based and machine learning hybrid system, is applied to accurately identify:  \n",
    "   - **Genes and Proteins**  \n",
    "   - **Species** (e.g., Homo sapiens, Mus musculus)  \n",
    "   - **Cell Lines** (e.g., HeLa, HEK293)  \n",
    "\n",
    "   GnormPlus performs context-aware disambiguation and normalization, linking entities to standard identifiers (e.g., NCBI Gene, Taxonomy IDs) where possible."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62f8577d-a35f-4c45-a16c-5723426b440f",
   "metadata": {},
   "outputs": [],
   "source": [
    "case='NLM-Gene' \n",
    "mkdir ./case/$case/gnormplus_tmp\n",
    "mkdir ./case/$case/gnormplus\n",
    "cd GNorm2\n",
    "java -Xmx40G -Xms40G -jar GNormPlus.jar \"../case/${case}/pubtator\" \"../case/${case}/gnormplus_tmp\" setup.txt\n",
    "#python post-processing_gnormplus.py -c \"${case}\"\n",
    "cd .."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd00bf0a-9c74-49e2-9b92-af4c32e77460",
   "metadata": {},
   "outputs": [],
   "source": [
    "case='NLM-Gene' \n",
    "cd GNorm2\n",
    "java -Xmx40G -Xms40G -jar GNormPlus.jar \"../case/${case}/gnormplus_tmp\" \"../case/${case}/gnormplus\" setup.txt\n",
    "cd .."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "de073e89-6523-4724-875f-d14845071dc0",
   "metadata": {},
   "source": [
    "**tmVar3**  \n",
    "   The tmVar3 tool is used to detect **genetic variants** mentioned in the text, including:  \n",
    "   - Single nucleotide variants (SNVs)  \n",
    "   - Insertions/deletions (indels)  \n",
    "   - Other sequence variations at DNA, RNA, and protein levels  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68a6635b-dc2c-4bce-8d2b-d92fd37ed148",
   "metadata": {},
   "outputs": [],
   "source": [
    "case='Medmention-diagnose' \n",
    "mkdir ./case/$case/tmvar3\n",
    "cd tmVar3\n",
    "java -Xmx5G -Xms5G -jar tmVar.jar ../case/$case/gnormplus_tmp ../case/$case/tmvar3 \n",
    "cd .."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b4d432e-6859-4629-bbcb-00f1e670e81b",
   "metadata": {},
   "source": [
    "All recognized entities from the three components are integrated and annotated in the standard **PubTator format**, ensuring compatibility with downstream analysis tools. Conflicts between overlapping predictions are resolved through priority rules and confidence scoring."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28c7476a-265d-4c2f-bc9c-b4dce720f522",
   "metadata": {},
   "source": [
    "**NER post-processing**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0298317-e2b5-487b-a4d6-4a99c3577132",
   "metadata": {},
   "outputs": [],
   "source": [
    "python post_process_ner.py -c Medmention-treatment"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f951bdc0-dbcd-4427-81a9-f63b0d77b7b4",
   "metadata": {},
   "source": [
    "#### **Step 3 Running named entity normalization**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3aa3e235-68fc-4ae1-ac7e-4ac41de94667",
   "metadata": {},
   "source": [
    "This stage focuses on normalizing the entities recognized in the previous NER phase, mapping them to standardized identifiers from established biomedical ontologies and databases. Normalization ensures consistency, enables interoperability, and facilitates downstream tasks such as knowledge graph construction or comparative analysis.\n",
    "\n",
    "The normalization process covers the following entity types with their respective target resources:\n",
    "\n",
    "- **Diseases**  \n",
    "  Normalized to **MeSH** (Medical Subject Headings) and/or **OMIM** (Online Mendelian Inheritance in Man) identifiers, providing standardized disease terminology and links to genetic disorders where applicable.\n",
    "\n",
    "- **Compounds** (drugs, chemicals, small molecules, metabolites)  \n",
    "  Mapped to canonical identifiers from databases such as **PubChem**, **ChEBI**, or **DrugBank** (specific resource may depend on model configuration).\n",
    "\n",
    "- **Cell Types**  \n",
    "  Normalized using **Cell Ontology (CL)** and/or **BRENDA Tissue Ontology (BTO)**, ensuring precise representation of cell populations and tissue-derived cells.\n",
    "\n",
    "- **Gene Ontology Terms** (biological processes, molecular functions, cellular components)  \n",
    "  Directly linked to **Gene Ontology (GO)** identifiers.\n",
    "\n",
    "- **Body Parts**  \n",
    "  Normalized to concepts in **UMLS** (Unified Medical Language System), leveraging its comprehensive anatomical terminology.\n",
    "\n",
    "- **Diagnostic Procedures**  \n",
    "  Mapped to corresponding **UMLS** concepts for standardized representation of tests, imaging, and diagnostic methods.\n",
    "\n",
    "- **Treatment Methods**  \n",
    "  Normalized to **UMLS** concepts, covering interventions such as surgery, radiotherapy, and therapeutic procedures.\n",
    "\n",
    "- **Food**  \n",
    "  Linked to relevant **UMLS** concepts describing dietary patterns or nutritional regimens.\n",
    "\n",
    "- **Geographical Locations**  \n",
    "  Normalized using **GeoNames** identifiers for countries, regions, cities, or other epidemiologically relevant locations.\n",
    "\n",
    "- **Genes and Proteins** (from GnormPlus)  \n",
    "  Normalized to **NCBI Gene** (Entrez Gene) identifiers.\n",
    "\n",
    "- **Species** (from GnormPlus)  \n",
    "  Mapped to **NCBI Taxonomy** IDs.\n",
    "\n",
    "- **Cell Lines** (from GnormPlus)  \n",
    "  Normalized to **RRID** (Research Resource Identifier) where available, or standard cell line database entries.\n",
    "\n",
    "- **Mutations** (from tmVar 3)  \n",
    "  Already standardized to **HGVS** nomenclature with links to databases such as dbSNP or ClinVar (no additional ontology mapping required).\n",
    "\n",
    "Normalization is performed using a combination of dictionary lookup, contextual disambiguation, and confidence scoring. Overlapping or conflicting annotations are resolved based on priority rules and source reliability. The final output augments the original PubTator file with standardized identifiers in the entity annotation lines, maintaining full compatibility with PubTator-centric tools and pipelines."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5efe522-94ed-488e-b1aa-222a8eca0800",
   "metadata": {},
   "outputs": [],
   "source": [
    "python normalizer.py -c Medmention-treatment"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "655fc950-d84c-47c8-984d-cd139d63c418",
   "metadata": {},
   "source": [
    "#### Step 3 Path reasoning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "afba246f-b1ad-4a96-aa63-167496f7d599",
   "metadata": {},
   "outputs": [],
   "source": [
    "cd pathfinder\n",
    "python co-occurrence.py -c HPV\n",
    "python pathfinder.py -c HPV -s species:10566 -t disease:MESH:D002583,OMIM:603956 "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
