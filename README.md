# Ontology‐Driven Graph‐Augmented Language Models for Interpretable Drug–Target Interaction Prediction
## Introduction
This repository contains the code for our proposed approach, which builds a semantic biomedical question-answering system by combining large language models (LLMs), ontology-guided extraction, and knowledge graphs. First, drug and protein keywords are used to retrieve relevant abstracts from PubMed. These abstracts are then processed by a large language model (LLM), guided by a prompt and a biomedical ontology, to extract entities and relationships. The extracted information is used to construct a knowledge graph (KG) in JSON format. Next, relevant subgraphs are retrieved based on embedding similarity and refined using the Prize-Collecting Steiner Tree algorithm to focus on the most informative context. These subgraphs are encoded both structurally (via GAT and MLP) and textually, then integrated as soft prompts into a frozen LLM. The KG is further enriched with ontology metadata from resources such as BioPortal and the Ontology Lookup Service, linking entities to standardized identifiers and preserving provenance. The result of this enrichment process is the Biomedical Generated Ontology (BioGenOnt), which serves as a structured and interoperable foundation for downstream reasoning. Finally, the LLM leverages the enriched subgraph and BioGenOnt context to generate accurate and interpretable predictions for drug–target interaction tasks.

## Requirements


pip install timeout_decorator
pip install langchain
pip install langgraph
pip install python-dotenv
pip install sentence_transformers
pip install torch_geometric
pip install sentencepiece
pip install protobuf
pip install langchain_community
pip install 'accelerate>=0.26.0'
pip install -U bitsandbytes
pip install bioservices
pip install chembl_webresource_client


## Repository Structure


This repository is organized into the following main components:

PubMed-Extraction/


This folder contains a Jupyter notebook responsible for extracting abstracts from PubMed. The retrieval is performed using keyword queries composed of drug and protein names. It enables the automatic collection of relevant biomedical literature for downstream knowledge graph construction. 

KG-Extraction/


This folder contains the codebase for knowledge graph (KG) extraction using a large language model (LLM) guided by a structured ontology and a custom prompt.

The main script is located in Extraction.py, which orchestrates the entity and relation extraction process.

Abstracts must first be segmented and stored in the file Abstracts_shunk.py.

The resulting triples are serialized into a JSON file, following the schema illustrated in example outputs found in the Results/ folder.

Prediction/


This folder contains the prediction module for Drug–Target Interaction (DTI).

The main script is Prediction.py, which loads the generated KG, retrieves relevant subgraphs, enriches the context with ontology metadata (BioGenOnt), and finally performs prediction using LLM-based reasoning.

Configuration Files (policy.txt) : 

The file policy.txt defines the paths and configuration settings used to run the DTI prediction pipeline. It specifies the location of all required input files such as knowledge graph elements, embeddings, datasets, and ontology. Here's a breakdown of what each path points to:

node_descriptions_file: JSON file containing textual descriptions of each node in the KG.

source_nodes_file: List of source node names involved in the KG edges.

target_nodes_file: List of target node names.

attribut_file: File containing edge attributes associated with each interaction.

drug_embedding_file: File with embedding vectors for drugs.

target_embedding_file: File with embedding vectors for protein targets.

missing_embedding_file: List of entities with missing embeddings.

input_file: Bipartite graph file with drug–target pairs to evaluate.

file2_path: CSV file listing all drug–target pairs used.

file_path: UniProt-based protein annotation file.

ontology_file: The OWL ontology file (custom_ontology-ro.owl) used for semantic reasoning.

validation_dataset: Ground-truth file for evaluating predictions (negative examples).

strategies_workflow: Specifies the prediction strategy (e.g., Simple).


### Installation Guide

To get started with our approach GNN-LLM-ONT, follow the steps below. The execution was tested on a uCloud server with Linux.

1. Clone the repository


git clone https://github.com/YourUsername/GNN-LLM-ONT.git
cd GNN-LLM-ONT


2. Install Python dependencies

Before running the code, make sure to install all required libraries listed in the Requirements section. Once these are installed, proceed with the main execution.

3. Run the prediction script

After setting up the environment and dependencies, you can directly launch the DTI prediction task using:


python3 Prediction/Prediction.py


⚠️ Make sure you have added your Groq API key, your Hugging Face access token, and obtained permission to use specific models (e.g., meta-llama/Llama-2-7b-chat-hf, BioMistral/BioMistral-7B). See the section Authentication & Model Access Setup for more details.
## Datasets

The Prediction/Dataset/ directory is organized into several subfolders containing the data required for drug–target interaction (DTI) prediction:

KG-LLM/

This folder contains the knowledge graph (KG) extracted from PubMed abstracts using an LLM guided by a biomedical ontology and a structured prompt. The KG is saved in four separate JSON files:

descriptions.json: Textual descriptions of each KG node.

source.json: Source entities for each edge in the KG.

target.json: Target entities for each edge.

attribut.json: Attributes or relationship types between source and target entities.

Ontology/

This folder includes the file:

custom_ontology-ro.owl: This is the BioGenOnt ontology used for semantic enrichment, including curated biomedical concepts and relationships.

## Authentication & Model Access Setup

To run the Knowledge Graph Extraction module, follow these steps to authenticate and gain access to the required LLMs:

1. ChatGroq API Key (for deepseek-r1-distill-llama-70b, mixtral-8x7b-32768, etc.)
To use models hosted on ChatGroq such as deepseek-coder, mixtral, and others:

Sign up or log in to https://console.groq.com

Navigate to your API settings and generate a ChatGroq API key

In the file groq_client.py, insert your API key in the following line:


client = Groq(api_key="your_api_key")


2. Hugging Face Token Access (for Meta & BioMistral models)
To access meta-llama/Llama-2-7b-chat-hf and BioMistral/BioMistral-7B:

Create a Hugging Face account at https://huggingface.co

Go to your settings → Access Tokens and generate a token

Important: You must request access to the following models:

meta-llama/Llama-2-7b-chat-hf

BioMistral/BioMistral-7B

Once access is granted, you must insert your Hugging Face token into the file Prediction.py by modifying the following code:


from huggingface_hub import login
login("your_token")
