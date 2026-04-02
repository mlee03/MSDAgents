# Chatbot test
Chatbot that will respond to inquiries about fre-cli based on the documentation
## Install with bash

1. Clone the repo and prepare directories:

```bash
git clone https://github.com/thomas-robinson/frechatbot.git
cd frechatbot
mkdir -p faiss_index
```

2. Clone the fre-cli repository (required for document loading):

```bash
if [ ! -d fre-cli ]; then
  git clone --recursive https://github.com/NOAA-GFDL/fre-cli.git
fi
```

3. Create the conda environment and install dependencies:

```bash
module load miniforge
conda deactivate
conda remove -n local-chatbot --all --yes
conda env create -f environment.yml
conda activate local-chatbot
pip install -r requirements-pip.txt
```

4. Set up Ollama (see [Ollama (gemma3) setup and usage](#ollama-gemma3-setup-and-usage) below)

5. Run the chatbot:

```bash
python chat.py
```

## Install with csh

1. Clone the repo and prepare directories:

```csh
git clone https://github.com/thomas-robinson/frechatbot.git
cd frechatbot
mkdir -p faiss_index
```

2. Clone the fre-cli repository (required for document loading):

```csh
if (! -d fre-cli) then
  git clone --recursive https://github.com/NOAA-GFDL/fre-cli.git
endif
```

3. Create the conda environment and install dependencies:

```csh
module load miniforge
conda deactivate
conda remove -n local-chatbot --all --yes
conda env create -f environment.yml
conda activate local-chatbot
pip install -r requirements-pip.txt
```

4. Set up Ollama (see [Ollama (gemma3) setup and usage](#ollama-gemma3-setup-and-usage) below)

5. Run the chatbot:

```csh
python chat.py
```

## Ollama (gemma3) setup and usage

This project uses Ollama's `gemma3` model for generation by default. The `ollama` Python client is included in `environment.yml`, but you must also run the Ollama daemon and pull `gemma3` locally.

### Setup

- Install the Ollama daemon following the official instructions: https://ollama.com/docs
- Pull the `gemma3` model (once Ollama is installed and running):

```bash
ollama pull gemma3
```

- Ensure the Ollama daemon is running:

```bash
ollama serve &
```

### Running the chatbot

After setting up the conda environment and installing pip requirements:

```bash
python chat.py
```

The chatbot will use `gemma3` for generation and display source documents by default.

If your Ollama daemon listens on a non-default URL, set the `OLLAMA_URL` environment variable:

```bash
export OLLAMA_URL="http://localhost:11434"
python chat.py
```

## FAISS vector store

The chatbot uses LangChain with HuggingFace sentence-transformers for embeddings and FAISS for the vector store.

- The first run of `python chat.py` will build the FAISS index under the `faiss_index/` directory.
- To force a rebuild of the index, remove the `faiss_index/` directory before running:

```bash
rm -rf faiss_index/
python chat.py
```


