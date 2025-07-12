
###  Python setup

```bash
cd hupd-lightrag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python download_and_extract_hupd.py 2018
```

### Now create .env from .env.example

```bash
cp .env.example .env
```

### Then modify neo4j creds, ollama details

```bash
python filter_jsons.py 10000 hupd_extracted/2018 hupd_processed
```

### On ollama runpod server

```bash
ollama pull qwen2.5:14b-instruct
ollama run qwen2.5:14b-instruct
ollama pull bge-m3:latest
```

### Back on codespace

```bash
git clone https://github.com/HKUDS/LightRAG.git
```

<!-- This will create a folder LightRAG. Now we build it using the docker compose yaml file outside, which already has this folder mentioned -->

### With Docker, in root dir

```bash
docker compose build
```

<!-- Terminal 1 -->
<!-- DozerDB -->
<!-- docker compose up neo4j -->

```bash
docker compose up lightrag
```

### Without Docker

```bash
cd LightRAG
pip install -e ".[api]"
cp .env LightRAG/
lightrag-server

python lightrag_uploader.py ./hupd_processed
```

