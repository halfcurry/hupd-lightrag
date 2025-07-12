


python download_and_extract_hupd.py 2018

<!-- python filtering/filter_g06_patents_optimized.py hupd_extracted/2018 -->

python filter_jsons.py 10000 hupd_extracted/2018 hupd_processed

<!-- python lightrag_integration/create_env.py -->

# create .env from .env.example
cp .env.example .env
(modify neo4j creds, ollama details)

<!-- (On ollama runpod server) -->
ollama pull qwen2.5:14b-instruct
ollama run qwen2.5:14b-instruct
ollama pull bge-m3:latest

<!-- (on codespace) -->
git clone https://github.com/HKUDS/LightRAG.git

<!-- This will create a folder LightRAG. Now we build it using the docker compose yaml file outside, which already has this folder mentioned -->
docker compose build

<!-- Terminal 1 -->
<!-- DozerDB -->
<!-- docker compose up neo4j -->

docker compose up lightrag

python lightrag_uploader.py ./hupd_processed

