export MILVUS_URI="http://localhost:19530"
# export OFFLINE_MODE=False
export CORS_ALLOW_ORIGIN="http://localhost:5173;http://localhost:8080"

export OLLAMA_BASE_URL=http://localhost:11434

export OPENAI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export OPENAI_API_KEY=AIzaSyAtg-6pi-4AsOil8vTU4z7VdTeap-AQl2k

export DATABASE_URL=postgresql://odoo:4c0K8sJGcxIercJDlmhs@localhost:5432/openwebui
export DATABASE_POOL_SIZE=10
export DATABASE_POOL_MAX_OVERFLOW=10
export DATABASE_POOL_TIMEOUT=30

PORT="${PORT:-8080}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload 