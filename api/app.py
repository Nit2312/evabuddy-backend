import re
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from dotenv import load_dotenv
import traceback
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_astradb import AstraDBVectorStore

from langchain_groq import ChatGroq
from langchain_core.embeddings import Embeddings
from huggingface_hub import InferenceClient

try:
    from api.answer_evaluator import evaluate_answer
except ImportError:
    from answer_evaluator import evaluate_answer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# CORS: Vercel frontend + optional FRONTEND_URL env + local dev
_CORS_ORIGINS = [
    "https://evabuddy-frontend.vercel.app",
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
CORS(app, origins=[o for o in _CORS_ORIGINS if o], supports_credentials=True)

# Security: input limits
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "10000"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "5000"))
MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "50000"))


@app.after_request
def security_headers(response):
    """Add security headers to all responses."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Global variables for RAG system
rag_chain = None
retriever = None
vectorstore = None
embeddings = None
total_cases = 0
total_chunks = 0
system_initialized = False

# Eval set for Recall@k
_eval_relevance = []

# RAG config
RAG_FETCH_K = int(os.getenv("RAG_FETCH_K", "15"))
RAG_USE_TOP_K = int(os.getenv("RAG_USE_TOP_K", "8"))
RAG_RERANK_MAX = int(os.getenv("RAG_RERANK_MAX", "15"))
RAG_SKIP_RERANK = os.getenv("RAG_SKIP_RERANK", "").lower() in ("1", "true", "yes")
RETRIEVAL_K = RAG_USE_TOP_K

# Pattern: line that starts the actual answer
_ANSWER_START = re.compile(
    r"^\s*(\d+[.)]\s|\*\*\d+\.\s|No procedure for this|No documentation for this)",
    re.IGNORECASE,
)


def _format_docs(docs):
    """Format Document-like objects into context string."""
    formatted = []
    for doc in docs:
        meta = (getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})) or {}
        content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else "")
        doc_type = meta.get("type", "unknown")
        if doc_type == "case_record":
            case_id = meta.get("CaseID") or meta.get("case_id") or ""
            job_name = meta.get("Job_Name") or meta.get("job_name") or ""
            formatted.append(f"CaseID: {case_id}, Job_Name: {job_name}\n{content}")
        elif doc_type == "pdf_document":
            filename = meta.get("filename") or ""
            formatted.append(f"PDF: {filename}\n{content}")
        else:
            formatted.append(f"Document: {meta}\n{content}")
    return "\n\n".join(formatted)


def _strip_leading_reasoning(text: str) -> str:
    """Keep only the answer: drop any reasoning/preamble before first numbered item."""
    if not text or not text.strip():
        return text
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if _ANSWER_START.search(line.strip()):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def get_astra_config():
    """Get Astra DB configuration from environment variables."""
    api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
    token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    namespace = os.getenv("ASTRA_DB_NAMESPACE")
    collection_name = os.getenv("ASTRA_DB_COLLECTION", "elevator_cases")

    if not api_endpoint or not token or not namespace:
        raise ValueError(
            "Missing Astra DB configuration. Set ASTRA_DB_API_ENDPOINT, "
            "ASTRA_DB_APPLICATION_TOKEN, and ASTRA_DB_NAMESPACE."
        )

    return {
        "api_endpoint": api_endpoint,
        "token": token,
        "namespace": namespace,
        "collection_name": collection_name,
    }


# Astra DB caps exact count_documents at 1000; use this and fall back to estimate when exceeded
_ASTRA_COUNT_UPPER_BOUND = 1000


def _get_astra_document_counts(astra_config):
    """Query Astra DB for case and document chunk counts. Returns (total_cases, total_chunks)."""
    try:
        from astrapy import DataAPIClient

        client = DataAPIClient(astra_config["token"])
        try:
            db = client.get_database(
                astra_config["api_endpoint"], keyspace=astra_config["namespace"]
            )
        except TypeError:
            db = client.get_database(
                astra_config["api_endpoint"], namespace=astra_config["namespace"]
            )
        coll = db.get_collection(astra_config["collection_name"])

        try:
            total = coll.count_documents({}, upper_bound=_ASTRA_COUNT_UPPER_BOUND)
            cases = 0
            chunks = 0
            try:
                cases = coll.count_documents(
                    {"metadata.type": "case_record"},
                    upper_bound=_ASTRA_COUNT_UPPER_BOUND,
                )
                chunks = coll.count_documents(
                    {"metadata.type": "pdf_document"},
                    upper_bound=_ASTRA_COUNT_UPPER_BOUND,
                )
            except Exception:
                cases = total
                chunks = 0
            if cases == 0 and chunks == 0 and total > 0:
                cases = total
            return (cases, chunks)
        except Exception as count_err:
            if "1000" in str(count_err) or "exceeds" in str(count_err).lower():
                try:
                    estimate = coll.estimated_document_count()
                    return (0, estimate)
                except Exception:
                    pass
            raise count_err
    except Exception as e:
        print(f"[RAG] Could not get Astra document counts: {e}", file=sys.stderr)
        return (0, 0)


class RouterHuggingFaceEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError("HF_TOKEN is required for endpoint embeddings.")
        self._client = InferenceClient(model=model_name, token=api_key)

    def embed_documents(self, texts):
        result = self._client.feature_extraction(texts)
        if isinstance(result, list) and result and isinstance(result[0], float):
            return [result]
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def load_and_process_data():
    """Load and process data for RAG system"""
    global rag_chain, retriever, vectorstore, embeddings, total_cases, total_chunks

    try:
        model_name = "sentence-transformers/all-mpnet-base-v2"
        embeddings = RouterHuggingFaceEmbeddings(
            api_key=os.getenv("HF_TOKEN"),
            model_name=model_name,
        )

        astra_config = get_astra_config()
        vectorstore = AstraDBVectorStore(
            embedding=embeddings,
            api_endpoint=astra_config["api_endpoint"],
            token=astra_config["token"],
            namespace=astra_config["namespace"],
            collection_name=astra_config["collection_name"],
        )

        total_cases, total_chunks = _get_astra_document_counts(astra_config)

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RAG_FETCH_K}
        )

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="qwen/qwen3-32b",
            temperature=0,
            max_tokens=1536
        )

        template = """
You are an elevator technician assistant. Use ONLY the retrieved context below. Output ONLY the final answer—no reasoning, no "First I will...", no "Looking at the context...", no step-by-step analysis. Start directly with the answer (numbered list or bullets). Do not add any step or claim not explicitly in the context.

Rules:
- First line of your response must be the answer (e.g. "1. ..." or a direct statement). Never start with reasoning or preamble.
- Include steps/precautions from the context in order. Cite each part: "PDF: <filename>" or "CaseID: <id>, Job_Name: <name>".
- Match the question to the most relevant procedure(s) in the context even if wording differs.
- Do not say "No procedure" if the context contains a clearly related procedure. Only say "No procedure for this in the retrieved documentation" when nothing in the context is relevant.
- No generic safety filler unless it appears in the context. Use clear numbering or bullets.
- State only what is explicitly in the context. Every claim must be traceable to a cited source.

Retrieved context:
{context}

User question: {question}

Reply with ONLY the answer. Your first line must start with "1." or "No procedure" or "No documentation"—nothing else. No preamble.
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        rag_chain = prompt | llm | StrOutputParser()

        return True, f"AI Copilot initialized successfully! Loaded {total_cases} cases and {total_chunks} document chunks."

    except Exception as e:
        print("RAG initialization failed:\n" + traceback.format_exc())
        return False, f"Error loading data: {type(e).__name__}: {e}"


def initialize_rag_system():
    """Initialize the RAG system if not already done"""
    global system_initialized

    if not system_initialized:
        success, message = load_and_process_data()
        system_initialized = success
        return success, message

    return True, "System already initialized"


def get_retrieved_sources(query):
    """Return top-k sources: vector search then optional rerank."""
    if not retriever:
        return []
    docs_sem = retriever.invoke(query)
    docs_to_rerank = docs_sem[:RAG_RERANK_MAX]
    if RAG_SKIP_RERANK or not docs_to_rerank:
        return docs_to_rerank[:RAG_USE_TOP_K]
    try:
        from api.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker()
        return reranker.rerank(query, docs_to_rerank, top_k=RAG_USE_TOP_K)
    except Exception as e:
        print("Cross-encoder reranking failed:", e, file=sys.stderr)
        return docs_to_rerank[:RAG_USE_TOP_K]


def _source_doc_key(doc: dict) -> str:
    if doc.get("type") == "case_record":
        return f"case:{str(doc.get('case_id', ''))}:{(doc.get('job_name') or '').strip()}"
    if doc.get("type") == "pdf_document":
        return f"pdf:{(doc.get('filename') or '').strip()}"
    return f"other:{id(doc)}"


def _load_eval_relevance():
    """Load complex_eval_results.json if it exists."""
    global _eval_relevance
    path = os.path.join(os.path.dirname(__file__), "..", "complex_eval_results.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    for item in data:
        q = (item.get("question") or "").strip()
        if not q:
            continue
        sources = item.get("sources") or []
        keys = set()
        for s in sources:
            if s.get("type") == "case_record":
                keys.add(f"case:{str(s.get('case_id', ''))}:{(s.get('job_name') or '').strip()}")
            elif s.get("type") == "pdf_document":
                keys.add(f"pdf:{(s.get('filename') or '').strip()}")
        if keys:
            _eval_relevance.append({
                "question_norm": " ".join(q.lower().split()),
                "question_original": q,
                "relevant_keys": keys,
            })


def _recall_at_k(user_query: str, source_docs: list):
    """Compute Recall@k when user query matches an eval question."""
    import string
    STOPWORDS = set([
        'the', 'is', 'at', 'which', 'on', 'for', 'and', 'or', 'to', 'of', 'in',
        'a', 'an', 'as', 'by', 'with', 'from', 'that', 'this', 'are', 'was', 'be',
        'it', 'has', 'have', 'but', 'not', 'if', 'so', 'do', 'does', 'can', 'will',
        'would', 'should', 'must', 'may', 'were', 'been', 'such', 'than', 'then',
        'when', 'where', 'who', 'whom', 'whose', 'how', 'what', 'why', 'about',
        'into', 'up', 'down', 'out', 'over', 'under', 'again', 'more', 'most',
        'some', 'any', 'each', 'few', 'other', 'all', 'both', 'own', 'same', 'very', 'just', 'now'
    ])

    def normalize(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = [t for t in text.split() if t not in STOPWORDS]
        return set(tokens)

    if not source_docs or not _eval_relevance:
        return None

    query_tokens = normalize(user_query or "")
    best_match = None
    best_ratio = 0.0
    for item in _eval_relevance:
        eval_tokens = normalize(item["question_norm"])
        intersection = query_tokens & eval_tokens
        union = query_tokens | eval_tokens
        jaccard = len(intersection) / len(union) if union else 0
        partial = len(query_tokens) > 0 and query_tokens.issubset(eval_tokens)
        if partial:
            best_match = item
            break
        if jaccard > best_ratio:
            best_ratio = jaccard
            best_match = item
    if not best_match:
        return None

    relevant = best_match["relevant_keys"]
    if not relevant:
        return None

    retrieved = {_source_doc_key(d) for d in source_docs}
    hit = len(retrieved & relevant)
    return round(hit / len(relevant), 4)


def _count_cited_sources(response_text: str, source_docs: list) -> int:
    cited = 0
    for doc in source_docs:
        if doc.get("type") == "case_record":
            case_id = str(doc.get("case_id", ""))
            job_name = (doc.get("job_name") or "").strip()
            if case_id and case_id in response_text:
                cited += 1
                continue
            if job_name and job_name in response_text:
                cited += 1
        elif doc.get("type") == "pdf_document":
            filename = (doc.get("filename") or "").strip()
            if filename and filename in response_text:
                cited += 1
            elif "PDF:" in response_text and filename:
                short = filename.replace(".pdf", "")[:30]
                if short in response_text:
                    cited += 1
    return cited


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health():
    """Simple health check — always fast, no RAG init required."""
    return jsonify({'status': 'ok', 'initialized': system_initialized})


@app.route('/api/status', methods=['GET'])
def api_status():
    """Get RAG system status."""
    return jsonify({
        'initialized': system_initialized,
        'total_cases': total_cases,
        'total_chunks': total_chunks,
        'model': 'qwen/qwen3-32b',
        'search_type': 'Semantic Similarity + Cross-Encoder Reranking'
    })


@app.route('/api/initialize', methods=['POST'])
def api_initialize():
    """Trigger RAG system initialization."""
    try:
        success, message = initialize_rag_system()
        return jsonify({
            'success': success,
            'message': message,
            'total_cases': total_cases,
            'total_chunks': total_chunks
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f"Error initializing system: {str(e)}"}), 500


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    Handle chat requests.

    Expected JSON body:
      {
        "message": "string",          # required
        "user_id": "string",          # optional — for audit logs only (history stored in Firestore by frontend)
        "session_id": "string"        # optional — for audit logs only
      }
    """
    try:
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON body'}), 400
        user_input = (data.get('message') or '').strip()

        if not user_input:
            return jsonify({'error': 'Message is required'}), 400
        if len(user_input) > MAX_MESSAGE_LENGTH:
            return jsonify({'error': f'Message too long (max {MAX_MESSAGE_LENGTH} characters)'}), 400

        if not system_initialized:
            success, message = initialize_rag_system()
            if not success:
                return jsonify({'error': message}), 500

        try:
            sources = get_retrieved_sources(user_input)
            context = _format_docs(sources) if sources else ""
            response = rag_chain.invoke({"context": context, "question": user_input})
            response = _strip_leading_reasoning(response or "")
            if not (response and response.strip()):
                response = (
                    "No answer was generated from the retrieved documents. "
                    "Please try again or rephrase your question."
                )
        except Exception as e:
            if 'groqstatus.com' in str(e) or 'Service unavailable' in str(e):
                return jsonify({
                    'error': "The AI service is temporarily unavailable. Please try again later."
                }), 503
            return jsonify({'error': f"Error generating response: {str(e)}"}), 500

        meta_get = lambda d, k: (d.get(k) or d.get(k.lower()) or "")
        source_docs = []
        for doc in sources:
            meta = (getattr(doc, "metadata", None) or (doc.get("metadata") if isinstance(doc, dict) else {})) or {}
            content = getattr(doc, "page_content", None) or (doc.get("page_content") if isinstance(doc, dict) else "")
            doc_type = meta.get("type", "unknown")
            if doc_type == "case_record":
                source_docs.append({
                    "case_id": meta_get(meta, "CaseID") or meta_get(meta, "case_id"),
                    "job_name": meta_get(meta, "Job_Name") or meta_get(meta, "job_name"),
                    "content": content,
                    "type": "case_record",
                })
            elif doc_type == "pdf_document":
                source_docs.append({
                    "filename": meta_get(meta, "filename"),
                    "content": content,
                    "type": "pdf_document",
                })
            else:
                source_docs.append({"metadata": meta, "content": content, "type": "unknown"})

        cited = _count_cited_sources(response, source_docs)
        recall = _recall_at_k(user_input, source_docs)
        retrieval_metrics = {
            'k': RETRIEVAL_K,
            'retrieved': len(source_docs),
            'cited_in_answer': cited,
            'precision_at_k': round(cited / RETRIEVAL_K, 4) if RETRIEVAL_K else 0,
            'recall_at_k': recall,
        }

        return jsonify({
            'response': response,
            'sources': source_docs,
            'retrieval_metrics': retrieval_metrics,
        })

    except Exception as e:
        return jsonify({'error': f"Error generating response: {str(e)}"}), 500


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """Run the technical answer evaluation agent on a RAG response."""
    try:
        data = request.get_json() or {}
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON body'}), 400
        question = (data.get('question') or '').strip()
        response_text = (data.get('response') or '')[:MAX_RESPONSE_LENGTH]
        sources = data.get('sources') if isinstance(data.get('sources'), list) else []

        if not question or not response_text:
            return jsonify({'error': 'question and response are required'}), 400
        if len(question) > MAX_QUESTION_LENGTH:
            return jsonify({'error': f'question too long (max {MAX_QUESTION_LENGTH} characters)'}), 400

        evaluation = evaluate_answer(question, response_text, sources)
        return jsonify(evaluation)
    except Exception as e:
        return jsonify({'error': f"Evaluation failed: {str(e)}"}), 500


# ── Startup initialization ────────────────────────────────────────────────────
import threading

def _startup_init():
    """Run RAG initialization in a background thread so the server is
    ready to serve requests immediately while the heavy model/vector-store
    setup completes in the background."""
    print("[startup] Starting RAG initialization in background thread...")
    success, message = initialize_rag_system()
    if success:
        print(f"[startup] RAG system ready ✓  {message}")
    else:
        print(f"[startup] RAG initialization failed ✗  {message}", file=sys.stderr)

# Load eval data then kick off RAG init — both run before the first request
_load_eval_relevance()
_init_thread = threading.Thread(target=_startup_init, daemon=True, name="rag-init")
_init_thread.start()

if __name__ == '__main__':
    # In production use gunicorn and set FLASK_DEBUG=0
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(debug=debug, host='0.0.0.0', port=int(os.getenv("PORT", "5001")))
