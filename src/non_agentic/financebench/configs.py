PATH_DATASET_JSONL = "./data/financebench/financebench_open_source.jsonl"
PROCESSED_PATH_DATASET_JSONL = "./data/financebench/financebench_open_source_processed.jsonl"
PATH_DOCUMENT_INFO_JSONL = "./data/financebench/financebench_document_information.jsonl"
PATH_RESULTS = "./data/financebench/results"
PATH_BASELINE_RESULTS = "./data/financebench/baseline_results"
PATH_PDFS = "./data/financebench/pdfs"
VS_CHUNK_SIZE = 1024
VS_CHUNK_OVERLAP = 30
VS_DIR_VS = "./data/financebench/vectorstores"

DATASET_PORTION = "OPEN_SOURCE"
FINANCEBENCH_CONFIGS = [
    {
        "provider": "openai",
        "model_name": "gpt-4o",
        "eval_mode": "singleStore",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4o",
        "eval_mode": "sharedStore",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4o",
        "eval_mode": "inContext",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4o",
        "eval_mode": "inContext_reverse",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4o",
        "eval_mode": "oracle_reverse",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4o",
        "eval_mode": "sharedStore",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "eval_mode": "sharedStore",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "eval_mode": "singleStore",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "eval_mode": "inContext",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "eval_mode": "closedBook",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {"provider": "openai", "model_name": "gpt-4-turbo", "eval_mode": "oracle", "temp": 0.01, "max_tokens": 2048},
    {"provider": "openai", "model_name": "gpt-4", "eval_mode": "sharedStore", "temp": 0.01, "max_tokens": 2048},
    {"provider": "openai", "model_name": "gpt-4", "eval_mode": "singleStore", "temp": 0.01, "max_tokens": 2048},
    {"provider": "openai", "model_name": "gpt-4", "eval_mode": "closedBook", "temp": 0.01, "max_tokens": 2048},
    {"provider": "openai", "model_name": "gpt-4", "eval_mode": "oracle", "temp": 0.01, "max_tokens": 2048},
    {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "eval_mode": "oracle_reverse",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "openai",
        "model_name": "gpt-4-turbo",
        "eval_mode": "inContext_reverse",
        "temp": 0.01,
        "max_tokens": 2048,
    },
    {
        "provider": "",
        "model_name": "",
        "eval_mode": "singleStore",
        "temp": None,
        "max_tokens": None,
    },  # SPECIAL MODE --> RETRIEVAL ONLY MODE (SINGLE STORE)
    {
        "provider": "",
        "model_name": "",
        "eval_mode": "sharedStore",
        "temp": None,
        "max_tokens": None,
    },  # SPECIAL MODE --> RETRIEVAL ONLY MODE (SHARED STORE)
]
