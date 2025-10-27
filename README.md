# PRISM: Prompt-Refined In-Context System Modeling for Financial Retrieval

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.12.3](https://img.shields.io/badge/python-3.12.3-blue.svg)](https://www.python.org/downloads/release/python-31203/)

**3rd Place Solution** for the [ACM ICAIF 2025 Agentic Retrieval Grand Challenge](https://www.kaggle.com/competitions/acm-icaif-25-ai-agentic-retrieval-grand-challenge).<br/>
Team members: Chun Chet Ng and Jia Yu Lim.

## Overview

PRISM (**P**rompt-**R**efined **I**n-Context **S**ystem **M**odeling) addresses the challenge of financial document retrieval through a two-stage pipeline architecture. The framework integrates three core methodologies: (1) prompt engineering for task-specific optimization (Prompt-Refined), (2) in-context few-shot learning for semantic alignment (In-Context), and (3) multi-agent system modeling for collaborative reasoning (System Modeling). We evaluate both non-agentic and agentic variants on the document and chunk ranking tasks introduced in FinAgentBench.

## Installation

```bash
git clone https://github.com/yourusername/prism_finagentbench.git
cd prism_finagentbench
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```
**LLM Provider**: Configure Azure OpenAI endpoint and key in the `.env` file.<br/>
This research was supported by Azure credits through the Microsoft for Startups program.

**Data Setup**: Download the dataset from the [Kaggle competition](https://www.kaggle.com/competitions/acm-icaif-25-ai-agentic-retrieval-grand-challenge/data) and place the JSONL files in the following structure:
```
prism_finagentbench/
└── data/
    ├── chunk_ranking_kaggle_dev.jsonl
    ├── chunk_ranking_kaggle_eval.jsonl
    ├── document_ranking_kaggle_dev.jsonl
    └── document_ranking_kaggle_eval.jsonl
```

## Usage

### Quick Start

Edit configuration in `main.py`:

```python
dry_run = False                  # Set True for testing
agentic_workflow = False         # False: non-agentic, True: agentic workflow
agentic_version = 4              # Agentic version (1-4), only used if agentic_workflow=True
agent_concurrency = 2            # Concurrent agents (agentic only)
use_doc_icl = True               # Enable ICL for document ranking
use_chunk_icl = True             # Enable ICL for chunk ranking
icl_n = 5                        # Number of ICL examples
```

Run the pipeline:
```bash
python main.py
```

### Workflow Comparison

| Workflow | Architecture | Speed | Cost | Characteristics |
|---------|-------------|-------|------|-----------------|
| **Non-Agentic** | Split-based processing | Fast | Low | Direct LLM prompting with split-based chunk processing |
| **Agentic V1** | Multi-role (4 agents) | Slow | High | Democratic consensus across diverse organizational perspectives |
| **Agentic V2** | Three-phase filtering | Medium | Medium-High | Progressive noise reduction through specialized phases |
| **Agentic V3** | Adaptive filtering | Medium-Fast | Medium | Confidence-based dynamic filtering with batch support |
| **Agentic V4** | Dual-analyst | Fast | Low-Medium | Balanced quantitative-qualitative evaluation |

### Agentic Architectures

**V1 - Multi-Role Democratic Consensus**
- Agents: CEO, Financial Analyst, Operations Manager, Risk Analyst
- Consensus: Arithmetic mean of all agent scores

**V2 - Three-Phase Specialized Processing**
- Phase 1: Noise removal (keeps 100-200 chunks)
- Phase 2: Candidate selection (keeps 50-100 chunks)
- Phase 3: Deep scoring with 4 specialized agents
- Consensus: Weighted ensemble (Relevance: 0.35, Context: 0.35, Evidence: 0.2, Diversity: -0.15)

**V3 - Two-Stage Adaptive Filtering**
- Stage 1: Confidence-based adaptive filtering (30-70% retention)
- Stage 2: Parallel deep scoring by 3 analytical agents
- Supports batch processing for cost optimization

**V4 - Dual-Analyst**
- Agents: Financial Analyst (quantitative) + Risk Analyst (qualitative)
- Consensus: Equal-weighted averaging

### Output Files

- **Submission**: `./submission_files/{run_id}_{timestamp}/{run_id}_{timestamp}_kaggle_submission.csv`
- **LLM Outputs**: `./llm_output/doc_output/{run_id}_{timestamp}` and `./llm_output/chunk_output/{run_id}_{timestamp}`
- **Checkpoints**: `./checkpoints/` (resume interrupted runs)

## Architecture

### Non-Agentic Workflow
Single-pass LLM evaluation with efficient split-based processing and lower LLM cost.
**Document Ranking**: Direct LLM prompting with In-Context Learning (ICL) examples → Top-5 document type selection

**Chunk Ranking**:
1. Split chunks into manageable subsets (default: 5 splits)
2. Rank chunks within each split using LLM with ICL
3. Extract top candidates from each split
4. Re-rank all candidates to produce final top-5

### Agentic Workflow
Multi-agent collaboration with LangGraph-orchestrated workflows and higher interpretability through agent reasoning.
**Document Ranking**: Question analysis → Parallel evaluation by specialized document agents (10-K, 10-Q, 8-K, DEF14A, Earnings) → Weighted consensus → Top-5 selection

**Chunk Ranking (Version-Dependent)**:
- **V1**: Parallel evaluation by 4 organizational role agents → Cross-agent discussion → Democratic consensus
- **V2**: Noise removal → Candidate selection → Deep scoring by 4 specialized agents → Weighted ensemble
- **V3**: Confidence-based quick filtering → Deep scoring by 3 analytical agents → Confidence-weighted aggregation
- **V4**: Parallel evaluation by 2 specialized analysts → Cross-analyst discussion → Equal-weighted consensus

## License

This project is licensed under the AGPL-3.0 License. See [LICENSE](LICENSE) for details.

## Contact

**Research Collaboration**: chunchet.ng [at] ailensgroup [dot] com, alexlow [at] ailensgroup [dot] com <br/>

© 2025 AI Lens Sdn. Bhd. (Company No. 1547854-U). Commercial rights reserved.
