# ScoringRFP_Agent

POC backend matching the agreed architecture.

Flow:
Draft Response Excel -> Excel Reader -> LangGraph -> Product/RFP Knowledge ChromaDB -> Claude Haiku Analysis -> Configurable Python Rules -> Skill Gap -> Training Catalog ChromaDB -> Haiku Recommendation -> Final Excel Report.

POC scope:
- RFP input is Excel only.
- RFP Excel should contain Requirement and optionally Partner Response.
- Product/RFP knowledge is uploaded as TXT or MD for this POC.
- Training catalog is Excel with Training Name and Training Description.
- ChromaDB is used for product/RFP knowledge and training retrieval.
- LangGraph orchestrates the workflow.
- Claude Haiku 4.5 performs analysis and recommendation.
- MCP is intentionally not included in this POC.

Run locally:
1. python -m venv .venv
2. Activate the virtual environment.
3. pip install -r requirements.txt
4. Copy .env.example to .env and add ANTHROPIC_API_KEY.
5. uvicorn app:app --reload

Swagger: http://localhost:8000/docs

APIs:
- POST /api/v1/knowledge/product : upload TXT/MD product knowledge.
- POST /api/v1/knowledge/training : upload training Excel.
- POST /api/v1/score : score partner draft Excel.
- POST /api/v1/score/report : score and return output Excel.

Decision logic:
The LLM returns coverage, confidence, explanation, evidence and skill_gap.
Python rules convert those fields into YES/PARTIAL/NO, score and training_required.
Thresholds are configurable in .env, so the RFP requirements remain dynamic.

Recommended RFP columns:
Requirement
Partner Response

Training Excel columns:
Training Name
Training Description
