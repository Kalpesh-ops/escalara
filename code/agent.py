import os
import sys
import json
import time
import argparse
from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# NEW: Universal OpenAI Client
from openai import OpenAI

# Load .env file
load_dotenv()

API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not API_KEY:
    print("FATAL: OPENROUTER_API_KEY environment variable not set in .env.")
    sys.exit(1)

# Initialize OpenAI client pointing to OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

# Primary and Fallback Models on OpenRouter
MODEL_NAME = "meta-llama/llama-3.3-70b-instruct" # Incredibly fast, great at JSON
FALLBACK_MODEL = "anthropic/claude-3-haiku"       # Cheap, reliable fallback


# 2. Pydantic Schemas for Structured Output
class TriageOutput(BaseModel):
    inferred_company: str = Field(description="The company this issue belongs to: HackerRank, Claude, Visa, or Unknown")
    request_type: str = Field(description="Must be one of: product_issue, feature_request, bug, invalid")
    search_queries: list[str] = Field(description="Max 2 optimized search queries to find the solution in a database.", max_length=2)

class FinalResponseOutput(BaseModel):
    status: str = Field(description="Must be either 'replied' or 'escalated'")
    product_area: str = Field(description="The general category of the issue (e.g., Billing, Login, API, Assessments)")
    response: str = Field(description="The final message to the user. Must be helpful and grounded only in the provided context.")
    justification: str = Field(description="Format exactly as: [Decision: X] | [Trigger: Y] | [Reasoning: Z]")

# 3. Core Agent Class
class OrchestrateAgent:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.load_corpus()

    def load_corpus(self):
        print(f"Loading corpus from {self.corpus_path}...")
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                self.corpus = json.load(f)
        except FileNotFoundError:
            print(f"FATAL: Corpus file not found at {self.corpus_path}. Run ingest.py first.")
            sys.exit(1)
            
        # Tokenize for BM25 (simple whitespace split)
        self.tokenized_corpus = [doc['text'].lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"Loaded {len(self.corpus)} chunks into BM25 index.")

    def retrieve(self, queries: list[str], top_k: int = 5) -> list[dict]:
        """Retrieve top chunks using BM25 across multiple sub-queries."""
        scored_chunks = {}
        for query in queries:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            
            for idx, score in enumerate(scores):
                if score > 0:
                    scored_chunks[idx] = scored_chunks.get(idx, 0) + score
        
        # Sort by accumulated score and take top_k
        top_indices = sorted(scored_chunks, key=scored_chunks.get, reverse=True)[:top_k]
        return [self.corpus[i] for i in top_indices]

    def call_llm(self, prompt: str, schema: BaseModel, system_instruction: str, max_retries: int = 2) -> dict:
        """Universal LLM wrapper using OpenRouter."""
        
        # Inject the Pydantic schema into the system prompt to guarantee JSON structure
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_with_schema = f"{system_instruction}\n\nYou MUST return ONLY raw, valid JSON matching this schema. Do not use markdown blocks:\n{schema_json}"
        
        models_to_try = [MODEL_NAME, FALLBACK_MODEL]
        
        for current_model in models_to_try:
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model=current_model,
                        messages=[
                            {"role": "system", "content": system_with_schema},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.0
                    )
                    
                    # Parse the string response into a dict
                    result_text = response.choices[0].message.content
                    return json.loads(result_text)
                    
                except Exception as e:
                    if '429' in str(e):
                        print(f"[{current_model}] Rate limit hit. Sleeping 10s... (Attempt {attempt + 1}/{max_retries})")
                        time.sleep(10)
                    else:
                        print(f"[{current_model}] Generation error: {e}")
                        break # Move to fallback model if it's a structural 500/503 error
            
            print(f"⚠️ Exhausted {current_model}. Switching to fallback...")
            
        raise Exception("FATAL: Max retries exceeded for all models.")

    def process_row(self, row) -> dict:
        # FIX: Ensure column extraction is completely case-insensitive and safe against Pandas NaNs
        row_dict = {str(k).lower(): v for k, v in row.items()}
        
        issue_raw = row_dict.get('issue', '')
        issue = "" if pd.isna(issue_raw) else str(issue_raw)
        
        subject_raw = row_dict.get('subject', '')
        subject = "" if pd.isna(subject_raw) else str(subject_raw)
        
        company_raw = row_dict.get('company', 'None')
        company = "None" if pd.isna(company_raw) else str(company_raw)
        
        # --- PASS 1: TRIAGE ---
        triage_prompt = f"Issue: {issue}\nSubject: {subject}\nCompany (if known): {company}\nAnalyze this ticket and generate the required JSON."
        triage_system = "You are a strict support triage router. Output valid JSON."
        
        try:
            triage_data = self.call_llm(triage_prompt, TriageOutput, triage_system)
        except Exception as e:
            print(f"Pass 1 Triage Failed: {e}")
            triage_data = {"inferred_company": company, "request_type": "invalid", "search_queries": [issue[:100]]}

        # Handle explicit injection/invalid strings
        if triage_data.get('request_type') == 'invalid':
            return {
                "status": "escalated",
                "product_area": "Security/Invalid",
                "response": "This request is out of scope or invalid.",
                "justification": "[Decision: escalated] | [Trigger: request_type=invalid] | [Reasoning: Triage marked input as invalid/malicious.]",
                "request_type": "invalid"
            }

        # --- RETRIEVAL ---
        chunks = self.retrieve(triage_data.get('search_queries', []))
        context_str = "\n\n".join([f"[ID: {c['id']}]\n{c['text']}" for c in chunks])

        # --- PASS 2: GENERATE & ESCALATE ---
        final_prompt = f"""
        User Issue: {issue}
        Subject: {subject}
        Company: {triage_data.get('inferred_company')}
        
        Retrieved Knowledge Context:
        {context_str}
        
        Instructions:
        1. Read the Context. If the context does not explicitly contain the answer, you MUST set status to 'escalated'.
        2. If the user asks about fraud, unauthorized charges, locked accounts, or legal action, you MUST set status to 'escalated'.
        3. If you can answer safely, set status to 'replied' and write a helpful response using ONLY the provided context. Cite the chunk ID in the justification.
        """
        final_system = "You are a support agent. You must ONLY use the provided chunks to answer. If the answer is not in the chunks, or if the issue involves fraud, unauthorized access, or legal threats, you MUST escalate."
        
        try:
            final_data = self.call_llm(final_prompt, FinalResponseOutput, final_system)
        except Exception as e:
            print(f"Pass 2 Generation Failed: {e}")
            final_data = {
                "status": "escalated",
                "product_area": "Unknown",
                "response": "Unable to process request securely.",
                "justification": f"[Decision: escalated] | [Trigger: parsing_error] | [Reasoning: LLM output failed to generate or parse.]"
            }
        
        final_data["request_type"] = triage_data.get("request_type", "product_issue")
        return final_data

def main():
    parser = argparse.ArgumentParser(description="Escalara Support Agent")
    # Updated default paths per your setup
    parser.add_argument("--input", type=str, default="support_tickets/sample_support_tickets.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="support_tickets/sample_output.csv", help="Path to save output CSV")
    args = parser.parse_args()

    # Ensure corpus is loaded from the directory where this script lives
    corpus_file = Path(__file__).parent / "corpus.json"
    
    agent = OrchestrateAgent(corpus_path=str(corpus_file))
    
    print(f"Loading input data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"FATAL: Input CSV not found at {args.input}")
        sys.exit(1)
        
    results = []
    print(f"Processing {len(df)} rows...")
    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}...")
        out_dict = agent.process_row(row)
        results.append(out_dict)
        time.sleep(4)
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Finished. Saved output to {args.output}")

if __name__ == "__main__":
    main()