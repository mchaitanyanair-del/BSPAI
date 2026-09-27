"""
Conversational layer on top of the RAG report generator.
KEY PRINCIPLE: retrieval re-runs every turn based on the new message,
never just replays the first turn's context. See conversation notes on
why static context degrades a "chatbot" back into a plain LLM.
"""
import re
from app.structured_retrieval import StructuredStore
from app.vector_store import PaperVectorStore
from app.model_service import ModelService
from app.rag_report import generate_explanation, SYSTEM_PROMPT

GENE_PATTERN = re.compile(r"\b[A-Z0-9\-]{2,10}\b")


class ChatSession:
    def __init__(self, store: StructuredStore, vectors: PaperVectorStore, model: ModelService):
        self.store = store
        self.vectors = vectors
        self.model = model
        self.history: list[dict] = []
        self.last_pair: tuple[str, str] | None = None

    def _extract_pair(self, message: str) -> tuple[str, str] | None:
        """
        Naive gene-name extraction: looks for two ALL-CAPS-ish tokens.
        EDIT ME: swap for a call to your LLM asking it to extract gene
        names as JSON if you want this more robust — worth doing before
        a live demo since regex will miss lowercase input etc.
        """
        candidates = [
            tok for tok in GENE_PATTERN.findall(message.upper())
            if tok not in {"THE", "AND", "WHY", "HOW", "FOR"}
        ]
        if len(candidates) >= 2:
            return candidates[0], candidates[1]
        return None

    def ask(self, message: str) -> str:
        pair = self._extract_pair(message)

        if pair:
            self.last_pair = pair
            gene_a, gene_b = pair
            structured_ctx = self.store.lookup(gene_a, gene_b)
            model_out = self.model.predict(gene_a, gene_b, structured_ctx)
            chunks = self.vectors.retrieve(message)
            reply = generate_explanation(structured_ctx, model_out, chunks)
        elif self.last_pair:
            # follow-up question about the same pair, e.g. "why is safety low?"
            gene_a, gene_b = self.last_pair
            structured_ctx = self.store.lookup(gene_a, gene_b)
            model_out = self.model.predict(gene_a, gene_b, structured_ctx)
            chunks = self.vectors.retrieve(message)  # re-retrieve based on THIS question
            reply = generate_explanation(structured_ctx, model_out, chunks)
        else:
            # general question, no pair in context yet -> paper retrieval only
            chunks = self.vectors.retrieve(message)
            if not chunks:
                reply = ("I don't have paper context relevant to that question, and "
                          "no gene pair has been specified yet. Ask about a specific "
                          "pair (e.g. 'BCMA and CD3') or a general method question.")
            else:
                reply = "\n".join(chunks)  # or route through generate_explanation with empty structured_ctx

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": reply})
        return reply
