import numpy as np, torch, transformers
from torch import nn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from typing import *
from utils.classes import *

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # * for devolopment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Bert

from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()
bert.to(device)

class GetVec(nn.Module):

  def __init__(self, tokenizer, encoder) -> None:
    super().__init__()

    self.tokenizer = tokenizer
    self.encoder = encoder

  def forward(self, text: str | List[str], return_tokens: bool = True, update: bool = False) -> Tuple[List[str], torch.Tensor]:
    """Gets word embedding vectors

    Args:
      text: str - the text to embed
      return_tokens: bool - if using List[str] for text
      update: bool - doesn't really do much

    Returns
      tokens: List[str] - the tokens from the tokenizer
      embeddings: torch.Tensor - the embedding values for these tokens

    """

    if update: print("Starting Tokenization!")

    tokens = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(device)

    str_tokens = None
    if return_tokens:
      str_tokens = self.tokenizer.convert_ids_to_tokens(tokens.input_ids.squeeze(0).tolist())

    if update: print("Starting Encoding!")

    with torch.no_grad():
        bert_output = self.encoder(**tokens)

        x = bert_output.last_hidden_state

    return str_tokens, x


embedder = GetVec(tokenizer, bert)

def get_embedding(text: str) -> torch.Tensor:
  """Get embedding for single sentence | word"""

  tokens, embed = embedder(text)
  return embed.squeeze(0)[1:-1].sum(dim=0)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/search')
def search(s: Search) -> SearchResult:
   
    embeddings = torch.hstack(s.embeddings)

    search_embed = get_embedding(s.query)

    search_results = (embeddings @ search_embed) / embeddings.norm(dim=1)

    _, top_ids = search_results.topk(5)

    return SearchResult(
        results = [s.ids[i] for i in top_ids.tolist()]
    )

@app.post("/embed")
def embed(e: EmbedRequest) -> EmbeddingResult:

    embeds = get_embedding(e.post)
   
    return EmbeddingResult(
        result = embeds.tolist()
    )