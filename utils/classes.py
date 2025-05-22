from typing import * 

from pydantic import BaseModel

class Search(BaseModel):

    query: str

    ids: List[int]
    embeddings: List[List[float]]

class SearchResult(BaseModel):

    # Top 5 ids
    results: List[int]

class EmbedRequest(BaseModel):

    post: str

class EmbeddingResult(BaseModel):

    result: List[float]