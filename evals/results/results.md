| Config | Chunker | Search | Rerank | Recall@5 | Recall@10 | MRR@10 | Mean latency (ms) |
|---|---|---|---|---|---|---|---|
| fixed+vector+base | fixed | vector | no | 0.667 | 0.889 | 0.378 | 23 |
| fixed+hybrid+base | fixed | hybrid | no | 0.889 | 0.944 | 0.671 | 20 |
| structure+vector+base | structure | vector | no | 0.444 | 0.667 | 0.344 | 18 |
| structure+hybrid+base | structure | hybrid | no | 0.944 | 0.944 | 0.633 | 32 |

**Not run:**
- fixed+vector+rerank: not run: reranker model unavailable (OSError)
- fixed+hybrid+rerank: not run: reranker model unavailable (OSError)
- structure+vector+rerank: not run: reranker model unavailable (OSError)
- structure+hybrid+rerank: not run: reranker model unavailable (OSError)
