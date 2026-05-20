# A gentle intro to KNN — for the layman

K-Nearest Neighbours (KNN) is simply the idea of finding the *k* most similar items to a given query in a collection.

Think of it like asking: *"given this song I like, what are the 5 most similar songs in my library?"* The algorithm measures the "distance" between items (how different they are) and returns the closest ones.

The two key parameters are:

- **k** — how many neighbours to return (e.g. the 5 most similar)
- **distance metric** — how "similarity" is measured (e.g. Euclidean, Manhattan, Hamming)

Everything else — VP-Trees, SIMD, approximate search — is just engineering to make that search fast at scale.

## Main applications of KNN search

1. **Image retrieval** — finding visually similar images by searching nearest neighbours in an embedding space (e.g. face recognition, reverse image search).
2. **Recommendation systems** — suggesting similar items (products, songs, articles) by finding the closest user or item embeddings.
3. **Anomaly detection** — flagging data points whose nearest neighbours are unusually distant as potential outliers or fraud cases.
4. **Semantic search** — retrieving documents or passages whose dense vector representations are closest to a query embedding (e.g. RAG pipelines).
5. **Broad-phase collision detection** — quickly finding candidate object pairs that might be colliding by looking up the nearest neighbours of each object's bounding volume, before running the expensive narrow-phase test.
6. **Soft body / cloth simulation** — finding the nearest mesh vertices or particles to resolve contact constraints and self-collision.
7. **Particle systems (SPH, fluid sim)** — each particle needs to know its neighbours within a radius to compute pressure and density forces.
