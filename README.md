# AutoStances-AllSides

Command Line interface to generate an Argument Diagram for a Topic of Interest. This tool employs the use of ChatGPT with Langchain + ChromaDB to query Reddit data and Congressional Hearing testimony as context for
generating the results.

Data is supplied internally, please contact Author for access.

1. First run load_data.py to upload data to local vector store.
2. Run main.py to generate output. Ensure preprocessing.py is also stored in the same directory. OpenAI key must be provided.


## Contextual Compression on Congressional Hearings & Similarity Search Retrieval on Reddit Data Example Output:
![output_v1](https://github.com/Armaniii/AutoStances-AllSides/assets/25016724/0b65afd4-1636-42e5-befb-28e058ad125a)



## Contextual Compression on Congressional Hearings & Multi-Query Retrieval on Reddit Data Example Output:
![output_v3_contextcompression_multiquery](https://github.com/Armaniii/AutoStances-AllSides/assets/25016724/98ae07f5-255a-4141-8677-b829d2c06935)
