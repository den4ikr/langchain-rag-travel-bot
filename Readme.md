# Budapest Travel Assistant — RAG Application

A Retrieval-Augmented Generation (RAG) application built with **Python, LangChain, and the OpenAI API** that answers questions about Budapest's attractions, public transportation, and practical travel tips using a custom knowledge base.

The project demonstrates how to design and implement an end-to-end RAG pipeline, combining semantic search, vector storage, context construction, and LLM-powered response generation.

## Features

- **Custom Knowledge Base** — Loads travel information from Markdown documents.
- **Document Processing** — Splits documents into smaller, overlapping chunks for retrieval.
- **Semantic Search** — Uses OpenAI embeddings to find relevant information based on the user's question.
- **Vector Storage** — Stores document chunks and their embeddings in ChromaDB.
- **LLM Orchestration** — Uses LangChain to connect retrieval, prompt construction, and response generation.
- **Context-Grounded Responses** — Provides retrieved context to GPT-4o-mini and instructs the model to acknowledge when the answer is not explicitly available.
- **Retrieval Transparency** — Displays retrieved document chunks and source metadata for inspection.

## Architecture

The application follows a Retrieval-Augmented Generation workflow:

```text
Markdown Documents
        |
        v
Document Loading & Chunking
        |
        v
OpenAI Embeddings
        |
        v
      ChromaDB
        |
        v
   User Question
        |
        v
  Semantic Retrieval
        |
        v
Retrieved Context + Question
        |
        v
  LangChain Prompt Pipeline
        |
        v
     GPT-4o-mini
        |
        v
   Generated Answer
```

### How It Works

1. **Document Ingestion** — Loads information about Budapest from Markdown files.
2. **Chunking** — Splits documents into smaller text segments with overlapping content.
3. **Embedding Generation** — Converts text chunks into vector representations using `text-embedding-3-small`.
4. **Vector Storage** — Stores the chunks and embeddings in a persistent ChromaDB collection.
5. **Semantic Retrieval** — Retrieves the two most relevant chunks for each user question.
6. **Context Construction** — Combines the retrieved content with the original question.
7. **Answer Generation** — Sends the constructed prompt to GPT-4o-mini through LangChain.
8. **Source Inspection** — Displays the retrieved content and associated source metadata.

## Tech Stack

| Technology               | Purpose                              |
| ------------------------ | ------------------------------------ |
| Python                   | Application logic                    |
| LangChain                | RAG pipeline and LLM orchestration   |
| OpenAI API               | Embeddings and response generation   |
| GPT-4o-mini              | Answer generation                    |
| `text-embedding-3-small` | Text embeddings                      |
| ChromaDB                 | Vector storage and similarity search |
| Markdown                 | Knowledge base documents             |
| python-dotenv            | Environment variable management      |

## Project Structure

```text
langchain_rag/
├── data/
│   ├── attractions.md
│   ├── transport.md
│   └── travel_tips.md
├── travel_db/
├── db_init.py
├── main.py
├── .env
└── README.md
```

- `data/` — Source documents for the knowledge base.
- `db_init.py` — Loads documents, splits them into chunks, generates embeddings, and initializes the ChromaDB collection.
- `main.py` — Initializes the retriever and LLM, runs the RAG pipeline, and provides a command-line interface.
- `travel_db/` — Persistent local vector database generated during indexing.
- `.env` — Stores the OpenAI API key.

## Getting Started

### Prerequisites

- Python 3.10 or later
- An OpenAI API key
- An active OpenAI API account with available API credits or billing

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd langchain_rag
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install langchain langchain-community langchain-core langchain-openai langchain-chroma langchain-text-splitters chromadb unstructured python-dotenv
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
API_KEY=your_openai_api_key
```

Never commit your actual API key to GitHub.

### 5. Initialize the Vector Database

Run the indexing script:

```bash
python db_init.py
```

This processes the source documents, generates embeddings, and creates the persistent ChromaDB collection.

Run the indexing script again whenever you need to rebuild the knowledge base after changing the source documents.

### 6. Start the Application

```bash
python main.py
```

Ask questions about Budapest in the terminal. The application retrieves relevant document chunks, generates a response, and displays the retrieved sources.

## Example Questions

```text
Which attractions should I visit during my first day in Budapest?

What public transport options are available in Budapest?

What should tourists check before visiting an attraction?

I enjoy walking and scenic views. What places would you recommend,
and how should I plan my day?

What is the population of Budapest in 2026?
```

The final question can be used to test how the application handles information that may not be present in its knowledge base.

## Design Considerations

- **Separation of indexing and querying** — Document ingestion and vector database initialization are handled separately from the runtime application.
- **Persistent storage** — ChromaDB stores the indexed data locally so the database can be reused across application runs.
- **Configurable retrieval** — The retriever is configured to return the two most relevant document chunks per query.
- **Context-based generation** — The prompt explicitly instructs the model to use the supplied context and acknowledge missing information.

## Limitations

- The assistant's knowledge is limited to the indexed documents and the context retrieved for each query.
- The application does not currently provide live travel information or real-time public transport updates.
- Prompt instructions do not guarantee that the model will always refuse to answer unsupported questions.
- Retrieval quality depends on the source documents, chunking strategy, embeddings, and retrieval configuration.

## Future Improvements

- Add automated evaluation of retrieval relevance and answer quality.
- Experiment with chunk sizes, overlap, and retrieval parameters.
- Add source citations to generated answers.
- Implement a web-based user interface.
- Explore conversation history and follow-up questions.

## License

Add a license before distributing or reusing this project publicly.
