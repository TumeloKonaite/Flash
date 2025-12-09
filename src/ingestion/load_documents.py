from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document  # optional, just for type hints

# Project root: .../Banking-rag/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Base data directory: .../Banking-rag/data
DATA_DIR = PROJECT_ROOT / "data"


def fetch_documents(base_dir: Path | str | None = None) -> List[Document]:
    """
    Fetch PDF documents from the data directory.

    This is the bulk ingestion path:
    - walks data/<doc_type>/ subfolders
    - loads all PDFs under each folder
    - tags documents with a 'doc_type' metadata field.
    """
    if base_dir is None:
        base_dir = DATA_DIR

    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"Data folder not found at: {base_dir}")

    # Each subfolder of data/ becomes a doc_type
    folders = [f for f in base_dir.glob("*") if f.is_dir()]

    documents: List[Document] = []

    for folder in folders:
        doc_type = folder.name  # e.g. 'product_terms', 'pricing_guides', etc.

        loader = DirectoryLoader(
            str(folder),
            glob="**/*.pdf",           # 🔹 focus on PDF files
            loader_cls=PyPDFLoader,    # 🔹 use PyPDFLoader under the hood
            show_progress=True,
        )

        folder_docs = loader.load()

        for doc in folder_docs:
            # add doc_type metadata for downstream filtering/routing
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    return documents


def load_uploaded_pdf(path: Path | str, doc_type: str = "uploaded") -> List[Document]:
    """
    Load a single uploaded PDF into LangChain Document objects.

    This is the online path:
    - FastAPI/Gradio saves the uploaded file to a temp location
    - we call this helper to get a list[Document]
    - we optionally tag the documents with a 'doc_type' like 'uploaded' or 'flashcards'
    """
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Uploaded PDF not found at: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()

    for doc in docs:
        # make sure there's a doc_type for downstream logic
        doc.metadata.setdefault("doc_type", doc_type)

    return docs
