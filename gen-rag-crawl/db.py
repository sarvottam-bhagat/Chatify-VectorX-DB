# db.py
import os
from typing import List, Dict, Any, Optional
from vecx.vectorx import VectorX
from dotenv import load_dotenv

load_dotenv()

# Global variables to store VectorX client and index
_vectorx_client = None
_vectorx_index = None
_encryption_key = None
_index_name = "pydantic_ai_docs"

def get_vectorx_client():
    """Get or create VectorX client."""
    global _vectorx_client
    if _vectorx_client is None:
        api_token = os.getenv("VECTORX_API_TOKEN")
        if not api_token:
            raise ValueError("VECTORX_API_TOKEN environment variable is required")
        _vectorx_client = VectorX(token=api_token)
    return _vectorx_client

def get_encryption_key():
    """Get or generate encryption key."""
    global _encryption_key
    if _encryption_key is None:
        # First try to load from environment variable
        stored_key = os.getenv("VECTORX_ENCRYPTION_KEY")
        if stored_key:
            _encryption_key = stored_key
            print("Using stored encryption key from environment")
        else:
            # Generate new key
            client = get_vectorx_client()
            _encryption_key = client.generate_key()
            print(f"Generated encryption key: {_encryption_key}")
            print("IMPORTANT: Save this encryption key securely! You'll need it to access your data.")
            print("Add this to your .env file: VECTORX_ENCRYPTION_KEY=" + _encryption_key)
    return _encryption_key

def init_vectorx_index():
    """Initialize VectorX index."""
    global _vectorx_index
    if _vectorx_index is None:
        client = get_vectorx_client()
        key = get_encryption_key()
        
        # First try to get the index if it exists
        try:
            _vectorx_index = client.get_index(name=_index_name, key=key)
            print(f"Using existing VectorX index: {_index_name}")
        except Exception:
            # Index doesn't exist, try to create it
            try:
                client.create_index(
                    name=_index_name,
                    dimension=1536,  # OpenAI text-embedding-3-small dimension
                    key=key,
                    space_type="cosine"
                )
                print(f"Created new VectorX index: {_index_name}")
                _vectorx_index = client.get_index(name=_index_name, key=key)
            except Exception as e:
                # If creation fails due to conflict, try to get the existing index
                if "already exists" in str(e).lower():
                    print(f"Index {_index_name} already exists, connecting to existing index...")
                    try:
                        _vectorx_index = client.get_index(name=_index_name, key=key)
                        print(f"Successfully connected to existing VectorX index: {_index_name}")
                    except Exception as get_error:
                        print(f"Error connecting to existing VectorX index: {get_error}")
                        raise
                else:
                    print(f"Error creating VectorX index: {e}")
                    raise
    return _vectorx_index

class VectorXCollection:
    """Wrapper class to provide ChromaDB-like interface for VectorX."""
    
    def __init__(self):
        self.index = init_vectorx_index()
    
    def add(self, documents: List[str], embeddings: List[List[float]], 
            metadatas: List[Dict[str, Any]], ids: List[str]):
        """Add documents to VectorX index."""
        try:
            vectors_data = []
            for i, (doc, embedding, metadata, doc_id) in enumerate(
                zip(documents, embeddings, metadatas, ids)
            ):
                vector_data = {
                    "id": doc_id,
                    "vector": embedding,
                    "meta": {
                        "content": doc,
                        **metadata
                    }
                }
                vectors_data.append(vector_data)
            
            self.index.upsert(vectors_data)
            print(f"Added {len(vectors_data)} documents to VectorX")
        except Exception as e:
            print(f"Error adding documents to VectorX: {e}")
            raise
    
    def query(self, query_embeddings: List[List[float]], n_results: int = 5,
              include: Optional[List[str]] = None, where: Optional[Dict] = None):
        """Query VectorX index for similar vectors."""
        try:
            # VectorX query expects a single vector, so we'll use the first one
            query_vector = query_embeddings[0] if query_embeddings else []
            
            # Build filter if where clause is provided
            filter_dict = None
            if where:
                # Convert ChromaDB where clause to VectorX filter format
                filter_dict = self._convert_where_to_filter(where)
            
            results = self.index.query(
                vector=query_vector,
                top_k=n_results,
                filter=filter_dict,
                include_vectors=False
            )
            
            # Convert VectorX results to ChromaDB-like format
            documents = [[]]
            metadatas = [[]]
            ids = [[]]
            
            for result in results:
                # Extract content from meta
                content = result.get('meta', {}).get('content', '')
                documents[0].append(content)
                
                # Extract metadata (excluding content)
                meta = result.get('meta', {}).copy()
                meta.pop('content', None)
                metadatas[0].append(meta)
                
                ids[0].append(result.get('id', ''))
            
            return {
                "documents": documents,
                "metadatas": metadatas,
                "ids": ids
            }
        except Exception as e:
            print(f"Error querying VectorX: {e}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]]}
    
    def get(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None,
            include: Optional[List[str]] = None, limit: Optional[int] = None):
        """Get documents from VectorX index."""
        try:
            if ids:
                # Get specific documents by ID
                documents = []
                metadatas = []
                result_ids = []
                
                for doc_id in ids:
                    try:
                        vector_data = self.index.get_vector(doc_id)
                        if vector_data:
                            content = vector_data.get('meta', {}).get('content', '')
                            documents.append(content)
                            
                            meta = vector_data.get('meta', {}).copy()
                            meta.pop('content', None)
                            metadatas.append(meta)
                            result_ids.append(doc_id)
                    except:
                        continue
                
                return {
                    "documents": documents,
                    "metadatas": metadatas,
                    "ids": result_ids
                }
            else:
                # For getting all documents, we'll use a small random vector query
                # since VectorX doesn't have a direct "get all" method
                try:
                    # Get index info to see if it has any data
                    index_info = self.index.describe()
                    total_vectors = index_info.get('count', 0)
                    
                    if total_vectors == 0:
                        return {
                            "documents": [],
                            "metadatas": [],
                            "ids": []
                        }
                    
                    # Use a small random vector for the query instead of zeros
                    import random
                    dimension = index_info.get('dimension', 1536)
                    random_vector = [random.uniform(-0.1, 0.1) for _ in range(dimension)]
                    
                    # Query with a limited top_k (VectorX max is 200)
                    query_limit = min(total_vectors, limit or 200, 200)
                    results = self.index.query(
                        vector=random_vector,
                        top_k=query_limit,
                        include_vectors=False
                    )
                    
                    documents = []
                    metadatas = []
                    result_ids = []
                    
                    for result in results:
                        content = result.get('meta', {}).get('content', '')
                        documents.append(content)
                        
                        meta = result.get('meta', {}).copy()
                        meta.pop('content', None)
                        metadatas.append(meta)
                        
                        result_ids.append(result.get('id', ''))
                    
                    return {
                        "documents": documents,
                        "metadatas": metadatas,
                        "ids": result_ids
                    }
                except Exception as e:
                    print(f"Error in get all documents fallback: {e}")
                    # If all else fails, return the count we know from describe
                    index_info = self.index.describe()
                    total_count = index_info.get('count', 0)
                    print(f"Index has {total_count} documents, but unable to retrieve them")
                    return {
                        "documents": [],
                        "metadatas": [],
                        "ids": []
                    }
        except Exception as e:
            print(f"Error getting documents from VectorX: {e}")
            return {"documents": [], "metadatas": [], "ids": []}
    
    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None):
        """Delete documents from VectorX index."""
        try:
            if ids:
                for doc_id in ids:
                    self.index.delete_vector(doc_id)
                print(f"Deleted {len(ids)} documents from VectorX")
            elif where:
                # Convert where clause to VectorX filter
                filter_dict = self._convert_where_to_filter(where)
                self.index.delete_with_filter(filter_dict)
                print(f"Deleted documents matching filter from VectorX")
        except Exception as e:
            print(f"Error deleting documents from VectorX: {e}")
    
    def _convert_where_to_filter(self, where: Dict) -> Dict:
        """Convert ChromaDB where clause to VectorX filter format."""
        # Basic conversion - VectorX uses different filter syntax
        # This is a simplified conversion, may need adjustment based on actual usage
        vectorx_filter = {}
        for key, value in where.items():
            if isinstance(value, str):
                vectorx_filter[key] = {"eq": value}
            elif isinstance(value, dict):
                vectorx_filter[key] = value
            else:
                vectorx_filter[key] = {"eq": value}
        return vectorx_filter

def init_collection():
    """Initialize and return VectorX collection wrapper."""
    return VectorXCollection()
