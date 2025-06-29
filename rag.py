from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(dotenv_path=".env")

def get_embedding_model():
    """
    Choose and return the appropriate embedding model based on configuration.
    Priority: Hugging Face > Ollama > Fallback to sentence-transformers
    """
    # Check which embedding model to use from environment variable
    embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'huggingface').lower()
    
    try:
        if embedding_provider == 'huggingface' or embedding_provider == 'hf':
            # Hugging Face embeddings (default)
            model_name = os.getenv('HF_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            print(f"Using Hugging Face embedding model: {model_name}")
            
            # Configure model with device preference
            model_kwargs = {'device': 'cpu'}  # Change to 'cuda' if you have GPU
            encode_kwargs = {'normalize_embeddings': True}
            
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
        elif embedding_provider == 'ollama':
            # Ollama embeddings
            model_name = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
            print(f"Using Ollama embedding model: {model_name}")
            return OllamaEmbeddings(model=model_name)
            
        else:
            # Fallback to default Hugging Face model
            print("Unknown embedding provider, falling back to Hugging Face")
            return HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
    except Exception as e:
        print(f"Error initializing {embedding_provider} embeddings: {e}")
        print("Falling back to sentence-transformers/all-MiniLM-L6-v2")
        
        # Final fallback
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

# Initialize the embedding model
embeddings = get_embedding_model()

def setup_rag_system():
    """Set up the RAG system with customer service knowledge base."""
    
    # Try to load PDF file first, then fallback to text file if needed
    try:
        # Load and preprocess the document (PDF)
        if os.path.exists("Customer_Service_Guidelines.pdf"):
            loader = PyPDFLoader(file_path="Customer_Service_Guidelines.pdf")
            print("Loading PDF knowledge base...")
        elif os.path.exists("customer_service_guidelines.txt"):
            loader = TextLoader(file_path="customer_service_guidelines.txt")
            print("Loading text knowledge base...")
        else:
            # Create a default knowledge base if no file exists
            print("Creating default knowledge base...")
            create_default_knowledge_base()
            loader = TextLoader(file_path="customer_service_guidelines.txt")
            
        # Split the document into smaller chunks of text for easier processing
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, 
            chunk_overlap=50
        )
        doc_splits = loader.load_and_split(text_splitter)
        print(f"Document split into {len(doc_splits)} chunks")
        
        # Create a Chroma vector store to hold the document embeddings
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="complaint-rag-chroma",
            embedding=embeddings,
            persist_directory="./chroma_db"  # Persist the vector store
        )
        
        # Create a retriever to perform searches on the vector store
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        print("RAG system setup completed successfully!")
        
        return retriever
        
    except Exception as e:
        print(f"Error setting up RAG system: {e}")
        return None

def create_default_knowledge_base():
    """Create a default customer service knowledge base if no file exists."""
    
    default_content = """
# Customer Service Guidelines for Complaint Management

## General Principles
- Always acknowledge the customer's concern with empathy
- Collect all necessary information before proceeding
- Provide clear and timely responses
- Follow up on complaint resolutions

## Complaint Handling Process

### Information Collection
When handling complaints, collect the following information:
1. Customer's full name
2. Contact phone number
3. Email address
4. Detailed description of the complaint

### Response Guidelines
- Be polite and professional at all times
- Ask for information one step at a time to avoid overwhelming customers
- Acknowledge the issue and apologize for any inconvenience
- Provide clear next steps and timelines

### Common Complaint Types
- Delivery issues (late, wrong address, damaged items)
- Product quality concerns
- Billing and payment problems
- Customer service experience issues

### Resolution Procedures
1. Document all complaint details accurately
2. Assign a unique complaint ID for tracking
3. Investigate the issue thoroughly
4. Provide appropriate compensation or resolution
5. Follow up to ensure customer satisfaction

### Communication Best Practices
- Use clear, simple language
- Be patient and understanding
- Confirm understanding by summarizing key points
- Provide realistic timelines for resolution

## Escalation Procedures
If a complaint cannot be resolved immediately:
1. Document all details in the system
2. Escalate to appropriate department
3. Inform customer of escalation and expected timeline
4. Ensure proper handoff and follow-up

## Quality Standards
- Respond to all complaints within 24 hours
- Aim for first-contact resolution when possible
- Maintain detailed records of all interactions
- Conduct follow-up surveys for satisfaction measurement

## Advanced Complaint Handling

### Difficult Customers
- Remain calm and professional
- Listen actively to understand their frustration
- Empathize with their situation
- Offer multiple solution options when possible
- Know when to escalate to a supervisor

### Documentation Requirements
- Record all customer interactions
- Include timestamps and agent information
- Note any promises made to customers
- Track resolution progress
- Maintain confidentiality of customer information

### Follow-up Procedures
- Contact customer within 24-48 hours of resolution
- Verify customer satisfaction with the solution
- Address any remaining concerns
- Update records with follow-up results
- Use feedback to improve processes

## Performance Metrics
- First Contact Resolution Rate
- Average Response Time
- Customer Satisfaction Score
- Complaint Resolution Time
- Escalation Rate
"""
    
    with open("customer_service_guidelines.txt", "w") as f:
        f.write(default_content)

# Initialize the RAG system
retriever = setup_rag_system()

def get_rag(query: str):
    """Returns information based on the query using RAG (Retrieval-Augmented Generation).
    Takes a string query as input and retrieves the most relevant document chunks.
    """
    if retriever is None:
        return "No knowledge base available"
    
    try:
        results = retriever.invoke(query)
        if results:
            # Combine multiple results if available
            combined_content = "\n".join([doc.page_content for doc in results])
            return combined_content
        else:
            return "No relevant information found"
    except Exception as e:
        print(f"Error in RAG retrieval: {e}")
        return "Error retrieving information"

def test_embedding_models():
    """Test different embedding models to see which ones work."""
    test_text = "How to handle customer complaints effectively?"
    
    print("Testing embedding models...")
    
    # Test Hugging Face models
    hf_models = [
        'sentence-transformers/all-MiniLM-L6-v2',
        'sentence-transformers/all-mpnet-base-v2',
        'sentence-transformers/paraphrase-MiniLM-L6-v2',
        'BAAI/bge-small-en-v1.5',
        'thenlper/gte-small'
    ]
    
    for model_name in hf_models:
        try:
            embeddings_test = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # Test embedding
            test_embedding = embeddings_test.embed_query(test_text)
            print(f"✅ {model_name}: Working (embedding dimension: {len(test_embedding)})")
        except Exception as e:
            print(f"❌ {model_name}: Error - {str(e)}")
    
    # Test Ollama if available
    try:
        ollama_embeddings = OllamaEmbeddings(model='nomic-embed-text')
        test_embedding = ollama_embeddings.embed_query(test_text)
        print(f"✅ Ollama nomic-embed-text: Working (embedding dimension: {len(test_embedding)})")
    except Exception as e:
        print(f"❌ Ollama nomic-embed-text: Error - {str(e)}")

# Test the system
if __name__ == "__main__":
    # Test embedding models
    print("=" * 50)
    test_embedding_models()
    print("=" * 50)
    
    # Test RAG system
    test_query = "How should I handle customer complaints?"
    result = get_rag(test_query)
    print("RAG Test Result:", result)