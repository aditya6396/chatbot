# Customer Complaint Management System

## Overview
This project is a comprehensive Customer Complaint Management System built using LangChain, Gradio, and a Flask API, integrated with a SQLite database. It features a chatbot interface (`CustomerCare Bot`) that assists users in filing complaints, retrieving complaint details, updating statuses, and accessing customer service best practices via a Retrieval-Augmented Generation (RAG) system. The solution supports multiple embedding models (Hugging Face, Ollama) and leverages APIs like Groq and OpenAI for enhanced AI capabilities, ensuring a robust and scalable customer support tool.

## Features
- **Complaint Filing**: Guides users step-by-step to register complaints with unique IDs, storing details in a SQLite database.
- **Complaint Retrieval**: Fetches and displays complaint details by ID, including name, phone, email, and creation time.
- **Status Updates**: Allows updating complaint statuses (e.g., "pending" to "resolved").
- **RAG Knowledge Base**: Provides expert tips on handling complaints and customer service best practices using document embeddings.
- **Multi-Model Support**: Supports Hugging Face and Ollama embeddings, with fallback to default models.
- **Gradio Interface**: Offers an interactive chatbot UI with custom styling for user-friendly interaction.
- **API Integration**: Includes a Flask API for backend complaint management, accessible via HTTP endpoints.
- **Error Handling**: Manages invalid inputs, database errors, and model initialization failures gracefully.

## Prerequisites
- Python 3.8+
- Required Libraries (from `requirements_txt.txt`):
  - `fastapi`
  - `uvicorn`
  - `pydantic`
  - `python-dotenv`
  - `gradio`
  - `langchain`
  - `langchain-core`
  - `langchain-groq`
  - `langchain-ollama`
  - `langchain-openai`
  - `langchain-community`
  - `langchain-huggingface`
  - `chromadb`
  - `pypdf`
  - `tiktoken`
  - `requests`
  - `sentence-transformers`
  - `torch`
  - `transformers`

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd customer-complaint-management
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements_txt.txt
   ```

4. **Configure Environment**:
   - Create a `.env` file based on `env_example.txt`:
     ```
     POSTGRES_PASSWORD=your_postgres_password
     POSTGRES_PORT=5432
     GROQ_APIKEY=your_groq_api_key
     OPENAI_APIKEY=your_openai_api_key
     EMAIL_ID=your_email@gmail.com
     EMAIL_PASSWORD=your_app_password
     ```
   - Update with your actual credentials.

5. **Run the Application**:
   - Start the Flask API: `python api_server.py`
   - Launch the Gradio chatbot: `python chatbot.py` or `python app.py`

## Usage

### Chatbot Interface
- Open the Gradio app (default: http://127.0.0.1:7860).
- Interact with the chatbot:
  - **File a Complaint**: Type "I want to file a complaint" and follow prompts.
  - **Retrieve Details**: Enter "Show details for CMP001".
  - **Update Status**: Use "Update status of complaint CMP001 to resolved".
  - **Query Knowledge**: Ask "How should I handle customer complaints?".

### API Endpoints
- **POST /complaints**: Create a new complaint (e.g., `{"name": "John Doe", "phone_number": "1234567890", "email": "john@example.com", "complaint_details": "Delayed order"}`).
- **GET /complaints/<complaint_id>**: Retrieve complaint details.
- **GET /**: Root endpoint confirmation.
- **GET /health**: Health check.

## Project Structure
- **`requirements_txt.txt`**: List of Python dependencies.
- **`env_example.txt`**: Example environment variable configuration.
- **`rag.py`**: Handles document loading, embedding, and RAG retrieval with multiple model support.
- **`db_complaints.py`**: Manages SQLite database operations for complaints.
- **`api_server.py`**: Flask API for complaint CRUD operations.
- **`complain.txt`**: Example chatbot interactions and expected responses.
- **`chatbot.py`**: Gradio-based chatbot with LangChain agent logic.
- **`app.py`**: Alternative Gradio app with hosted API integration.

## Configuration
- **Embedding Provider**: Set `EMBEDDING_PROVIDER` (e.g., `huggingface`, `ollama`) and `HF_EMBEDDING_MODEL` or `OLLAMA_EMBEDDING_MODEL` in `.env`.
- **API URL**: Update `API_BASE_URL` in `app.py` to match the hosted server (e.g., `https://chatbot-sgfr.onrender.com`).
- **Database**: Uses `complaints.db` locally; configure `POSTGRES_PASSWORD` and `POSTGRES_PORT` for PostgreSQL if needed.

## Notes
- The system initializes with sample data in `complaints.db`.
- Supports CPU; switch to `'cuda'` in `model_kwargs` for GPU acceleration.
- Today's date and time: 05:24 PM IST on Wednesday, August 20, 2025.

## Troubleshooting
- **API Not Responding**: Ensure `api_server.py` is running or the hosted URL is accessible.
- **Embedding Errors**: Verify model names and internet connectivity for Hugging Face/Ollama.
- **Database Issues**: Check `complaints.db` permissions and `.env` configuration.
- **Gradio Launch Failure**: Ensure port 7860 is free.

## Contributing
Contributions are welcome! Submit PRs for UI enhancements, additional tools, or database optimizations.

## Future Scope
- **Multi-Language Support**: Extend the chatbot to handle complaints in multiple languages.
- **Email Notifications**: Integrate email alerts using `EMAIL_ID` and `EMAIL_PASSWORD` for status updates.
- **Advanced Analytics**: Add dashboards to analyze complaint trends and resolution times.
- **Scalability**: Migrate to PostgreSQL for larger datasets and deploy with a load balancer.

## Hosting
This application is hosted live on Hugging Face Spaces at [https://huggingface.co/spaces/addy6396/ASB_chatbot]. Access the deployed version for real-time complaint management without local setup.
