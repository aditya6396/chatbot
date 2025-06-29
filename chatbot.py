from dotenv import load_dotenv
import gradio as gr
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.schema import AIMessage, HumanMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
import os
import requests
import json
from db_complaints import *
from rag import *

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

# API Base URL (Flask server runs on localhost:8000)
API_BASE_URL = "http://127.0.0.1:8000"

def load_chain():
    # Initialize database if not already done
    try:
        initialize_database()
    except:
        pass  # Database might already be initialized by api_server.py
    
    # Define available tools for complaint handling
    tools = [create_complaint, get_complaint_details, update_complaint_status]

    # System prompt defining chatbot's role, behavior, and constraints
    system = '''
You are ComplaintBot, an empathetic and professional AI chatbot specializing in handling customer complaints efficiently. Your primary goal is to assist customers by collecting their complaint information, creating complaint records, and retrieving complaint details when requested.

### Role
- **Primary Function:** Assist customers with filing complaints and retrieving complaint information.
- **Complaint Collection:** Gather name, phone number, email, and complaint details from customers.
- **Complaint Management:** Create new complaints and retrieve existing complaint details.

### Persona
- **Identity:** A warm, patient, and professional AI dedicated to customer service.
- **Behavior:**  
    - Be empathetic and acknowledge customer concerns.
    - Collect required information step by step if not provided initially.
    - Ask for missing details one at a time to avoid overwhelming the customer.
    - Provide clear confirmation when complaints are created.
    - Format complaint details nicely when retrieving information.

### Complaint Process
1. **Complaint Creation:**
   - Collect name, phone number, email, and complaint details
   - Ask for missing information one field at a time
   - Once all information is collected, create the complaint using the create_complaint tool
   - Provide the complaint ID and confirmation message

2. **Complaint Retrieval:**
   - When customer asks to see complaint details or provides a complaint ID
   - Use get_complaint_details tool to fetch information
   - Display the information in a clear, formatted manner

### Required Information for New Complaints:
- **Name:** Customer's full name
- **Phone Number:** Valid phone number
- **Email:** Valid email address  
- **Complaint Details:** Description of the issue

### Constraints
1. **Information Collection:** Do not proceed with complaint creation until ALL required fields are provided.
2. **One Field at a Time:** Ask for missing information one field at a time to keep the conversation natural.
3. **Validation:** Ensure email format is valid and phone number is provided.
4. **Clear Communication:** Keep responses concise and friendly.
5. **Focus:** Stay focused on complaint-related conversations.

You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names} ONLY

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
"action": $TOOL_NAME,
"action_input": $INPUT
}}
```

Follow this format:

RAG: gives extra info relevant to question
Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
{{
"action": $TOOL_NAME (one of the {tool_names}),
"action_input": $INPUT
}}
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
"action": "Final Answer",
"action_input": "response to customer"
}}
```

Begin! This is a conversation, so ask for missing information via "Final Answer" before using tools. Only use tools when you have all required information or when retrieving complaint details. Reminder to ALWAYS respond with a valid json blob of a single action.

Example interactions:
- Customer: "I want to file a complaint" → Ask for their name first
- Customer: "Show me complaint ABC123" → Use get_complaint_details with the ID
- Customer provides partial info → Ask for the next missing field

Format is Action:```$JSON_BLOB```then Observation
'''

    # Human message template with placeholders for dynamic input
    human = '''
RAG: {retriever}
Question: {input}
Thought: If I need to collect complaint information, ask for missing details one at a time using "Final Answer". If I have a complaint ID to lookup or all details to create a complaint, use the appropriate tool.
{agent_scratchpad}
'''

    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
        ]
    )
    
    # Choose the LLM model (comment/uncomment based on preference)
    llm = ChatGroq(temperature=0, groq_api_key=os.getenv('GROQ_APIKEY'), model_name="meta-llama/llama-4-scout-17b-16e-instruct")
    # llm = OllamaLLM(model="llama3.2")
    # llm = ChatGroq(temperature=0, groq_api_key=os.getenv('GROQ_APIKEY'), model_name="mixtral-8x7b-32768")
    # llm = ChatGroq(temperature=0, groq_api_key=os.getenv('GROQ_APIKEY'), model_name="llama-3.3-70b-versatile")
    # llm = ChatOpenAI(temperature=0, model="gpt-4o", api_key=os.environ.get("OPENAI_APIKEY"))

    # Configure LLM to stop processing at certain responses
    llm_with_stop = llm.bind(stop=["\nFinal Answer", "\nNone"])

    # Create a structured chat agent
    agent = create_structured_chat_agent(llm_with_stop, tools, prompt)

    # Define agent executor settings
    chain = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=100,
    )

    return chain

def predict(message, history, chain):
    # Format chat history into LangChain message format
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))

    # Invoke the chatbot and get a response
    response = chain.invoke({
        "input": message,
        "chat_history": history_langchain_format,
        "retriever": get_rag(message)
    })
    print("Agent:", response['output'])
    return response['output'], chain

# Custom CSS for a vibrant, modern interface with colorful buttons
custom_css = """
body {
    background-color: #e6f0fa;
    font-family: 'Poppins', Arial, sans-serif;
}
.gradio-container {
    max-width: 900px;
    margin: auto;
    padding: 30px;
    background-color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}
.header {
    background: linear-gradient(90deg, #007bff, #00d4ff);
    color: white;
    padding: 25px;
    border-radius: 12px 12px 0 0;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}
.header h1 {
    margin: 0;
    font-size: 2.5em;
    font-weight: 600;
}
.description {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 8px;
    margin: 20px 0;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    font-size: 1.1em;
}
.chatbot-container {
    border: 1px solid #d0e7ff;
    border-radius: 10px;
    background-color: #ffffff;
}
.chat-message.bot {
    background-color: #e3f2fd;
    border-left: 4px solid #007bff;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 10px;
}
.chat-message.user {
    background-color: #e6ffed;
    border-left: 4px solid #28a745;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 10px;
}
.chatbot-textbox {
    border: 2px solid #007bff;
    border-radius: 8px;
}
button.submit {
    background-color: #28a745;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
}
button.submit:hover {
    background-color: #218838;
}
button.clear {
    background-color: #dc3545;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
}
button.clear:hover {
    background-color: #c82333;
}
button.retry {
    background-color: #fd7e14;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
}
button.retry:hover {
    background-color: #e86c12;
}
button.undo {
    background-color: #6f42c1;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
}
button.undo:hover {
    background-color: #5a32a3;
}
.footer {
    text-align: center;
    color: #555;
    margin-top: 30px;
    font-size: 0.95em;
}
"""

if __name__ == "__main__":
    # Initialize Gradio interface with custom theme
    with gr.Blocks(css=custom_css) as block:
        gr.Markdown(
            """
            <div class='header'>
                <h1>🌟 CustomerCare Bot</h1>
                <p>Your friendly assistant for seamless customer support!</p>
            </div>
            """,
            elem_classes=["header"]
        )
        gr.Markdown(
            """
            **Welcome to CustomerCare Bot!** 😊  
            I'm here to make your experience smooth and stress-free:  
            - 📋 **File a Complaint** - Share your details, and I'll guide you step-by-step  
            - 🔎 **Track Complaints** - Enter your complaint ID (e.g., CMP001) to check status  
            - 💡 **Get Expert Tips** - Ask for customer service best practices  
            
            **Ready to start?** Type your request below, like "I want to file a complaint" or "Show details for CMP001".
            """,
            elem_classes=["description"]
        )
        
        chain_state = gr.State(load_chain)
        chatbot = gr.ChatInterface(
            fn=predict,
            additional_inputs=[chain_state],
            additional_outputs=[chain_state],
            title="Customer Care Bot",
            description="Your trusted partner for managing customer complaints",
            theme="soft"
        )
        
        gr.Markdown(
            """
            <div class='footer'>
                Powered by xAI | Built with Gradio & LangChain | © 2025
            </div>
            """,
            elem_classes=["footer"]
        )

    # Launch Gradio app
    block.launch()

    # Uncomment below for a shareable link
    # block.launch(share=True)