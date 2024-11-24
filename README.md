# Q&A Chatbot using Langchain and Google Gemini

This project implements a Question & Answer (Q&A) chatbot using Langchain and Google Gemini's language model. The chatbot is designed to allow users to input their data, and then ask questions based on that data. The system leverages vector databases to store user data and the Google Gemini model to answer questions related to the stored data.

## Features

- Users can input their data, which will be stored in a vector store for fast retrieval.
- Users can ask questions about their stored data.
- The chatbot uses the Google Gemini model to process queries and return relevant answers.
- The system provides personalized responses by filtering data based on the user's identity (username).

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.7 or higher
- A Google API key for accessing Google Gemini (generative AI)
- Installed necessary Python packages (`langchain`, `chromadb`, `google-auth`, etc.)

## Installation

Follow these steps to get your development environment set up:

1. Clone the repository:

   ```bash
   git clone https://github.com/alihassanml/Q-A-Chatbot-Langchain-Gemini.git
   cd Q-A-Chatbot-Langchain-Gemini
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Google API credentials:
   
   - Follow the [Google Cloud Authentication guide](https://cloud.google.com/docs/authentication/getting-started) to generate a service account key.
   - Set your `GOOGLE_API_KEY` in the environment variables:

     ```bash
     export GOOGLE_API_KEY="your-google-api-key"
     ```

## Usage

1. **Input User Data**: 
   The user can input their data, which will be stored and indexed in a vector database (Chroma in this case). The data can be anything, such as documents, information, etc.

2. **Ask Questions**:
   After data is inputted, the user can ask questions. The model will retrieve the relevant information from the database and respond accordingly.

   Example usage in Python:

   ```python
   # Import necessary libraries
   from langchain.chains import ConversationChain
   from langchain.chat_models import ChatGoogleGenerativeAI
   from langchain.prompts import PromptTemplate
   from langchain.vectorstores import Chroma
   from langchain.embeddings import GoogleGenerativeAIEmbeddings

   # Define the Google Gemini model
   api_key = "your-google-api-key"
   model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.2, api_key=api_key)

   # Define the Q&A chain with Langchain
   prompt_template = """
   Answer the question as detailed as possible from the context provided. If answer not available then answer is yes or no. Don't give wrong answer.
   Context: \n{context}\n
   Question: \n{question}\n
   Answer:
   """
   prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
   chain = ConversationChain(llm=model, prompt=prompt)

   # Example input data and user question
   username = "ali"
   data = ["What is machine learning?", "Machine learning is a subset of artificial intelligence."]
   
   # Add data to Chroma (vector store)
   embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
   vectorstore = Chroma(collection_name="user_data", embedding_function=embeddings, persist_directory="./data")
   vectorstore.add_texts(texts=data, metadatas=[{"username": username}])

   # Perform similarity search and get a response
   query_text = "What is machine learning?"
   results = vectorstore.similarity_search(query=query_text, k=5, filter={"username": username})
   
   # Pass the results and user query to the chatbot model
   context = [result.page_content for result in results]
   response = chain({"input_documents": context, "question": query_text}, return_only_outputs=True)
   
   print(response['output'])
   ```

## Configuration

You can adjust the following settings in the code:

- **API Key**: Replace `"your-google-api-key"` with your actual Google API key.
- **Model**: You can use different models, like `gemini-1.5-pro` for high-quality responses.
- **Vector Store**: You can change the vector store settings (e.g., Chroma configuration) if needed.

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Langchain: https://github.com/hwchase17/langchain
- Google Gemini: https://cloud.google.com/generative-ai

```

### Instructions:
1. **Replace `your-google-api-key`** with your actual Google API key for the Gemini model.
2. This `README` contains sections on installation, usage, and configuration to guide users who want to run your chatbot project.
3. The example Python script shows how to interact with the chatbot, add user data to the vector store, and get answers from the model.

