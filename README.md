# Vilnius RAG Chatbot

This Streamlit application is an AI-powered Retrieval-Augmented Generation (RAG) chatbot that answers questions about Vilnius using three main sources:
- A YouTube video transcript
- A local text file (`Vilnius.txt`)
- The Lithuanian Wikipedia article for Vilnius

## Features

- **Multi-source knowledge:** Combines information from YouTube, Wikipedia, and a local file.
- **Semantic search:** Uses vector embeddings to retrieve the most relevant content chunks for your query.
- **Conversational AI:** Powered by a GPT-4.1 model for natural language answers.
- **Interactive UI:** Built with Streamlit for easy use.

## How it works

1. **Data Loading:**  
   - Downloads and transcribes a YouTube video.
   - Reads a local text file.
   - Fetches and parses the Wikipedia article.
2. **Chunking:**  
   - Splits all content into manageable text chunks using `RecursiveCharacterTextSplitter`.
3. **Vector Store:**  
   - Embeds the chunks and stores them in a Chroma vector database.
4. **Retrieval & Generation:**  
   - For each user question, retrieves the top relevant chunks and generates an answer using the GPT model.

## Setup

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/vilnius-rag-chatbot.git
   cd vilnius-rag-chatbot
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Environment variables:**
   - Create a `.env` file in the project root with the following:
     ```
     SECRET=your_openai_or_azure_api_key
     WIKI_USER_EMAIL=your_email@example.com
     ```

4. **Add your local file:**
   - Place `Vilnius.txt` in the project directory.

## Running the App

```sh
streamlit run streamlit-langchain.py
```

## Usage

- Enter your questions about Vilnius in the text area and submit.
- The app will display the answer and show the most relevant content sources used.

## Sources

1. [YouTube Video](https://www.youtube.com/watch?v=o0PmNtsqCzA)
2. Local file: `Vilnius.txt`
3. [Wikipedia Article](https://lt.wikipedia.org/wiki/Vilnius)

---

**Note:**  
- Make sure you have a valid API key for the embedding and chat model.
- The app requires internet access to fetch the YouTube transcript and Wikipedia content.

## License

MIT