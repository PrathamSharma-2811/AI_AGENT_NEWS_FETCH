
# News Fetch AI

**News Fetch AI** is an advanced tool designed to deliver precise, relevant, and up-to-date news articles tailored to user queries. By leveraging LangChain's AI agents, this project automates the process of fetching and summarizing news, ensuring users receive organized and relevant information effortlessly.

## Project Overview

**News Fetch AI** revolutionizes news access by using LangChain's intelligent agents to interpret queries, choose the best tools, and automate the extraction and summarization of content. This approach guarantees users get comprehensive and tailored news updates.

### Key Features

1. **Intelligent News Search and Aggregation**: 
   - LangChain agents perform keyword-based searches using the News API, covering endpoints like "sources," "top-headlines," and "everything."

2. **Automated Content Extraction and Summarization**: 
   - Uses tools like the Article Extractor and Summarizer to efficiently condense information from news articles, providing users with key insights.

3. **Efficient Web Scraping**: 
   - Jina's web scraping capabilities are used to extract data from various URLs, ensuring no relevant information is overlooked.

4. **Advanced LLM Integration**:
   - Integrates with HuggingFace and Google Gemini models to enhance the understanding of user queries, ensuring accurate and context-aware responses.

5. **User-Friendly Interface**:
   - Built with Streamlit, offering a seamless and interactive experience for users.
   - A REST API backend using Flask handles queries efficiently.

### Technical Stack

- **Python**: Core language for development.
- **Flask**: Handles the REST API for processing queries.
- **Streamlit**: Provides an intuitive web interface.
- **NewsAPI**: Fetches news articles based on user input.
- **RapidAPI**: Extracts and summarizes article content.
- **Jina AI**: Used for web scraping.
- **LangChain and HuggingFace**: Advanced NLP models and agents for intelligent responses.

## How It Works

1. **User Query**: Users input a query via Streamlit or the REST API.
2. **Agent Processing**: LangChain agents analyze the query, choose the appropriate tools, and execute actions (e.g., news search, content extraction).
3. **Automated Response**: Using the selected tools and LLMs, the system fetches and summarizes news, generating a structured response with article titles, summaries, images, and URLs.
4. **Output**: The response is presented in the user interface or returned as a JSON object.

## Conclusion

**News Fetch AI** leverages the power of LangChain's AI agents to intelligently automate news searching and summarization. By choosing the best tools and integrating advanced NLP models, it provides a streamlined, accurate, and user-friendly way to stay informed.
