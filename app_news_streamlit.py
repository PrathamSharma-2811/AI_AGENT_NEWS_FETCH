import requests
from bs4 import BeautifulSoup
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import os
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import streamlit as st
from newsapi import NewsApiClient

load_dotenv()

class ArticleExtractorSummarizer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.host = "article-extractor-and-summarizer.p.rapidapi.com"
        self.headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": self.host,
            "Content-Type": "application/json"
        }

    def extract_text(self, url):
        endpoint = "https://article-extractor-and-summarizer.p.rapidapi.com/extract"
        payload = {"url": url}
        response = requests.post(endpoint, json=payload, headers=self.headers)
        return response.json()

    def summarize(self, url):
        endpoint = "https://article-extractor-and-summarizer.p.rapidapi.com/summarize"
        payload = {"url": url}
        response = requests.post(endpoint, json=payload, headers=self.headers)
        return response.json()

    def summarize_text(self, text, lang="en"):
        endpoint = "https://article-extractor-and-summarizer.p.rapidapi.com/summarize-text"
        payload = {
            "lang": lang,
            "text": text
        }
        response = requests.post(endpoint, json=payload, headers=self.headers)
        return response.json()

    def extract_text_from_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()

class JinaScraper:

    def jinaai_readerapi_web_scrape_url(self, url):
        response = requests.get("https://r.jina.ai/" + url)
        return response.text

class NewsSearchInput(BaseModel):
    keyword: str = Field(description="Search keyword")
    endpoint: str = Field(description="NewsAPI endpoint to use ('sources', 'top-headlines', or 'everything')")
    sources: Optional[str] = Field(default=None, description="Comma-separated list of news sources")

class CustomNewsSearchTool(BaseTool):
    name = "NewsSearch"
    description = "Fetches news articles based on a keyword and endpoint"
    args_schema: Type[BaseModel] = NewsSearchInput
    return_direct: bool = True

    def _run(
        self, keyword: str, endpoint: str, sources: Optional[str] = None, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Fetch news articles and format them."""
        api_key = os.getenv("news_api")  # Replace with your News API key

        if endpoint == "sources":
            url = "https://newsapi.org/v2/top-headlines/sources"
            params = {"apiKey": api_key, "language": "en"}
        elif endpoint == "top-headlines":
            url = "https://newsapi.org/v2/top-headlines"
            params = {"apiKey": api_key, "q": keyword, "language": "en","from_param":'2024-08-01'}
        elif endpoint == "everything":
            url = "https://newsapi.org/v2/everything"
            params = {"apiKey": api_key, "q": keyword, "sources": sources, "language": "en"}
        else:
            return "Invalid endpoint specified."

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "ok":
                if endpoint == "sources":
                    sources = data.get("sources", [])
                    return "\n".join([f"{source['name']}: {source['description']}" for source in sources])

                articles = data.get("articles", [])
                formatted_articles = []
                for article in articles[:4]:
                    formatted_articles.append({
                        "title": article.get("title", "No Title"),
                        "snippet": article.get("description", "No Snippet"),
                        "publisher": article.get("source", {}).get("name", "No Publisher"),
                        "newsUrl": article.get("url", "No URL"),
                        "imageUrl": article.get("urlToImage"),
                    })

                # Improved result formatting
                result = ""
                for article in formatted_articles:
                    result += f"### {article['title']}\n"
                    result += f"**Snippet**: {article['snippet']}\n"
                    result += f"**Publisher**: {article['publisher']}\n"
                    result += f"[Read More]({article['newsUrl']})\n"
                    if article['imageUrl']:
                        result += f"![Image]({article['imageUrl']})\n"
                    result += "---\n"
                return result
            else:
                return "Failed to retrieve news items."
        except requests.exceptions.RequestException as e:
            return f"An error occurred: {e}"

    async def _arun(
        self, keyword: str, endpoint: str, sources: Optional[str] = None, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Fetch news articles and format them asynchronously."""
        return self._run(keyword, endpoint, sources, run_manager=run_manager.get_sync())

class LangChainAgent:
    def __init__(self, api_key):
        genai.configure(api_key=os.environ["gemini_key"])
        self.article_tool = ArticleExtractorSummarizer(api_key)
        self.news_search_tool = CustomNewsSearchTool()
        self.llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=os.getenv("hf_token"))
#         self.llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro-latest",
#     api_key=os.getenv('gemini_key'),
# )
        self.jina_scraper = JinaScraper()

        # Define the prompt
        self.prompt = PromptTemplate(
    input_variables=["input", "tools", "agent_scratchpad", "tool_names"],

template="""
   YOU ARE NEWS_FETCH.AI, A HIGHLY INTELLIGENT AND SPECIALIZED BOT DESIGNED TO PROVIDE USERS WITH THE MOST ACCURATE, RELEVANT, AND UP-TO-DATE NEWS BASED ON THEIR QUERIES.


    You have access to the following tools:

    {tools}

    ### IMPORTANT ###
    1. Every News articles you return mus include title ,url to image, url and description of the news article(s).
    2. If the query is broad or requires multiple articles, provide the top 2 relevant news items

    Follow this structured approach to ensure accuracy:

            1. **Extract Subject**: Identify the main subject or keyword from the user's query.
            2. **Select Action**: Choose the most appropriate tool from {tool_names}.
            3. **Action Execution**: Pass the extracted subject as the keyword to the selected tool.
            4. **Observation**: Record the output or result generated by the action.
            5. **Synthesize**: Analyze the results and prepare a coherent final response.

    Use the following format:

            Question: {input}
            Thought: Analyze the input to understand the user's request and determine the best approach.
            Action: Choose the appropriate action from {tool_names} and extract the subject for the news search.
            Action Input: Pass the extracted keyword to the `NewsSearch` tool.
            Observation: Record the result of the action.
            Thought: Synthesize the information and prepare the final response.
            Final Answer:the final answer to the original input question line by line , The news aand its title ,render the images from imag links, url should be represented together in a single point formatted manner and should be clearly seperated from other articles information,Correctly align the articles displaying as well. 
    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
)

        

        # Define the tools for the agent
        self.tools = [
            Tool(
                name="NewsSearch",
                func=lambda keyword: self.news_search_tool._run(keyword=keyword),
                description="Fetches news from keywords using the 'everything' endpoint. Provides articles including title, image URL, clickable link, and a brief description."
            ),
            Tool(
                name="extract_text",
                func=self.article_tool.extract_text,
                description="Extracts text from a given URL."
            ),
            Tool(
                name="summarize",
                func=self.article_tool.summarize,
                description="Summarizes the content of a given URL."
            ),
            Tool(
                name="extract_text_from_html",
                func=self.article_tool.extract_text_from_html,
                description="Extracts plain text from HTML content."
            ),
            Tool(
                name="jinaai_readerapi_web_scrape_url",
                func=self.jina_scraper.jinaai_readerapi_web_scrape_url,
                description="Reads data from a URL using Jina's web scraping API."
            ),
        ]

        # Create the ReAct agent
        self.react_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Create the AgentExecutor
        self.agent_executor = AgentExecutor(
            agent=self.react_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            
        )

    def run(self, query, endpoint="everything"):
        self.endpoint = endpoint  # Set the endpoint to use in the tools
        try:
            result = self.agent_executor.invoke({"input": query})

            if isinstance(result, dict):
                return result
            else:
                return {"output": result}
        except Exception as e:
            return {"error": str(e)}

# Define the Streamlit UI

def main():
    st.title("AI News Feed")
    st.image("G:\AI_News_Feed\image.png")

    st.write("Enter a query to fetch and summarize news articles or scrape a URL.")

    query = st.text_input("Query")
    endpoint = st.selectbox("Select News Formats", ["sources", "top-headlines", "everything"])

    if st.button("Get News"):
        api_key = os.getenv("extract__key")
        langchain_agent = LangChainAgent(api_key)
        result = langchain_agent.run(query, endpoint)
        st.markdown(result['output'])
        # st.markdown(result)

if __name__ == "__main__":
    main()

