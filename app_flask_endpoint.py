import requests
from bs4 import BeautifulSoup
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
import streamlit as st
from newsapi import NewsApiClient
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import re




app = Flask(__name__)
CORS(app)

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
    """Reads the data in a url and gives an output in a formatted text"""

    def jinaai_readerapi_web_scrape_url(self, url):
        response = requests.get("https://r.jina.ai/" + url)
        return response.text

class NewsSearchInput(BaseModel):
    keyword: str = Field(description="Search keyword")
    endpoint: str = Field(description="NewsAPI endpoint to use ('sources', 'top-headlines', or 'everything')")
    sources: Optional[str] = Field(default=None, description="Comma-separated list of news sources")

class CustomNewsSearchTool(BaseTool):
    name = "NewsSearch"
    description = "Fetches news articles based on a keyword using the 'everything' endpoint."
    args_schema: Type[BaseModel] = NewsSearchInput
    return_direct: bool = True

    def _run(self, keyword: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> dict:
        """Fetch news articles using the 'everything' endpoint."""
        api_key = os.getenv("news_api")  # Replace with your actual News API key

        # Construct the URL with the 'everything' endpoint
        url = f"https://newsapi.org/v2/everything"

        params = {
            "q": keyword,
            "from": '2024-08-01',  # Set your desired date range
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": api_key,
            "page":1,
            "domains":"bbc.co.uk, techcrunch.com"
        }

        try:
            response = requests.get(url,params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            if data.get("status") == "ok":
                return self._format_articles(data.get("articles"))
            else:
                return {"error": "Failed to retrieve news items."}

        except requests.exceptions.RequestException as e:
            return {"error": f"An error occurred: {e}"}

    def _format_articles(self, articles):
        """Helper method to format the list of articles into a structured JSON response."""
        formatted_articles = []
        for article in articles:
            if not all(key in article for key in ["title", "url", "urlToImage"]):
                continue  # Skip articles missing essential information
            formatted_article = {
                "title": article.get("title"),
                "description": article.get("description", article.get("content")),
                "url": article.get("url"),
                "imageUrl": article.get("urlToImage"),
            }
            formatted_articles.append(formatted_article)
        return {"articles": formatted_articles}



class LangChainAgent:
    def __init__(self, api_key):
        self.article_tool = ArticleExtractorSummarizer(api_key)

        self.news_search_tool = CustomNewsSearchTool()
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2", 
            huggingfacehub_api_token=os.getenv("hf_token")
        )
#         self.llm = ChatGoogleGenerativeAI(
#     model="gemini-1.0-pro",
#     api_key=os.getenv('gemini_key'),
# )
        self.jina_scraper = JinaScraper()

        # Define the tools first
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


        # Updated prompt template
        self.prompt = ChatPromptTemplate.from_template("""
YOU ARE NEWS_FETCH.AI, A HIGHLY INTELLIGENT AND SPECIALIZED BOT DESIGNED TO PROVIDE USERS WITH THE MOST ACCURATE, RELEVANT, AND UP-TO-DATE NEWS BASED ON THEIR QUERIES.


    You have access to the following tools:

    {tools}

    ### IMPORTANT ###
    1. Every News articles you return mus include title ,url to image, url and description of the news article(s).
    2. If the query is broad or requires multiple articles, provide the top 2 relevant news items.

   
    Use the following format:


            Question: {input}
            Thought: Analyze the input to understand the user's request and determine the best approach.
            Action: Choose the appropriate action from {tool_names} and extract the subject for the news search.
            Action Input: Pass the extracted keyword to the `NewsSearch` tool.
            Observation: Record the result of the action.
            Thought: Synthesize the information and prepare the final response.
            Final Answer:the final answer to the original input question considering the important instructions. , The news and its title ,render the images from image links, url should be represented together in a single point formatted manner  and should be clearly seperated from other articles information,Correctly align the articles displaying as well point by point. 

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    
)


        # Create the agent with the updated prompt template
        self.react_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
            
        )

        self.agent_executor = AgentExecutor(
            agent=self.react_agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            
        )

    def run(self, query):
        try:
            # Initialize the scratchpad for thoughts and actions tracking
            result = self.agent_executor.invoke({"input": query})

            if isinstance(result, dict):
                return result
            else:
                return {"output": result}
        except Exception as e:
            return {"error": str(e)}




@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('query', '')
    api_key = os.getenv("extract__key")
    langchain_agent = LangChainAgent(api_key)

    result = langchain_agent.run(question)

    if "output" in result:
        # Directly return the structured JSON response from the agent
        return jsonify(result["output"])
    else:
        return jsonify({"error": result.get("error", "Unknown error occurred")})



if __name__ == '__main__':
    app.run(host='0.0.0.0',port =8000,debug=True)