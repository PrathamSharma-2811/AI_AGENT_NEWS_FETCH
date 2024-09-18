import requests
from bs4 import BeautifulSoup
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from flask import Flask, request, jsonify
from datetime import datetime, timedelta





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
    description = "Fetches news articles according to parameters passed to newsapi.org api."

    def _run(self, keyword: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> dict:
        """Fetch news articles based on the keyword provided in api parameters"""
        api_key = os.getenv("news_api")

        # Construct the URL with the 'everything' endpoint
        url = f"https://newsapi.org/v2/everything"

        # Dynamically calculate the date range (past 7 days)
        today = datetime.today().strftime('%Y-%m-%d')
        last_week = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')

        params = {
            "q": keyword,
            "from": last_week,  # 7 days ago
            "to": today,  # Today's date
            "sortBy": "publishedAt",  # Sort by publication date
            "language": "en",
            "apiKey": api_key,
            "page": 1,
            "domains": "ndtv.com,bbc.com,hindustantimes.com,news.un.org"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()

            if data.get("status") == "ok":
                
                return  data['articles']
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

        self.news_search = CustomNewsSearchTool()

        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
            huggingfacehub_api_token=os.getenv("hf_token")
        )
#         self.llm = ChatGoogleGenerativeAI(
#     model="gemini-1.0",
#     api_key=os.getenv('gemini_key'),
# )
        self.jina_scraper = JinaScraper()

        # Define the tools first
        self.tools = [
            Tool(
                 name="news_search._run",
                func=lambda keyword: self.news_search._run(keyword=keyword),
                description="Always use this to Fetch news articles according to query you provide.the query should be a subject not a statement. It returns  articles including title, image URL, clickable link, and a brief description."
            ),
             Tool(
                name="jinaai reader",
                func=self.jina_scraper.jinaai_readerapi_web_scrape_url,
                description="Reads data from a URL using Jina's web scraping API."
            ),
           
           
           
        ]

        

        # Updated prompt template

        self.prompt= ChatPromptTemplate.from_template("""


The First thing should be to Greet users !

Answer the following questions as best you can, but speaking as NEWS_FETCH.AI, A HIGHLY INTELLIGENT AND SPECIALIZED BOT DESIGNED TO PROVIDE USERS WITH THE MOST ACCURATE, RELEVANT, AND UP-TO-DATE NEWS BASED ON THEIR QUERIES . You have access to the following tools:

{tools}
                                                      
Tool 1: news search  
Use this to Fetch news articles according to the query provided.the query should be one subject not a statement. It provides articles with titles, image URLs, clickable links, and a brief description.It will return for sure be patient .

Tool 2: jinaai reader 
Reads data from a URL using Jina's web scraping API.

                                                                                                          
When you get Output from the news search tool then The news and its title ,render the images from image links, url should be represented together in formatted manner like below:
[
Title: ,
Description: ,
URL: ,
Image_URL:
 ]  
                                                      
IF user ask you summarize an article by providing link use tools and return asnwer in a well explained manner.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question .  

Begin! Remember to answer as a compansionate AI News Fetch when giving your final answer and do not give out of context answers and greet everyone!.

Question: {input}
{agent_scratchpad}
 """)

        # Create the agent with the updated prompt template
        self.react_agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt,
            stop_sequence=["\nObservation:"],
            
            
        )

        self.agent_executor = AgentExecutor(
            agent=self.react_agent,
            tools=self.tools,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True
            
            
        )
        

    def run(self, query):
        try:
            # Initialize the scratchpad for thoughts and actions tracking
            result = self.agent_executor.invoke({"input": query,"tools":self.tools,"agent_scratchpad":""})

            if isinstance(result, dict):
                return result
            else:
                return {"output": result}
        except Exception as e:
            return {"error": str(e)}
