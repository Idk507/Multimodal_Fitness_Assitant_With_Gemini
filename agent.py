import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Please set HUGGINGFACEHUB_API_TOKEN in your environment variables.")

# Import LLM and agent components from LangChain
from langchain.llms import HuggingFaceHub
from langchain.agents import initialize_agent, Tool
from langchain.tools import DuckDuckGoSearchResults, RequestsGetTool

# Initialize an LLM using a HuggingFaceHub model (for example, FLAN-T5-XL)
llm = HuggingFaceHub(
    model="google/flan-t5-xl", 
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Set up available tools.
# The DuckDuckGoSearchResults tool allows the agent to search for live product or hotel data.
duck_tool = DuckDuckGoSearchResults()
# The RequestsGetTool fetches webpage content that can be rephrased.
requests_tool = RequestsGetTool()

tools = [
    Tool(
        name="DuckDuckGo_Search",
        func=duck_tool.run,
        description=(
            "Useful for searching the web for product recommendations, live protein supplements, "
            "diet food hotels, and related nutritional or booking information."
        )
    ),
    Tool(
        name="Requests_Get",
        func=requests_tool.run,
        description=(
            "Use this tool to retrieve detailed webpage content given a URL. "
            "Helpful for extracting product details, prices, links, and reviews."
        )
    )
]

# Initialize the agent with the tools and the LLM.
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True
)

def main():
    print("Welcome to the AI-Driven Fitness & Product Recommendation Agent!")
    print("You can ask for workout advice, protein supplement recommendations, diet hotel suggestions, etc.")
    print("The agent will use live web data to respond with current product details and links.\n")
    
    while True:
        query = input("Enter your query (or 'q' to quit): ").strip()
        if query.lower() == "q":
            print("Exiting the agent. Goodbye!")
            break

        try:
            # The agent uses its integrated tools to search the web and then rephrase the result.
            response = agent.run(query)
            print("\nAgent Response:")
            print(response)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
