from langchain.agents import create_react_agent, AgentExecutor
from langchain.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_experimental.tools import PythonREPLTool
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
# from langchain.memory import ConversationBufferMemory

# dotenv loading
from dotenv import load_dotenv
import os
load_dotenv()

# API key loading
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

class TelecomServiceAgent:
    """A class to handle telecom service recommendations using a generative AI model."""
    def __init__(self):
        """Initialize the service agent with required components."""
        # self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-2.0-flash", api_key=api_key)
        self.tools = self._create_tools()
        self.prompt = self._create_prompt()
        self.agent_executor = self._create_agent_executor()
 
    def _create_tools(self):
        """Create and return the list of tools."""
        db = SQLDatabase.from_uri("sqlite:///telecom.db")
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)
        tools = toolkit.get_tools()
        tools.append(PythonREPLTool())
        return tools
 
    def _create_prompt(self):
        """Create and return the prompt template."""
        template = """You are a telecom service advisor who helps customers find the best plan for their needs.
 
        When recommending plans, consider:
        1. The customer's usage patterns (data, voice, SMS)
        2. Number of people/devices that will use the plan
        3. Special requirements (international calling, streaming, etc.)
        4. Budget constraints
 
        Always explain WHY a particular plan is a good fit for their needs.
 
        If the customer asks for a plan that is not available, suggest the closest alternative.
 
        You have access to the following tools:
 
        {tools}
 
        Use the following format:
 
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
 
        Question : {input}
        Thought : {agent_scratchpad}
        """
 
        return PromptTemplate.from_template(template=template).partial(
            tools=render_text_description(self.tools),
            tool_names=", ".join([t.name for t in self.tools]),
            # chat_history=self.memory.load_memory_variables({}).get("chat_history", "") 
        )
 
    def _create_agent_executor(self):
        """Create and return the agent executor."""
        agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=20,
            # memory=self.memory 
        )
 
    def process_query(self, query):
        """Process a service recommendation query"""
        try:
            agent_output = self.agent_executor.invoke({
                "input": query,
                "agent_scratchpad": []
            })
            return agent_output
        except Exception as e:
            return f"Error processing your query: {str(e)}"
