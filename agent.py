import os
from dotenv import load_dotenv
# import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
# from langchain.agents import AgentType
# from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
import gradio as gr
import uuid
from langchain_openai import ChatOpenAI
# from langchain_anthropic import AnthropicLLM
# from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.history import RunnableWithMessageHistory


load_dotenv()


# langsmith_api_key = os.environ.get("LANGSMITH_API_KEY")
# os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "Local SQL Agent"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
os.environ["OPENAI_API_KEY"] = os.environ.get("OPEN_AI_KEY")

# os.environ["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_KEY")
# os.environ["AUTOGEN_USE_DOCKER"] = "False"



def connectDatabase():
    global db
    mysql_uri = f"mysql+mysqlconnector://root:admin@host.docker.internal:3306/classicmodels"
    db = SQLDatabase.from_uri(mysql_uri)
    

def getDatabase():
    mysql_uri = f"mysql+mysqlconnector://root:admin@host.docker.internal:3306/classicmodels"
    db = SQLDatabase.from_uri(mysql_uri)
    return db
# 

# mysql+mysqlconnector://root:admin@localhost:3306/classicmodels
# def getLLM():
#     llm = ChatOllama(model="llama3.1:8b-instruct-q4_0", timeout=1200) 
#     return llm

# def JustgetLLM():
#     newllm = ChatGroq(model="llama-3.1-70b-versatile")
#     return newllm

def OpenAILLM():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return llm

# def getAnthropicLLM():
#     llm = ChatAnthropic(model="claude-3-5-sonnet-20240620",   temperature=0)
#     return llm
    

def exampleSelector():
    examples = [
    { 
        "input": "List all products in the 'Classic Cars' product line.",
        "query": "SELECT * FROM products WHERE productLine = 'Classic Cars';"
    },
    {
        "input": "Find all orders placed by the customer with ID 103.",
        "query": "SELECT * FROM orders WHERE customerNumber = 103;",
    },
    {
        "input": "List all customers from the USA.",
        "query": "SELECT * FROM customers WHERE country = 'USA';",
    },
    {
        "input": "Find the total amount of payments made by customer 112.",
        "query": "SELECT SUM(amount) FROM payments WHERE customerNumber = 112;",
    },
    {
        "input": "How many products are there in total?",
        "query": "SELECT COUNT(*) FROM products;",
    },
    {
        "input": "List all employees who report to employee number 1143.",
        "query": "SELECT * FROM employees WHERE reportsTo = 1143;",
    },
    {
        "input": "Find the total number of orders.",
        "query": "SELECT COUNT(*) FROM orders;",
    },
    {
        "input": "List all products with a buy price greater than $50.",
        "query": "SELECT * FROM products WHERE buyPrice > 50;",
    },
    {
        "input": "Who are the top 5 customers by credit limit?",
        "query": "SELECT customerNumber, creditLimit FROM customers ORDER BY creditLimit DESC LIMIT 5;",
    },
    {
        "input": "How many employees are there in the 'Sales' department?",
        "query": "SELECT COUNT(*) FROM employees WHERE jobTitle LIKE '%Sales%';",
    },
    ]

    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,
    k=2,
    input_keys=["input"],
    )
    return example_selector


def fewshotsgpt(example_select):
    system_prefix = """You are an agent designed to interact with a SQL database. 
    Given an input question, create a syntactically correct {dialect} query to run, then look at the output of the query and return both the query and a the output in a single formatted string.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.

    Your response should be in the following format:
    - Output: [Output here]

    Rules:
    1. Always double-check the SQL query for correctness before execution.
    2. Do not modify the database (no INSERT, UPDATE, DELETE, DROP commands).
    3. If you encounter an error with the query, attempt to correct it before returning the final response.
    4. Don't display the SQL query.
    5. Provide a clear and concise explanation of the results in natural language.
    6. If the question is not related to the database, respond with "I don't know."
    7- Important: show the output in table format

    Specific Instructions:
    1. **Greetings and Goodbyes**: If the user greets (e.g., "hello", "hi") or says goodbye (e.g., "bye", "goodbye"), respond politely with a greeting or farewell message.
    2. **Database-Related Questions**: Identify if the question is related to the database. Examples include questions about data, structure, or specific queries. Construct and execute the SQL query accordingly.
    3. **Non-Database Questions**: If the question is not related to the database, respond with "I don't know."

    Example Interaction:
    - User Question: "What are the top 5 highest-paid employees?"
    - Output: "Here are the five highest-paid customers listed in descending order of total payments.
    customerName	     totalPayments 	
    Euro+ Shopping Channel 715738.98
    Mini Gifts Distributors Ltd. 584188.24
    Australian Collectors, Co. 180585.07
    Muscle Machine Inc 177913.95
    Dragon Souveniers, Ltd. 156251.03
    "

    Below are some additional examples of user questions and the corresponding SQL queries and responses:"""

    dynamic_few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_select,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect"],
        prefix=system_prefix,
        suffix="When responding, always follow the specified format."
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=dynamic_few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    return full_prompt
   

def insertFewshots(example_select):
    system_prefix = """You are an agent designed to interact with a Microsoft SQL database.
Given an input question, create a syntactically correct MS SQL query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

You have access to the following tools for interacting with the database:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
If you see you are repeating yourself, just provide final answer and exit.

Here are some examples of user inputs and their corresponding SQL queries:"""
    dynamic_few_shot_prompt = FewShotPromptTemplate(
    example_selector = example_select,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input"],
    prefix=system_prefix,
    suffix=""
    )
    
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=dynamic_few_shot_prompt),
             ("human", "{input}"),
             ("system", "{agent_scratchpad}"),
        ]
    )
    return full_prompt

def Prefix():
    MSSQL_AGENT_PREFIX = """

    You are an agent designed to interact with a SQL database.
    ## Instructions:
    - Given an input question, create a syntactically correct {dialect} query
    to run, then look at the results of the query and return the answer.
    - Unless the user specifies a specific number of examples they wish to
    obtain, **ALWAYS** limit your query to at most {top_k} results.
    - You can order the results by a relevant column to return the most
    interesting examples in the database.
    - Never query for all the columns from a specific table, only ask for
    the relevant columns given the question.
    - You have access to tools for interacting with the database.
    - You MUST double check your query before executing it.If you get an error
    while executing a query,rewrite the query and try again.
    - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.)
    to the database.
    - DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
    OF THE CALCULATIONS YOU HAVE DONE.
    - Your response should be in Markdown. However, **when running  a SQL Query
    in "Action Input", do not include the markdown backticks**.
    Those are only for formatting the response, not for executing the command.
    - ALWAYS, as part of your final answer, explain how you got to the answer
    on a section that starts with: "Explanation:". Include the SQL query as
    part of the explanation section.
    - If the question does not seem related to the database, just return
    "I don\'t know" as the answer.
    - Specific Instructions:
        Greetings and Goodbyes**: If the user greets (e.g., "hello", "hi") or says goodbye (e.g., "bye", "goodbye"), respond politely with a greeting or farewell message.
        Database-Related Questions**: Identify if the question is related to the database. Examples include questions about data, structure, or specific queries. Construct and execute the SQL query accordingly.
        Non-Database Questions**: If the question is not related to the database, respond with "I don't know."
    
    - Only use the below tools. Only use the information returned by the
    below tools to construct your query and final answer.
    - Do not make up table names, only use the tables returned by any of the
    tools below.

    ## Tools:

    """

    return MSSQL_AGENT_PREFIX

def Format():
    
    MSSQL_AGENT_FORMAT_INSTRUCTIONS = """

## Use the following format:

Question: the input question you must answer.
Thought: you should always think about what to do.
Action: the action to take, should be one of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.



Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
SELECT TOP (10) customerName, amount
FROM payments
JOIN customers ON payments.customerNumber = customers.customerNumber
WHERE paymentDate LIKE '2023%'
ORDER BY amount DESC;


Observation:
[('Atelier graphique', 54321.00), ('Mini Gifts Distributors Ltd.', 50000.00), ('Diecast Classics Inc.', 45000.00), ('Auto Associés & Cie.', 42000.00), ('Technics Stores Inc.', 40000.00)]
Thought:I now know the final answer
Final Answer: The top 10 customers who made the highest payments in 2023 include "Atelier graphique" with a payment amount of $54,321, followed by "Mini Gifts Distributors Ltd." with $50,000.

Explanation:
I queried the payments table to retrieve the top 10 customers with the highest payment amounts in 2023. I joined the customers table to match customer names with their payment amounts. The query ordered the results by payment amount in descending order to find the customers who made the highest payments. The TOP (10) clause was used to limit the results to the top 10 payments.

```sql
SELECT TOP (10) customerName, amount
FROM payments
JOIN customers ON payments.customerNumber = customers.customerNumber
WHERE paymentDate LIKE '2023%'
ORDER BY amount DESC;```
===> End of Example

"""
    return MSSQL_AGENT_FORMAT_INSTRUCTIONS

        
# def getTools():
#     tools = [QuerySQLDataBaseTool(db = getDatabase()), InfoSQLDatabaseTool(db = getDatabase()), ListSQLDatabaseTool(db = getDatabase()), QuerySQLCheckerTool(db = getDatabase(), llm = getLLM())]
#     return tools


def InsertTools(full_prompt,tools):
    prompt_val = full_prompt.invoke(
        {
            "tool_names" : [tool.name for tool in tools],
            "tools" : [tool.name + " - " + tool.description.strip() for tool in tools],
        }
    )
    return prompt_val


def agentExecutor(llm, db, fullprompt):
    # agent = create_react_agent(llm, tools, fullprompt)
    # toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    
    # agent_executor_SQL = create_sql_agent(
    # prefix=prefix,
    # format_instructions = format,
    # llm=llm,
    # toolkit=toolkit,
    # top_k=30,
    # verbose=True
    # )

    # agent = create_react_agent(llm, tools, fullprompt)

    # agent = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)
    
    # agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose = True)
    # agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True,prompt=fullprompt)
    agent_executor = create_sql_agent(llm=llm,db=db,prompt=fullprompt,verbose=True, agent_type="openai-tools" ,handle_parsing_errors=True)

    return agent_executor

def JustAgentExecutor(llm, db,theprompt):
    # agent_executor= create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True,prompt=theprompt)
    return agent_executor



selector=exampleSelector()
fullprompt=fewshotsgpt(selector)
# fulprompt= insertFewshots(selector)
# the_prompt=JustPrompt()
# tools=getTools()
# llm=JustgetLLM()
gpt=OpenAILLM()
# claude=getAnthropicLLM()
# llama=JustgetLLM()
# claude=AnthropicLLM()
db=getDatabase()
# prefix=Prefix()
# format=Format()
agent_executor=agentExecutor(gpt,db,fullprompt)

def get_session_history(session_id):
    last_k_messages = 4
    chat_message_history = SQLChatMessageHistory(
    session_id=session_id, connection = "sqlite:///memory.db", table_name = "local_table"
    )

    messages = chat_message_history.get_messages()
    chat_message_history.clear()
    
    for message in messages[-last_k_messages:]:
        chat_message_history.add_message(message)
    
    print("chat_message_history ", chat_message_history)
    return chat_message_history



agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
with gr.Blocks() as demo:
    
    state = gr.State("")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])


    def respond(message, chatbot_history, session_id):
        if not chatbot_history:
            session_id = uuid.uuid4().hex

        print("Session ID: ", session_id)

        response = agent_with_chat_history.invoke(
                                        {"input": message},
                                        {"configurable": {"session_id": session_id}},
                                        )

        chatbot_history.append((message, response['output']))
        return "", chatbot_history, session_id

    msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])

demo.launch(server_name="0.0.0.0", server_port=7860)


