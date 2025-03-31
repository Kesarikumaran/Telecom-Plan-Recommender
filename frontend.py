import streamlit as st
from backend import TelecomServiceAgent  # Importing your backend class

# Initialize the telecom service agent
agent = TelecomServiceAgent()

# Streamlit UI setup
st.set_page_config(page_title="Plan Recommender Chatbot", layout="centered")

st.title("💬 Plan Recommender Chatbot")
st.markdown(
    "<p style='color: orange; font-size:15px;'>"
    "This AI-powered chatbot helps users find the best telecom plans based on their needs. "
    "It analyzes user queries and provides personalized plan recommendations using advanced "
    "Generative AI techniques. Simply type your requirements, and the chatbot will suggest "
    "the most suitable telecom plans for you!"
    "</p>",
    unsafe_allow_html=True
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history with proper formatting
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict) and "output" in message["content"]:
            st.markdown(message["content"]["output"])  # Extract and display only the "output"
        else:
            st.markdown(message["content"])  # Display normal messages

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the backend to process the query
    bot_response = agent.process_query(user_input)

    # Extract bot response properly
    if isinstance(bot_response, dict) and "output" in bot_response:
        bot_message = bot_response["output"]
    else:
        bot_message = "Sorry, I couldn't process your request."

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(bot_message)

    # Store formatted bot response in session state
    st.session_state["messages"].append({"role": "assistant", "content": bot_message})  # Ensure it's a string
