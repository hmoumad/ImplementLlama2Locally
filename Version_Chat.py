from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp

import streamlit as st
from langchain_community.llms import LlamaCpp

MODEL_PATH = "C:/Users/hamza/Desktop/TestModel/Model/llama-2-7b-chat.Q2_K.gguf"

# Chargement du modèle LlamaCpp
def load_model():
    llama_model = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_tokens=2000,
        top_p=1,
        verbose=True
    )
    return llama_model

# Fonction pour obtenir la réponse du modèle à partir de la question donnée
def get_response(question):
    model = load_model()
    response = model(question)
    return response

def main():
    st.title("Chatbot Llama-2 Locally")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Si l'utilisateur a posé une question
    if user_question := st.chat_input("Ask me Question about your PDF File 📖"):   

         # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "message": user_question})

        # Generate response from the chatbot
        bot_response = get_response(user_question)
        st.session_state.chat_history.append({"role": "bot", "message": bot_response})

    # Display chat history
    for item in st.session_state.chat_history:
        if item["role"] == "user":
            with st.chat_message("user"):
                st.markdown(user_question)
        else:
            # st.write("Bot: ", item["message"])
            with st.chat_message("assistant"):
                st.write(bot_response)

if __name__ == "__main__":
    main()
