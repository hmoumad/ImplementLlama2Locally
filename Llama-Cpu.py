from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp

MODEL_PATH = "C:/Users/hamza/Desktop/TestModel/Model/llama-2-7b-chat.Q2_K.gguf"

# 1. Create function to load Llama Model
def Load_Mdel():

    # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    Llama_Model = LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        max_tokens=2000,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
    )

    return Llama_Model


llm = Load_Mdel()

prompt = """
Question: what is the famous food and drink in morocco give me one food and one drink ??
"""
response = llm(prompt)