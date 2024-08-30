from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import ollama
template="""
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
"""

model = OllamaLLM(model='phi3:latest')

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context=""
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        result = chain.invoke({"context":context, "question":user_input})
        print("Bot: ", result)
        context += f"\nUser: {user_input}\nAI: {result}"

if __name__ == '__main__':
    handle_conversation()

# emdb =ollama.embeddings(
#     model = "mxbai-embed-large",
#     prompt= 'What a beatuiful day'
# )
#
# print(len(emdb['embedding']))
# print(emdb)