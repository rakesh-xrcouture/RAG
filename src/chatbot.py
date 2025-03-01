""" # from langchain_groq import ChatGroq
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.prompts import PromptTemplate
# from config import config
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ChatBot:
#     def __init__(self, vectorstore):
#         # Initialize ChatGroq with the proper configuration
#         self.llm = ChatGroq(
#             api_key=config.GROQ_API_KEY,
#             model_name="llama-3.3-70b-versatile",
#             streaming=False,
#             temperature=0.7
#         )
        
#         # Initialize conversation memory
#         self.memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             return_messages=True
#         )
        
#         # Define custom prompts
#         self.condense_prompt = self._get_condense_prompt()
#         self.qa_prompt = self._get_qa_prompt()
        
#         # Create the conversational chain
#         self.chain = ConversationalRetrievalChain.from_llm(
#             llm=self.llm,
#             retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
#             memory=self.memory,
#             condense_question_prompt=self.condense_prompt,
#             combine_docs_chain_kwargs={"prompt": self.qa_prompt},
#             verbose=False  # Disable verbose to avoid clutter
#         )
    
#     def _get_condense_prompt(self):
#         ""Custom prompt for condensing follow-up questions.""
#         return PromptTemplate(
#             input_variables=["chat_history", "question"],
#             template=(
#                 "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. "
#                 "If the follow-up question is a greeting or casual, respond naturally.\n\n"
#                 "Chat History:\n{chat_history}\n\n"
#                 "Follow Up Input: {question}\n"
#                 "Standalone question:"
#             )
#         )
    
#     def _get_qa_prompt(self):
#         ""Custom prompt for generating answers.""
#         return PromptTemplate(
#             input_variables=["context", "question"],
#             template=(
#                 "You are a knowledgeable and friendly real estate sales agent for Jain's Aadhya. "
#                 "Use the following pieces of context to answer the user's question. "
#                 "If the context is not relevant to the question, politely inform the user that you don't have the information. "
#                 "Do not make up answers.\n\n"
#                 "Context:\n{context}\n\n"
#                 "Question: {question}\n"
#                 "Answer:"
#             )
#         )
    
#     def chat(self, query: str) -> str:
#         ""Handle user queries and return the chatbot's response.""
#         if query.lower() in ["exit", "quit"]:
#             return "Goodbye!"
        
#         try:
#             response = self.chain.invoke({"question": query})
#             return response["answer"]
#         except Exception as e:
#             logger.error(f"Error during chat: {str(e)}")
#             return "Sorry, I encountered an error while processing your request. Please try again."
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from config import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, vectorstore):
        # Initialize ChatGroq with the proper configuration
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            streaming=False,
            temperature=0.7
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Define custom prompts
        self.condense_prompt = self._get_condense_prompt()
        self.qa_prompt = self._get_qa_prompt()
        
        # Create the conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            condense_question_prompt=self.condense_prompt,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            verbose=False  # Disable verbose to avoid clutter
        )
    
    def _get_condense_prompt(self):
        ""Custom prompt for condensing follow-up questions.""
        return PromptTemplate(
            input_variables=["chat_history", "question"],
            template=(
                "Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. "
                "If the follow-up question is a greeting or casual, respond naturally.\n\n"
                "Chat History:\n{chat_history}\n\n"
                "Follow Up Input: {question}\n"
                "Standalone question:"
            )
        )
    
    def _get_qa_prompt(self):
        ""Custom prompt for generating answers.""
        return PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a knowledgeable and friendly real estate sales agent for Jain's Aadhya. "
                "Use the following pieces of context to answer the user's question. "
                "If the context is not relevant to the question, politely inform the user that you don't have the information. "
                "Do not make up answers.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        )
    
    def chat(self, query: str) -> str:
        ""Handle user queries and return the chatbot's response.""
        if query.lower() in ["exit", "quit"]:
            return "Goodbye!"
        
        try:
            # Pass the query directly to the chain (no manual embedding needed)
            response = self.chain.invoke({"question": query})
            return response["answer"]
        except Exception as e:
            logger.error(f"Error during chat: {str(e)}")
            return "Sorry, I encountered an error while processing your request. Please try again."
        
"""


from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from src.config import config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, vectorstore):
        # Initialize ChatGroq with adjusted parameters
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            streaming=False,
            temperature=0.5  # Reduced temperature for more focused responses
        )
        
        # Initialize conversation memory with summary
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )
        
        # Define custom prompts
        self.condense_prompt = self._get_condense_prompt()
        self.qa_prompt = self._get_qa_prompt()
        
        # Create the conversational chain with improved retrieval
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={
                    "k": 3  # Reduced number of documents for more focused context
                    #"score_threshold": 0.7,  # Only include relevant documents
                    #"fetch_k": 10  # Fetch more candidates initially for better filtering
                }
            ),
            memory=self.memory,
            condense_question_prompt=self.condense_prompt,
            combine_docs_chain_kwargs={
                "prompt": self.qa_prompt,
                "document_separator": "\n----------------------\n"  # Better document separation
            },
            return_source_documents=True,  # Helpful for debugging
            verbose=False
        )
    
    def _get_condense_prompt(self):
        """Enhanced prompt for better question understanding."""
        return PromptTemplate(
            input_variables=["chat_history", "question"],
            template=(
                "Given the following conversation and a follow-up question, rephrase the follow-up question "
                "to be a standalone question that captures the full context of the conversation. "
                "Focus on real estate related aspects if present. " "try to sell the property to the user. " "understand the users needs and preferences. "
                "If the question is conversational or a greeting, keep it natural.\n\n"
                "Chat History:\n{chat_history}\n\n"
                "Follow Up Input: {question}\n"
                "Standalone question:"
            )
        )
    
    def _get_qa_prompt(self):
        """Enhanced prompt for more relevant and accurate answers."""
        return PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a knowledgeable and friendly real estate sales agent named SIA for Jain's Aadhya. "
                "Answer the question based primarily on the provided context. "
                "If the context doesn't contain enough relevant information, say so clearly. " "just answer the question based on the context provided. "
                "If the question is about real estate but the context doesn't help, provide general guidance "
                "while noting that specific details about Jain's Aadhya properties may be limited.\n\n"
                "When discussing properties:\n"
                "1. Prioritize accuracy over completeness\n"
                "2. Stay focused on the specific question asked\n\n"
                "3. Dont talk about the properties that are not available\n"
                "4. Talk only about Jain's Aadhya properties\n\n"
                "5. Keep the woord limit to 50 - 100 words\n\n"
                "6. Ask meaningful question to understand the user's needs\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer:"
            )
        )
    
    def chat(self, query: str) -> str:
        """Enhanced chat handling with better error management and response filtering."""
        if query.lower() in ["exit", "quit", "bye", "goodbye"]:
            return "Thank you for your interest in Jain's Aadhya! Have a great day!"
        
        try:
            # Check for empty or too short queries
            if not query.strip() or len(query.strip()) < 2:
                return "I didn't catch that. Could you please rephrase your question?"
            
            # Process the query
            response = self.chain.invoke({
                "question": query,
                "chat_history": self.memory.chat_memory.messages
            })
            
            # Log retrieved documents for debugging
            logger.debug(f"Retrieved documents: {response.get('source_documents', [])}")
            
            # Return the answer or a fallback response
            answer = response.get("answer", "").strip()
            if not answer:
                return "I apologize, but I couldn't find relevant information to answer your question. Could you please rephrase or ask about something specific about our properties?"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error during chat: {str(e)}", exc_info=True)
            return ("I apologize, but I encountered an issue while processing your request. "
                   "Please try asking your question again, or contact our support team for assistance.")