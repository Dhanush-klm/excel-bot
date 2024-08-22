import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import tiktoken
from tiktoken import get_encoding
import time
import pandas as pd
import json
import io

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "excel-data"

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create or connect to the Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
index = pc.Index(INDEX_NAME)

# System Instruction for the AI
SYSTEM_INSTRUCTION = """You are an AI assistant provide only the accurate answer answer does not make assumptions to the answer """

def get_sheet(file, sheet_name=None):
    try:
        # Read the Excel file
        if sheet_name:
            df = pd.read_excel(file, sheet_name=sheet_name)
        else:
            # If no sheet_name is provided, read the first sheet
            df = pd.read_excel(file)
        
        # Convert DataFrame to JSON
        json_data = df.to_json(orient='records')
        
        # Parse JSON string to Python object
        data = json.loads(json_data)
        
        return data, pd.ExcelFile(file).sheet_names
    except Exception as e:
        st.error(f"An error occurred while processing the Excel file: {str(e)}")
        return None, None

def extract_text_from_excel(excel_file, sheet_name=None):
    data, sheet_names = get_sheet(excel_file, sheet_name)
    if data is not None:
        return [("Excel Document", json.dumps(data, indent=2))], sheet_names
    return None, None

# Function to truncate text
def truncate_text(text, max_tokens):
    tokenizer = get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return tokenizer.decode(tokens[:max_tokens])

# Function to count tokens
def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to get embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def upsert_to_pinecone(text, source):
    chunks = [text[i:i+8000] for i in range(0, len(text), 8000)]  # Split into 8000 character chunks
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        metadata = {
            "source": source,
            "text": chunk
        }
        vector_id = f"{source}_{i}"
        vectors.append((vector_id, embedding, metadata))
    index.upsert(vectors=vectors)
    time.sleep(1)

# Function to query Pinecone
def query_pinecone(query, top_k=5):
    query_embedding = get_embedding(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    contexts = []
    for match in results['matches']:
        if 'text' in match['metadata']:
            contexts.append(match['metadata']['text'])
        else:
            contexts.append(f"Content from {match['metadata'].get('source', 'unknown source')}")
    return contexts

def get_answer(query):
    contexts = query_pinecone(query)
    context = " ".join(contexts)
    max_context_tokens = 8000
    truncated_context = truncate_text(context, max_context_tokens)
    
    # Debug: Print retrieved context
    st.text("Retrieved context:")
    st.text(truncated_context)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Query: {query}\n\nContext: {truncated_context}"}
        ]
    )
    answer = response.choices[0].message.content.strip()
    
    # Debug: Print generated answer
    st.text("Generated answer:")
    st.text(answer)
    
    return answer

# Streamlit Interface
st.set_page_config(page_title="Excel-Bot", layout="wide")
st.title("Excel-Bot")


# Sidebar for file upload
with st.sidebar:
    st.header("Upload Customer Touchpoint Data")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            texts, sheet_names = extract_text_from_excel(uploaded_file)
            
            if texts is not None:
                if sheet_names and len(sheet_names) > 1:
                    selected_sheet = st.selectbox("Select a sheet", options=["First Sheet"] + sheet_names)
                    if selected_sheet != "First Sheet":
                        texts, _ = extract_text_from_excel(uploaded_file, selected_sheet)
                
                total_token_count = 0
                for source, text in texts:
                    token_count = num_tokens_from_string(text)
                    total_token_count += token_count
                    # Upsert to Pinecone
                    upsert_to_pinecone(text, source)
                    st.text(f"Uploaded: {source}")
                st.subheader("Uploaded Documents")
                st.text(f"Total token count: {total_token_count}")
            else:
                st.warning("Failed to process the uploaded file. Please try again with a different Excel file.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# Main content area
st.header("Ask Your Question")
user_query = st.text_input("What would you like to know about?")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Generating answer..."):
            answer = get_answer(user_query)
            st.subheader("Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question before searching.")
