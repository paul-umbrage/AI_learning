"""
Simple PDF creation using fpdf (lighter weight alternative)
"""
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

def create_pdf_simple(output_path):
    """Create PDF using fpdf"""
    global HAS_FPDF
    if not HAS_FPDF:
        print("Installing fpdf2...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "fpdf2", "-q"], 
                                 stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            from fpdf import FPDF
            HAS_FPDF = True
        except:
            print("Could not install fpdf2. Trying alternative method...")
            return False
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Introduction to Artificial Intelligence", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "What is Artificial Intelligence?", ln=True)
    pdf.ln(3)
    
    pdf.set_font("Arial", "", 11)
    text1 = """Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions or predictions.

Deep Learning is a further subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields such as image recognition, natural language processing, and speech recognition.

Key Concepts:

1. Supervised Learning: The algorithm learns from labeled training data, making predictions or decisions based on input-output pairs.

2. Unsupervised Learning: The algorithm finds hidden patterns in data without labeled examples, such as clustering or dimensionality reduction.

3. Reinforcement Learning: An agent learns to make decisions by interacting with an environment and receiving rewards or penalties for its actions."""
    
    pdf.multi_cell(0, 6, text1)
    
    # Page 2
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Natural Language Processing", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 11)
    text2 = """Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a valuable way.

Key NLP Tasks:

- Text Classification: Categorizing text into predefined classes
- Named Entity Recognition: Identifying entities like names, locations, dates
- Sentiment Analysis: Determining the emotional tone of text
- Machine Translation: Translating text from one language to another
- Question Answering: Answering questions based on context

Modern NLP uses transformer architectures, such as BERT and GPT, which have achieved remarkable results in understanding and generating human language. These models are trained on vast amounts of text data and can perform a wide variety of language tasks.

Embeddings are vector representations of words or sentences that capture semantic meaning. They allow machines to understand relationships between words and concepts, enabling similarity searches and context understanding."""
    
    pdf.multi_cell(0, 6, text2)
    
    # Page 3
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Retrieval-Augmented Generation (RAG)", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 11)
    text3 = """Retrieval-Augmented Generation, or RAG, is a technique that combines information retrieval with generative AI models. It enhances the capabilities of language models by providing them with relevant context from external knowledge sources.

How RAG Works:

1. Document Ingestion: Documents are processed, chunked, and converted into embeddings that are stored in a vector database.

2. Query Processing: When a user asks a question, the query is converted into an embedding and used to search for similar document chunks.

3. Context Retrieval: The most relevant chunks are retrieved from the vector database based on semantic similarity.

4. Response Generation: The retrieved context is provided to the language model along with the user's question, enabling it to generate accurate and contextually relevant responses.

Benefits of RAG:

- Up-to-date information: Can access current information not in training data
- Source citations: Can cite specific documents and pages
- Reduced hallucinations: Grounded in actual documents
- Domain-specific knowledge: Can be customized for specific domains

This approach is particularly useful for building chatbots that can answer questions about specific documents, such as research papers, manuals, or company knowledge bases."""
    
    pdf.multi_cell(0, 6, text3)
    
    pdf.output(output_path)
    print(f"✅ Created PDF: {output_path}")
    return True

if __name__ == "__main__":
    import sys
    output_path = sys.argv[1] if len(sys.argv) > 1 else "../documents/example_ai_learning.pdf"
    create_pdf_simple(output_path)

