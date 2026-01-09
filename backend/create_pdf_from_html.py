"""
Create PDF from HTML using system tools (works on macOS)
"""
import subprocess
import os
import tempfile

def create_pdf_from_html(output_path):
    """Create PDF from HTML content using system tools"""
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Introduction to Artificial Intelligence</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #333;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        .page-break {
            page-break-before: always;
        }
    </style>
</head>
<body>
    <h1>Introduction to Artificial Intelligence</h1>
    
    <h2>What is Artificial Intelligence?</h2>
    <p>Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.</p>
    
    <p>Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions or predictions.</p>
    
    <p>Deep Learning is a further subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields such as image recognition, natural language processing, and speech recognition.</p>
    
    <h2>Key Concepts:</h2>
    <ol>
        <li><strong>Supervised Learning:</strong> The algorithm learns from labeled training data, making predictions or decisions based on input-output pairs.</li>
        <li><strong>Unsupervised Learning:</strong> The algorithm finds hidden patterns in data without labeled examples, such as clustering or dimensionality reduction.</li>
        <li><strong>Reinforcement Learning:</strong> An agent learns to make decisions by interacting with an environment and receiving rewards or penalties for its actions.</li>
    </ol>
    
    <div class="page-break"></div>
    <h1>Natural Language Processing</h1>
    
    <p>Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It enables machines to understand, interpret, and generate human language in a valuable way.</p>
    
    <h2>Key NLP Tasks:</h2>
    <ul>
        <li><strong>Text Classification:</strong> Categorizing text into predefined classes</li>
        <li><strong>Named Entity Recognition:</strong> Identifying entities like names, locations, dates</li>
        <li><strong>Sentiment Analysis:</strong> Determining the emotional tone of text</li>
        <li><strong>Machine Translation:</strong> Translating text from one language to another</li>
        <li><strong>Question Answering:</strong> Answering questions based on context</li>
    </ul>
    
    <p>Modern NLP uses transformer architectures, such as BERT and GPT, which have achieved remarkable results in understanding and generating human language. These models are trained on vast amounts of text data and can perform a wide variety of language tasks.</p>
    
    <p>Embeddings are vector representations of words or sentences that capture semantic meaning. They allow machines to understand relationships between words and concepts, enabling similarity searches and context understanding.</p>
    
    <div class="page-break"></div>
    <h1>Retrieval-Augmented Generation (RAG)</h1>
    
    <p>Retrieval-Augmented Generation, or RAG, is a technique that combines information retrieval with generative AI models. It enhances the capabilities of language models by providing them with relevant context from external knowledge sources.</p>
    
    <h2>How RAG Works:</h2>
    <ol>
        <li><strong>Document Ingestion:</strong> Documents are processed, chunked, and converted into embeddings that are stored in a vector database.</li>
        <li><strong>Query Processing:</strong> When a user asks a question, the query is converted into an embedding and used to search for similar document chunks.</li>
        <li><strong>Context Retrieval:</strong> The most relevant chunks are retrieved from the vector database based on semantic similarity.</li>
        <li><strong>Response Generation:</strong> The retrieved context is provided to the language model along with the user's question, enabling it to generate accurate and contextually relevant responses.</li>
    </ol>
    
    <h2>Benefits of RAG:</h2>
    <ul>
        <li><strong>Up-to-date information:</strong> Can access current information not in training data</li>
        <li><strong>Source citations:</strong> Can cite specific documents and pages</li>
        <li><strong>Reduced hallucinations:</strong> Grounded in actual documents</li>
        <li><strong>Domain-specific knowledge:</strong> Can be customized for specific domains</li>
    </ul>
    
    <p>This approach is particularly useful for building chatbots that can answer questions about specific documents, such as research papers, manuals, or company knowledge bases.</p>
</body>
</html>"""
    
    # Create temporary HTML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        html_path = f.name
        f.write(html_content)
    
    try:
        # Try using macOS's built-in tools
        # Method 1: Try using cupsfilter (if available)
        try:
            result = subprocess.run(
                ['cupsfilter', html_path],
                capture_output=True,
                check=True
            )
            with open(output_path, 'wb') as f:
                f.write(result.stdout)
            print(f"✅ Created PDF using cupsfilter: {output_path}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Method 2: Use Python's weasyprint if available
        try:
            import weasyprint
            weasyprint.HTML(filename=html_path).write_pdf(output_path)
            print(f"✅ Created PDF using weasyprint: {output_path}")
            return True
        except ImportError:
            pass
        
        # Method 3: Provide instructions for manual conversion
        print("⚠️  Could not automatically create PDF.")
        print(f"   HTML file created at: {html_path}")
        print(f"   Please open this file in a browser and print to PDF:")
        print(f"   - Open: {html_path}")
        print(f"   - File > Print > Save as PDF")
        print(f"   - Save to: {output_path}")
        return False
        
    finally:
        # Keep HTML file for manual conversion
        pass

if __name__ == "__main__":
    import sys
    output_path = sys.argv[1] if len(sys.argv) > 1 else "../documents/example_ai_learning.pdf"
    create_pdf_from_html(output_path)

