"""
Prompt template system for RAG.

Provides flexible prompt templates with variable substitution
and multiple strategies for different use cases.
"""

from typing import Dict, Any, Optional, List
from enum import Enum


class PromptStrategy(Enum):
    """Different prompt strategies for various use cases"""
    STRICT = "strict"  # Strict adherence to context
    CONVERSATIONAL = "conversational"  # More natural, conversational
    TECHNICAL = "technical"  # Technical, detailed responses
    SUMMARIZE = "summarize"  # Focus on summarization
    QNA = "qna"  # Question-answering focused


class PromptTemplate:
    """Template for building prompts with variable substitution"""
    
    def __init__(self, template: str, variables: List[str]):
        """
        Initialize prompt template
        
        Args:
            template: Template string with {variable} placeholders
            variables: List of required variable names
        """
        self.template = template
        self.variables = variables
    
    def render(self, **kwargs) -> str:
        """
        Render template with provided variables
        
        Args:
            **kwargs: Variable values to substitute
        
        Returns:
            Rendered prompt string
        """
        # Check all required variables are provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")
        
        return self.template.format(**kwargs)


# System message templates
SYSTEM_TEMPLATES = {
    PromptStrategy.STRICT: """You are a helpful assistant that provides accurate information based on the provided context.

Rules:
- Only use information from the provided context
- If the context doesn't contain enough information, explicitly state that
- When referencing information from the context, cite it using numbered citations like [1], [2], [3] etc.
- The citation number corresponds to the [Context N] label in the provided context
- Do not make up or infer information not in the context""",

    PromptStrategy.CONVERSATIONAL: """You are a friendly and helpful assistant. You answer questions based on the provided context from documents, but you can also have natural conversations.

Guidelines:
- Use the provided context to answer questions accurately
- When referencing specific information, cite it using numbered citations like [1], [2], [3] etc.
- The citation number corresponds to the [Context N] label in the provided context
- Be conversational and engaging
- If context is insufficient, acknowledge it naturally""",

    PromptStrategy.TECHNICAL: """You are a technical assistant that provides detailed, accurate information based on documentation.

Requirements:
- Provide comprehensive, technical answers based on the context
- Include specific details, numbers, and technical terms from the context
- Cite sources using numbered citations like [1], [2], [3] etc. when referencing information
- The citation number corresponds to the [Context N] label in the provided context
- If information is missing, clearly state what's not available in the context""",

    PromptStrategy.SUMMARIZE: """You are an assistant that summarizes information from documents.

Instructions:
- Create concise summaries based on the provided context
- Highlight key points and main ideas
- Maintain accuracy while being brief
- Cite sources using numbered citations like [1], [2], [3] etc. for summarized information
- The citation number corresponds to the [Context N] label in the provided context""",

    PromptStrategy.QNA: """You are a question-answering assistant that provides direct, accurate answers based on documents.

Approach:
- Answer questions directly and concisely
- Use the provided context as the primary source
- Cite sources using numbered citations like [1], [2], [3] etc. for each fact you mention
- The citation number corresponds to the [Context N] label in the provided context
- If the answer isn't in the context, say so clearly"""
}


# User message templates
USER_TEMPLATES = {
    PromptStrategy.STRICT: """Based on the following context from documents, please answer the question.

{context}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- When referencing information, cite it using numbered citations like [1], [2], [3] etc.
- The citation number [N] corresponds to [Context N] in the context above
- If the context doesn't contain the answer, state: "The provided context does not contain enough information to answer this question." """,

    PromptStrategy.CONVERSATIONAL: """Here's some context from documents that might help answer your question:

{context}

Your question: {query}

Please provide a helpful answer based on the context above. When you reference specific information, cite it using numbered citations like [1], [2], [3] etc. The citation number [N] corresponds to [Context N] in the context above. Feel free to be conversational.""",

    PromptStrategy.TECHNICAL: """Technical Context from Documents:

{context}

Technical Question: {query}

Please provide a detailed technical answer based on the context. Include:
- Specific details and technical terms
- Source citations using numbered format [1], [2], [3] etc. when referencing information
- The citation number [N] corresponds to [Context N] in the context above
- Any relevant technical specifications or numbers from the context""",

    PromptStrategy.SUMMARIZE: """Please summarize the following information from documents:

{context}

Question/Topic: {query}

Provide a concise summary that addresses the question/topic. When referencing information, cite it using numbered citations like [1], [2], [3] etc. The citation number [N] corresponds to [Context N] in the context above.""",

    PromptStrategy.QNA: """Context from Documents:

{context}

Question: {query}

Answer the question directly based on the context. When mentioning facts, cite them using numbered citations like [1], [2], [3] etc. The citation number [N] corresponds to [Context N] in the context above."""
}


def get_system_prompt(strategy: PromptStrategy = PromptStrategy.CONVERSATIONAL,
                      use_rag: bool = True) -> str:
    """
    Get system prompt based on strategy
    
    Args:
        strategy: Prompt strategy to use
        use_rag: Whether RAG is enabled
    
    Returns:
        System prompt string
    """
    if not use_rag:
        return "You are a helpful assistant."
    
    return SYSTEM_TEMPLATES.get(strategy, SYSTEM_TEMPLATES[PromptStrategy.CONVERSATIONAL])


def get_user_prompt(query: str, context: str, 
                   strategy: PromptStrategy = PromptStrategy.CONVERSATIONAL) -> str:
    """
    Get user prompt with context based on strategy
    
    Args:
        query: User's question
        context: Retrieved context from RAG
        strategy: Prompt strategy to use
    
    Returns:
        User prompt string
    """
    if not context:
        return query
    
    template_str = USER_TEMPLATES.get(strategy, USER_TEMPLATES[PromptStrategy.CONVERSATIONAL])
    template = PromptTemplate(template_str, ["query", "context"])
    
    return template.render(query=query, context=context)


def build_rag_prompt(query: str, context: str, 
                    strategy: PromptStrategy = PromptStrategy.CONVERSATIONAL,
                    use_rag: bool = True):
    """
    Build complete prompt (system + user) for RAG
    
    Args:
        query: User's question
        context: Retrieved context from RAG
        strategy: Prompt strategy to use
        use_rag: Whether RAG is enabled
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    system_prompt = get_system_prompt(strategy, use_rag)
    user_prompt = get_user_prompt(query, context, strategy) if use_rag else query
    
    return system_prompt, user_prompt


# Custom template builder
def create_custom_template(template_str: str, variables: List[str]) -> PromptTemplate:
    """
    Create a custom prompt template
    
    Args:
        template_str: Template string with {variable} placeholders
        variables: List of variable names used in template
    
    Returns:
        PromptTemplate instance
    """
    return PromptTemplate(template_str, variables)

