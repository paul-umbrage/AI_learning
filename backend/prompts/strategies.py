"""
Prompt strategies for different use cases.

This module provides predefined strategies and utilities for
selecting and managing prompt strategies.
"""

from typing import Dict, Any, Optional
from prompts.templates import PromptStrategy, build_rag_prompt


# Strategy descriptions
STRATEGY_DESCRIPTIONS = {
    PromptStrategy.STRICT: {
        "name": "Strict",
        "description": "Strictly adheres to context, no inference",
        "use_case": "When accuracy is critical and you want to avoid hallucinations"
    },
    PromptStrategy.CONVERSATIONAL: {
        "name": "Conversational",
        "description": "Natural, friendly responses with context",
        "use_case": "General purpose chat with document context"
    },
    PromptStrategy.TECHNICAL: {
        "name": "Technical",
        "description": "Detailed technical responses with specifications",
        "use_case": "Technical documentation, specifications, detailed answers"
    },
    PromptStrategy.SUMMARIZE: {
        "name": "Summarize",
        "description": "Focused on creating concise summaries",
        "use_case": "When you need brief overviews or summaries"
    },
    PromptStrategy.QNA: {
        "name": "Q&A",
        "description": "Direct question-answering format",
        "use_case": "FAQ-style questions, direct answers"
    }
}


def get_strategy_description(strategy: PromptStrategy) -> Dict[str, str]:
    """Get description for a prompt strategy"""
    return STRATEGY_DESCRIPTIONS.get(strategy, {})


def list_strategies() -> Dict[str, Dict[str, str]]:
    """List all available strategies with descriptions"""
    return {
        strategy.value: get_strategy_description(strategy)
        for strategy in PromptStrategy
    }


def select_strategy(query: str, context_type: Optional[str] = None) -> PromptStrategy:
    """
    Automatically select strategy based on query characteristics
    
    Args:
        query: User query
        context_type: Optional hint about context type (e.g., "technical", "general")
    
    Returns:
        Recommended PromptStrategy
    """
    query_lower = query.lower()
    
    # Technical indicators
    technical_keywords = ["how does", "implementation", "specification", "architecture", 
                         "algorithm", "code", "technical", "design"]
    if any(kw in query_lower for kw in technical_keywords) or context_type == "technical":
        return PromptStrategy.TECHNICAL
    
    # Summarization indicators
    summary_keywords = ["summarize", "overview", "summary", "brief", "key points"]
    if any(kw in query_lower for kw in summary_keywords):
        return PromptStrategy.SUMMARIZE
    
    # Q&A indicators
    qna_keywords = ["what is", "who is", "when", "where", "which", "define"]
    if any(kw in query_lower for kw in qna_keywords):
        return PromptStrategy.QNA
    
    # Default to conversational
    return PromptStrategy.CONVERSATIONAL


# Strategy presets for common scenarios
STRATEGY_PRESETS = {
    "default": PromptStrategy.CONVERSATIONAL,
    "accurate": PromptStrategy.STRICT,
    "detailed": PromptStrategy.TECHNICAL,
    "brief": PromptStrategy.SUMMARIZE,
    "faq": PromptStrategy.QNA
}


def get_strategy_from_preset(preset: str) -> PromptStrategy:
    """Get strategy from preset name"""
    return STRATEGY_PRESETS.get(preset.lower(), PromptStrategy.CONVERSATIONAL)

