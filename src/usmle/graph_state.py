"""
LangGraph State Schema for MedQG Question Generation Pipeline

This module defines the state structure for the question generation workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional


class QuestionGenerationState(TypedDict):
    """
    State schema for the question generation graph.

    This state flows through all nodes in the LangGraph pipeline.
    """
    # Input parameters
    clinical_note: str
    keypoint: str
    topic: str
    max_attempts: int

    # Iteration tracking
    n_attempts: int

    # Generated question components
    context: Optional[str]
    question: Optional[str]
    correct_answer: Optional[str]
    distractor_options: Optional[str]

    # Answer generation
    attempted_answer: Optional[str]
    reasoning: Optional[str]

    # Feedback and scores
    context_feedback: Optional[Dict[str, Any]]
    context_score: str
    question_feedback: Optional[Dict[str, Any]]
    question_score: str
    correct_answer_feedback: Optional[Dict[str, Any]]
    correct_answer_score: str
    distractor_option_feedback: Optional[Dict[str, Any]]
    distractor_option_score: str
    reasoning_feedback: Optional[Dict[str, Any]]
    reasoning_score: str

    # Output tracking - accumulates all iterations
    content_to_fb_ret: List[Dict[str, Any]]

    # Control flow
    should_continue: bool
