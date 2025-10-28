"""
LangGraph-based End-to-End Pipeline for MedQG

This module implements the complete MedQG workflow using LangGraph:
1. Topic Generation
2. Keypoint Generation
3. Question Generation with Iterative Refinement
"""

import json
import os
from typing import TypedDict, List, Dict, Any, Optional

from langgraph.graph import StateGraph, END
try:
    from langchain import PromptTemplate, FewShotPromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate, FewShotPromptTemplate
try:
    from langchain.chat_models import ChatOpenAI
except ImportError:
    from langchain_openai import ChatOpenAI
try:
    from langchain.chains import LLMChain
except ImportError:
    from langchain.chains.llm import LLMChain
import requests

from src.usmle.question_graph import QuestionGenerationGraph
from src.utils import retry_parse_fail_prone_cmd
from dotenv import load_dotenv

load_dotenv(dotenv_path='usmle.env')

OPENAIKEY = os.getenv("OPENAIKEY")
OPENAIORG = os.getenv("OPENAIORG")
COLBERT_API = os.getenv("COLBERT_API")
ENGINE = os.getenv("ENGINE") or "gpt-4"


class PipelineState(TypedDict):
    """
    State schema for the complete MedQG pipeline.
    """
    # Input
    clinical_note: str
    max_attempts: int

    # Topic generation
    topic_examples_path: str
    topic_list: Optional[List[str]]

    # Keypoint generation
    keypoint_examples_path: str
    current_topic: Optional[str]
    retrieved_kp_score: Optional[float]
    retrieved_keypoints: Optional[str]
    keypoint: Optional[str]

    # Question generation
    generated_questions: List[Dict[str, Any]]

    # Control
    topics_to_process: List[str]
    current_topic_index: int


class MedQGPipeline:
    """
    Complete MedQG pipeline using LangGraph.

    This orchestrates:
    1. Topic generation from clinical notes
    2. Keypoint extraction for each topic
    3. Question generation with iterative refinement
    """

    # USMLE topics list
    USMLE_TOPICS = [
        "the cause/infectious agent or predisposing factor(s)",
        "underlying processes/pathways",
        "underlying anatomic structure or physical location",
        "mechanisms, drugs",
        "knows signs/symptoms of selected disorders",
        "knows individual's risk factors for development of condition",
        "knows what to ask to obtain pertinent additional history",
        "predicts the most likely additional physical finding",
        "select most appropriate laboratory or diagnostic study",
        "interprets laboratory or other study findings",
        "predicts the most likely laboratory or diagnostic study result",
        "most appropriate laboratory or diagnostic study after change in patient status",
        "select most likely diagnosis",
        "recognizes factors in the history, or physical or laboratory study findings",
        "interprets laboratory or other diagnostic study results and identifies current/future status of patient",
        "recognizes associated conditions of a disease",
        "recognizes characteristics of disease relating to natural history or course of disease",
        "risk factors for conditions amenable to prevention or detection",
        "identifies patient groups at risk",
        "knows common screening tests",
        "selects appropriate preventive agent or technique",
        "knows appropriate counseling regarding current and future problems",
        "educates patients",
        "selects most appropriate pharmacotherapy",
        "assesses patient adherence, recognizes techniques to increase adherence",
        "recognizes factors that alter drug requirements",
        "Knows adverse effects of various drugs or recognizes signs and symptoms of drug (and drug-drug) interactions",
        "knows contraindications of various medications",
        "knows modifications of a therapeutic regimen within the context of continuing care",
        "appropriate monitoring to evaluate effectiveness of pharmacotherapy or adverse effects",
        "most appropriate management of selected conditions",
        "immediate management or priority in management",
        "follow-up or monitoring approach regarding the management plan",
        "current/short-term management",
        "severity of patient condition in terms of need for referral for surgical treatments/procedures",
        "appropriate surgical management",
        "preoperative/postoperative",
        "Selecting Clinical Interventions (Mixed Management)",
        "indications for surveillance for recurrence or progression of disease following treatment",
        "how to monitor a chronic disease in a stable patient where a change in patient status might indicate a need to change therapy",
        "most appropriate long-term treatment"
    ]

    def __init__(self, engine: str = None):
        """
        Initialize the pipeline.

        Args:
            engine: LLM engine to use
        """
        self.engine = engine or ENGINE
        self.llm = ChatOpenAI(
            model=self.engine,
            temperature=0.7,
            openai_api_key=OPENAIKEY,
            openai_organization=OPENAIORG
        )
        self.question_graph = QuestionGenerationGraph(engine=self.engine)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the complete pipeline graph.

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(PipelineState)

        # Add nodes
        workflow.add_node("generate_topics", self.generate_topics_node)
        workflow.add_node("generate_keypoint", self.generate_keypoint_node)
        workflow.add_node("generate_question", self.generate_question_node)

        # Set entry point
        workflow.set_entry_point("generate_topics")

        # Add edges
        workflow.add_edge("generate_topics", "generate_keypoint")
        workflow.add_edge("generate_keypoint", "generate_question")

        # Conditional edge to process all topics
        workflow.add_conditional_edges(
            "generate_question",
            self.should_process_more_topics,
            {
                "continue": "generate_keypoint",
                "end": END
            }
        )

        return workflow.compile()

    @retry_parse_fail_prone_cmd
    def generate_topics_node(self, state: PipelineState) -> Dict[str, Any]:
        """
        Node: Generate USMLE topics from clinical note.

        Args:
            state: Current state

        Returns:
            Updated state with topic list
        """
        print("\n=== GENERATING TOPICS ===")

        with open(state['topic_examples_path']) as f:
            fs_dict = json.load(f)

        topic_sim_ex_prompt = PromptTemplate(
            suffix="Clinical notes and their associated topics",
            input_variables=["clinical_note", "topic_list"],
            template="Clinical note: {clinical_note}\nTopic list: {topic_list}"
        )

        topic_prompt = FewShotPromptTemplate(
            examples=fs_dict,
            example_prompt=topic_sim_ex_prompt,
            suffix="Please select five (5) topics from the provided list of USMLE topics that are closely related to the given clinical note, use the clinical topic and topic list examples as a reference. These topics should be suitable for creating USMLE context-based questions that align with the content of the clinical notes. \nClinical Note: {clinical_note}\n USMLE topics: {usmle_topics}\nTopic list:",
            input_variables=["clinical_note", "usmle_topics"]
        )

        topic_chain = LLMChain(llm=self.llm, prompt=topic_prompt)
        topic_output = topic_chain.run({
            "clinical_note": state['clinical_note'],
            "usmle_topics": self.USMLE_TOPICS
        })

        # Parse topics (assuming comma-separated or numbered list)
        topics = [t.strip().strip('0123456789.- ') for t in topic_output.split('\n') if t.strip()]
        topics = [t for t in topics if t]  # Remove empty strings

        print(f"Generated topics: {topics}")

        return {
            "topic_list": topics,
            "topics_to_process": topics,
            "current_topic_index": 0
        }

    @retry_parse_fail_prone_cmd
    def retrieve_sim_keypoints_colbert(self, content: str, k: int = 1) -> Dict[str, Any]:
        """
        Retrieve similar keypoints using ColBERT API.

        Args:
            content: Content to search for
            k: Number of results

        Returns:
            Top-k keypoint result
        """
        p = {'query': content, 'k': k}
        r = requests.get(COLBERT_API, params=p)
        res_json = r.json()
        qbank_topk = res_json['topk'][0]
        print(f"Retrieved keypoint: {qbank_topk}")
        return qbank_topk

    @retry_parse_fail_prone_cmd
    def generate_keypoint_node(self, state: PipelineState) -> Dict[str, Any]:
        """
        Node: Generate keypoint for current topic.

        Args:
            state: Current state

        Returns:
            Updated state with keypoint
        """
        current_topic = state['topics_to_process'][state['current_topic_index']]
        print(f"\n=== GENERATING KEYPOINT for topic: {current_topic} ===")

        # Retrieve similar keypoints
        retrieved_kp = self.retrieve_sim_keypoints_colbert(state['clinical_note'], 1)
        retrieved_kp_score = retrieved_kp['score']
        retrieved_keypoints = retrieved_kp['text']

        with open(state['keypoint_examples_path']) as f:
            fs_dict = json.load(f)

        kp_sim_ex_prompt = PromptTemplate(
            suffix="Example keypoints for a given clinical note and based on a topic: ",
            input_variables=["clinical_note", "topic", "keypoint"],
            template="Clinical note: {clinical_note}\nTopic: {topic}\nKeypoint: {keypoint}"
        )

        keypoint_prompt = FewShotPromptTemplate(
            examples=fs_dict,
            example_prompt=kp_sim_ex_prompt,
            suffix="Please extract a keypoint from the provided list of USMLE concepts. These concepts are organized in a hierarchical manner, starting from the most general and progressively becoming more specific. The keypoint you extract should ideally be specific and concise, covering one or two USMLE concepts. This keypoint will be used as the central focus for generating a USMLE question based on a clinical note within the specified topic. The goal is to ensure a strong and relevant connection between the concept and the question..:\nClinical Note: {clinical_note}\nTopic: {topic}\n USMLE concepts: {usmle_concepts}\nKeypoint:",
            input_variables=["clinical_note", "topic", "usmle_concepts"]
        )

        keypoint_chain = LLMChain(llm=self.llm, prompt=keypoint_prompt)
        keypoint_output = keypoint_chain.run({
            "clinical_note": state['clinical_note'],
            "topic": current_topic,
            "usmle_concepts": retrieved_keypoints
        })

        print(f"Generated keypoint: {keypoint_output}")

        return {
            "current_topic": current_topic,
            "retrieved_kp_score": retrieved_kp_score,
            "retrieved_keypoints": retrieved_keypoints,
            "keypoint": keypoint_output
        }

    def generate_question_node(self, state: PipelineState) -> Dict[str, Any]:
        """
        Node: Generate question with iterative refinement.

        Args:
            state: Current state

        Returns:
            Updated state with generated question
        """
        print(f"\n=== GENERATING QUESTION for topic: {state['current_topic']} ===")

        # Use the question generation graph
        content_to_fb = self.question_graph.generate(
            clinical_note=state['clinical_note'],
            keypoint=state['keypoint'],
            topic=state['current_topic'],
            max_attempts=state['max_attempts']
        )

        # Store the result
        question_result = {
            "clinical_note": state['clinical_note'],
            "topic": state['current_topic'],
            "keypoint": state['keypoint'],
            "retrieved_kp_score": state['retrieved_kp_score'],
            "retrieved_keypoints": state['retrieved_keypoints'],
            "content_to_fb": content_to_fb
        }

        generated_questions = state['generated_questions'].copy()
        generated_questions.append(question_result)

        return {
            "generated_questions": generated_questions,
            "current_topic_index": state['current_topic_index'] + 1
        }

    def should_process_more_topics(self, state: PipelineState) -> str:
        """
        Conditional edge: Check if more topics need processing.

        Args:
            state: Current state

        Returns:
            "continue" if more topics, "end" if done
        """
        if state['current_topic_index'] < len(state['topics_to_process']):
            return "continue"
        return "end"

    def run(self, clinical_note: str,
            topic_examples_path: str = "data/prompt/usmle/topic_fewshot.jsonl",
            keypoint_examples_path: str = "data/prompt/usmle/keypoint_fewshot.jsonl",
            max_attempts: int = 4) -> List[Dict[str, Any]]:
        """
        Run the complete pipeline.

        Args:
            clinical_note: Clinical note text
            topic_examples_path: Path to topic generation examples
            keypoint_examples_path: Path to keypoint generation examples
            max_attempts: Max iterations for question refinement

        Returns:
            List of generated questions with metadata
        """
        initial_state: PipelineState = {
            "clinical_note": clinical_note,
            "max_attempts": max_attempts,
            "topic_examples_path": topic_examples_path,
            "topic_list": None,
            "keypoint_examples_path": keypoint_examples_path,
            "current_topic": None,
            "retrieved_kp_score": None,
            "retrieved_keypoints": None,
            "keypoint": None,
            "generated_questions": [],
            "topics_to_process": [],
            "current_topic_index": 0
        }

        final_state = self.graph.invoke(initial_state)
        return final_state['generated_questions']


# Convenience function
def run_full_pipeline(clinical_note: str, max_attempts: int = 4) -> List[Dict[str, Any]]:
    """
    Run the complete MedQG pipeline.

    Args:
        clinical_note: Clinical note text
        max_attempts: Max iterations for question refinement

    Returns:
        List of generated questions
    """
    pipeline = MedQGPipeline()
    return pipeline.run(clinical_note, max_attempts=max_attempts)
