"""
LangGraph-based Question Generation Pipeline for MedQG

This module implements the question generation workflow using LangGraph,
replacing the manual while-loop orchestration with a stateful graph.
"""

import os
import re
from typing import Dict, Any

from langgraph.graph import StateGraph, END
from src.usmle.graph_state import QuestionGenerationState
from src.usmle.task_init_lgc import UsmleQgenTaskInitLgc
from src.usmle.task_iterate import UsmleQgenTaskIterate
from src.usmle.answer import UsmleQgenAnswer
from src.usmle.feedback_lgc import UsmleQgenFeedbackLgc
from src.usmle.gen_order import gen_order


ENGINE = os.getenv("ENGINE")


def generate_options(distractor_options: str, correct_answer: str) -> str:
    """
    Generate shuffled multiple choice options from distractors and correct answer.

    Args:
        distractor_options: String containing distractor options
        correct_answer: The correct answer

    Returns:
        Formatted string with shuffled options
    """
    import random

    print(f"distract: {distractor_options}")
    distractor_options = distractor_options.replace('\n', '')
    distractor_options = distractor_options.replace(' :', ':')

    ustr = ') ' if 'a)' in distractor_options.lower() else ': '
    option_key_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    distractor_options_list = [correct_answer]

    for key in option_key_list:
        key_with_sep = key + ustr
        try:
            opt = re.search(re.escape(key_with_sep) + r"(.*?)" + re.escape(key_with_sep),
                            distractor_options, re.IGNORECASE).group(1)
            print(opt[:-1])
            distractor_options_list.append(opt[:-1].strip())
        except:
            print(f"Except: {distractor_options}")
            opt = re.search(re.escape(key_with_sep) + r"(.*)", distractor_options, re.IGNORECASE).group(1)
            print(opt)
            distractor_options_list.append(opt.strip())
            break

    random.shuffle(distractor_options_list)
    options = ''
    for k in range(len(distractor_options_list)):
        options += option_key_list[k] + ' : ' + distractor_options_list[k]
    return options


def get_dec_score(score: str) -> float:
    """
    Convert fraction score string to decimal.

    Args:
        score: Score in format "X/Y"

    Returns:
        Decimal score (X/Y)
    """
    split = score.split('/')
    num = int(re.sub("[^0-9]", "", split[0]))
    deno = int(re.sub("[^0-9]", "", split[1]))
    return num / deno


def check_stop(context_score: str, question_score: str, correct_answer_score: str,
               distractor_option_score: str, reasoning_score: str) -> bool:
    """
    Check if quality threshold is met for all components.

    Args:
        context_score: Score for context
        question_score: Score for question
        correct_answer_score: Score for correct answer
        distractor_option_score: Score for distractors
        reasoning_score: Score for reasoning

    Returns:
        True if should continue iterating, False if threshold met
    """
    threshold = 0.9
    if (get_dec_score(context_score) >= threshold and
        get_dec_score(reasoning_score) >= threshold and
        get_dec_score(question_score) >= threshold and
        get_dec_score(correct_answer_score) >= threshold and
        get_dec_score(distractor_option_score) >= threshold):
        print("===== Stop condition met ======\n\n")
        return False
    return True


class QuestionGenerationGraph:
    """
    LangGraph-based question generation pipeline.

    This class orchestrates the entire question generation workflow using
    a stateful graph with nodes for initialization, iteration, feedback, and evaluation.
    """

    def __init__(self, engine: str = None):
        """
        Initialize the question generation graph.

        Args:
            engine: LLM engine to use (defaults to ENGINE env var)
        """
        self.engine = engine or ENGINE

        # Initialize components
        self.task_init_lgc = UsmleQgenTaskInitLgc(
            engine=self.engine,
            prompt_examples="data/prompt/usmle/init.jsonl"
        )
        self.task_answer = UsmleQgenAnswer(
            engine=self.engine,
            prompt_examples="data/prompt/usmle/answer.jsonl"
        )
        self.task_feedback_lgc = UsmleQgenFeedbackLgc(
            engine=self.engine,
            prompt_examples="data/prompt/usmle/feedback.jsonl",
            rubrics_path="data/prompt/usmle/reasoning_rubrics.jsonl"
        )
        self.task_iterate = UsmleQgenTaskIterate(
            engine=self.engine,
            prompt_examples="data/prompt/usmle/iterate.jsonl"
        )

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph for question generation.

        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(QuestionGenerationState)

        # Add nodes
        workflow.add_node("generate_question", self.generate_question_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("evaluate_feedback", self.evaluate_feedback_node)

        # Set entry point
        workflow.set_entry_point("generate_question")

        # Add edges
        workflow.add_edge("generate_question", "generate_answer")
        workflow.add_edge("generate_answer", "evaluate_feedback")

        # Add conditional edge for iteration logic
        workflow.add_conditional_edges(
            "evaluate_feedback",
            self.should_continue_decision,
            {
                "continue": "generate_question",
                "end": END
            }
        )

        # Compile the graph
        return workflow.compile()

    def generate_question_node(self, state: QuestionGenerationState) -> Dict[str, Any]:
        """
        Node: Generate or iterate question components.

        On first attempt (n_attempts == 0): Generate initial question.
        On subsequent attempts: Iterate based on feedback.

        Args:
            state: Current state

        Returns:
            Updated state with question components
        """
        print(f"\n{state['n_attempts']} GEN_QUESTION> Clinical Note: {state['clinical_note'][:100]}...")
        print(f"Topic: {state['topic']}, Keypoint: {state['keypoint']}")

        if state['n_attempts'] == 0:
            # Initial generation
            context, question, correct_answer, distractor_options = self.task_init_lgc(
                clinical_note=state['clinical_note'],
                keypoint=state['keypoint'],
                topic=state['topic'],
                order_enum=gen_order.step_by_step
            )
        else:
            # Iterative improvement based on feedback
            # Build content_to_fb from current state
            content_to_fb = [{
                "context": state['context'],
                "question": state['question'],
                "topic": state['topic'],
                "keypoint": state['keypoint'],
                "attempted_answer": state['attempted_answer'],
                "reasoning": state['reasoning'],
                "correct_answer": state['correct_answer'],
                "distractor_options": state['distractor_options'],
                "context_feedback": state['context_feedback'],
                "context_score": state['context_score'],
                "question_feedback": state['question_feedback'],
                "question_score": state['question_score'],
                "correct_answer_feedback": state['correct_answer_feedback'],
                "correct_answer_score": state['correct_answer_score'],
                "distractor_option_feedback": state['distractor_option_feedback'],
                "distractor_option_score": state['distractor_option_score'],
                "reasoning_feedback": state['reasoning_feedback'],
                "reasoning_score": state['reasoning_score']
            }]

            context, question, correct_answer, distractor_options = self.task_iterate(
                clinical_note=state['clinical_note'],
                keypoint=state['keypoint'],
                topic=state['topic'],
                content_to_fb=content_to_fb
            )

        print(f"{state['n_attempts']} GEN> Context: {context}\nQuestion: {question}\n"
              f"Correct answer: {correct_answer}\nDistractor options: {distractor_options}")

        return {
            "context": context,
            "question": question,
            "correct_answer": correct_answer,
            "distractor_options": distractor_options
        }

    def generate_answer_node(self, state: QuestionGenerationState) -> Dict[str, Any]:
        """
        Node: Generate attempted answer and reasoning.

        Args:
            state: Current state

        Returns:
            Updated state with attempted answer and reasoning
        """
        options = generate_options(
            distractor_options=state['distractor_options'],
            correct_answer=state['correct_answer']
        )

        attempted_answer, reasoning = self.task_answer(
            context=state['context'],
            question=state['question'],
            options=options
        )

        print(f"{state['n_attempts']} ANSWER> Attempted: {attempted_answer[:100]}...")
        print(f"Reasoning: {reasoning[:100]}...")

        return {
            "attempted_answer": attempted_answer,
            "reasoning": reasoning
        }

    def evaluate_feedback_node(self, state: QuestionGenerationState) -> Dict[str, Any]:
        """
        Node: Evaluate all components and generate feedback.

        Args:
            state: Current state

        Returns:
            Updated state with feedback and scores
        """
        (context_feedback, context_score,
         question_feedback, question_score,
         reasoning_feedback, reasoning_score,
         correct_answer_feedback, correct_answer_score,
         distractor_option_feedback, distractor_option_score) = self.task_feedback_lgc(
            clinical_note=state['clinical_note'],
            keypoint=state['keypoint'],
            topic=state['topic'],
            context=state['context'],
            question=state['question'],
            correct_answer=state['correct_answer'],
            distractor_options=state['distractor_options'],
            attempted_answer=state['attempted_answer'],
            reasoning=state['reasoning']
        )

        print(f"{state['n_attempts']} SCORES> Context: {context_score} | Question: {question_score} | "
              f"Answer: {correct_answer_score} | Distractors: {distractor_option_score} | "
              f"Reasoning: {reasoning_score}")

        # Store this iteration's results
        iteration_result = {
            "context": state['context'],
            "question": state['question'],
            "topic": state['topic'],
            "keypoint": state['keypoint'],
            "attempted_answer": state['attempted_answer'],
            "reasoning": state['reasoning'],
            "correct_answer": state['correct_answer'],
            "distractor_options": state['distractor_options'],
            "context_feedback": context_feedback,
            "context_score": context_score,
            "question_feedback": question_feedback,
            "question_score": question_score,
            "correct_answer_feedback": correct_answer_feedback,
            "correct_answer_score": correct_answer_score,
            "distractor_option_feedback": distractor_option_feedback,
            "distractor_option_score": distractor_option_score,
            "reasoning_feedback": reasoning_feedback,
            "reasoning_score": reasoning_score
        }

        # Append to results list
        content_to_fb_ret = state['content_to_fb_ret'].copy()
        content_to_fb_ret.append(iteration_result)

        # Increment attempts
        n_attempts = state['n_attempts'] + 1

        return {
            "context_feedback": context_feedback,
            "context_score": context_score,
            "question_feedback": question_feedback,
            "question_score": question_score,
            "correct_answer_feedback": correct_answer_feedback,
            "correct_answer_score": correct_answer_score,
            "distractor_option_feedback": distractor_option_feedback,
            "distractor_option_score": distractor_option_score,
            "reasoning_feedback": reasoning_feedback,
            "reasoning_score": reasoning_score,
            "content_to_fb_ret": content_to_fb_ret,
            "n_attempts": n_attempts
        }

    def should_continue_decision(self, state: QuestionGenerationState) -> str:
        """
        Conditional edge: Decide whether to continue iterating or stop.

        Args:
            state: Current state

        Returns:
            "continue" if should iterate more, "end" if done
        """
        # Check if max attempts reached
        if state['n_attempts'] >= state['max_attempts']:
            print(f"\n===== Max attempts ({state['max_attempts']}) reached ======\n")
            return "end"

        # Check if quality threshold met
        should_continue = check_stop(
            state['context_score'],
            state['question_score'],
            state['correct_answer_score'],
            state['distractor_option_score'],
            state['reasoning_score']
        )

        if not should_continue:
            return "end"

        print(f"\n===== Continuing iteration {state['n_attempts']}/{state['max_attempts']} ======\n")
        return "continue"

    def generate(self, clinical_note: str, keypoint: str, topic: str,
                 max_attempts: int = 4) -> list:
        """
        Generate a USMLE question with iterative refinement.

        Args:
            clinical_note: Clinical note text
            keypoint: Test keypoint/concept
            topic: USMLE topic
            max_attempts: Maximum refinement iterations

        Returns:
            List of iteration results with feedback
        """
        # Initialize state
        initial_state: QuestionGenerationState = {
            "clinical_note": clinical_note,
            "keypoint": keypoint,
            "topic": topic,
            "max_attempts": max_attempts,
            "n_attempts": 0,
            "context": None,
            "question": None,
            "correct_answer": None,
            "distractor_options": None,
            "attempted_answer": None,
            "reasoning": None,
            "context_feedback": None,
            "context_score": "0/1",
            "question_feedback": None,
            "question_score": "0/1",
            "correct_answer_feedback": None,
            "correct_answer_score": "0/1",
            "distractor_option_feedback": None,
            "distractor_option_score": "0/1",
            "reasoning_feedback": None,
            "reasoning_score": "0/1",
            "content_to_fb_ret": [],
            "should_continue": True
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        return final_state['content_to_fb_ret']


# Convenience function for backward compatibility
def autofb_usmleqgen(clinical_note: str, keypoint: str, topic: str,
                     max_attempts: int = 4) -> list:
    """
    Generate USMLE question with automatic feedback and refinement.

    This function provides backward compatibility with the original implementation.

    Args:
        clinical_note: Clinical note text
        keypoint: Test keypoint/concept
        topic: USMLE topic
        max_attempts: Maximum refinement iterations

    Returns:
        List of iteration results with feedback
    """
    graph = QuestionGenerationGraph()
    return graph.generate(clinical_note, keypoint, topic, max_attempts)
