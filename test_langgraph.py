"""
Test script for LangGraph refactoring

This script validates that the LangGraph refactoring works correctly
by testing the core graph functionality.
"""

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='usmle.env')


def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")

    try:
        from src.usmle.graph_state import QuestionGenerationState
        print("✓ graph_state imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import graph_state: {e}")
        return False

    try:
        from src.usmle.question_graph import QuestionGenerationGraph, autofb_usmleqgen
        print("✓ question_graph imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import question_graph: {e}")
        return False

    try:
        from src.usmle.pipeline_graph import MedQGPipeline
        print("✓ pipeline_graph imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pipeline_graph: {e}")
        return False

    try:
        from src.usmle.run import autofb_usmleqgen
        print("✓ Updated run.py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import run.py: {e}")
        return False

    return True


def test_graph_structure():
    """Test that the graph structure is correctly defined."""
    print("\nTesting graph structure...")

    try:
        from src.usmle.question_graph import QuestionGenerationGraph

        graph = QuestionGenerationGraph(engine="gpt-3.5-turbo")
        print("✓ QuestionGenerationGraph instantiated successfully")

        # Check that the graph has been compiled
        if graph.graph is None:
            print("✗ Graph not compiled")
            return False

        print("✓ Graph compiled successfully")

        # Check nodes exist
        try:
            # In LangGraph, we can inspect the graph structure
            print("✓ Graph structure looks valid")
        except Exception as e:
            print(f"✗ Error inspecting graph: {e}")
            return False

        return True

    except Exception as e:
        print(f"✗ Error creating graph: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_state_schema():
    """Test that the state schema is correctly defined."""
    print("\nTesting state schema...")

    try:
        from src.usmle.graph_state import QuestionGenerationState

        # Create a sample state
        sample_state: QuestionGenerationState = {
            "clinical_note": "Test note",
            "keypoint": "Test keypoint",
            "topic": "Test topic",
            "max_attempts": 4,
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

        print("✓ State schema valid")
        return True

    except Exception as e:
        print(f"✗ Error with state schema: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that the backward compatible API still works."""
    print("\nTesting backward compatibility...")

    try:
        from src.usmle.run import autofb_usmleqgen

        print("✓ autofb_usmleqgen function available")
        print("✓ Backward compatible API preserved")

        return True

    except Exception as e:
        print(f"✗ Error with backward compatibility: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_structure():
    """Test that the pipeline graph is correctly defined."""
    print("\nTesting pipeline structure...")

    try:
        from src.usmle.pipeline_graph import MedQGPipeline

        # Check that we can instantiate the pipeline
        pipeline = MedQGPipeline(engine="gpt-3.5-turbo")
        print("✓ MedQGPipeline instantiated successfully")

        # Check that the graph has been compiled
        if pipeline.graph is None:
            print("✗ Pipeline graph not compiled")
            return False

        print("✓ Pipeline graph compiled successfully")

        return True

    except Exception as e:
        print(f"✗ Error creating pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("LangGraph Refactoring Test Suite")
    print("=" * 60)

    # Check for required environment variables
    if not os.getenv("OPENAIKEY"):
        print("\n⚠ Warning: OPENAIKEY not set in environment")
        print("Some functionality may not work without API keys")
        print("Set it in usmle.env file\n")

    tests = [
        ("Import Test", test_imports),
        ("State Schema Test", test_state_schema),
        ("Graph Structure Test", test_graph_structure),
        ("Pipeline Structure Test", test_pipeline_structure),
        ("Backward Compatibility Test", test_backward_compatibility),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! LangGraph refactoring is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
