import re
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from enum import Enum

# 대신 직접 정의
class JudgeType(Enum):
    RUBRIC_AND_RESPONSE = "rubric_and_response"
    RUBRIC_RESPONSE_AND_GOLD = "rubric_response_and_gold"
    RESPONSE_COMPARISON = "response_comparison"

def extract_final_answer(raw_output: str) -> str:
    """
    Extracts the final answer from a raw output that may contain unnecessary chain-of-thought (CoT) details,
    using the MULTILINGUAL_ANSWER_PATTERN.

    Steps:
      1) Pattern matching: Capture the text following markers such as "정답:", "답변:", "Answer:", etc.
      2) If no match is found, return the original raw_output.
      3) If a match is found, process the captured group ("content")—for example, split by newline and take the first part,
         or simply apply strip()—as appropriate.

    Returns:
        str: The extracted final answer (or the original raw_output if no match is found).
    """
    match = re.search(MULTILINGUAL_ANSWER_PATTERN, raw_output, flags=re.DOTALL)
    if match:
        # The "content" group captures the actual final answer part.
        content = match.group("content")
        # Final processing: here we simply strip the whitespace.
        return content.strip()
    else:
        # If the pattern is not found, return the original raw output.
        return raw_output

def default_cot_parser(raw_output: str) -> Tuple[str, str]:
    """
    Default chain-of-thought (CoT) parser.
    Uses the extract_final_answer function to extract the final answer from the raw output,
    and considers everything before the final answer as the chain-of-thought.

    Returns:
        Tuple[str, str]: A tuple (chain_of_thought, final_answer).
    """
    final_answer = extract_final_answer(raw_output)
    if final_answer:
        # Find the last occurrence of the final_answer within raw_output
        idx = raw_output.rfind(final_answer)
        if idx != -1:
            chain_of_thought = raw_output[:idx].strip()
        else:
            chain_of_thought = ""
    else:
        chain_of_thought = ""
    return chain_of_thought, final_answer


MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:",
    r"Final\s*Answer\s*:",
    r"답변\s*:",
    r"정답\s*:",
    r"정답은\s*:",
    r"답\s*:",
    r"답은\s*:",
    r"답안\s*:",
    r"答案\s*[:：]",
    r"解答\s*[:：]",
    r"回答\s*[:：]",
    r"答\s*[:：]",
    r"Jawaban\s*:",
    r"Réponse\s*:",
    r"Resposta\s*:",
    r"Jibu\s*:",
    r"Idahun\s*:",
    r"Ìdáhùn\s*:",
    r"Idáhùn\s*:",
    r"Àmọ̀nà\s*:",
    r"Àdáhùn\s*:",
    r"Ànúgọ\s*:",
    r"Àṣàyàn\s*:",
]

MULTILINGUAL_ANSWER_PATTERN = (
    r"(?i)(" + "|".join(MULTILINGUAL_ANSWER_REGEXES) + r")\s*(?P<content>.+)"
)

JUDGE_PROMPTS = {
    JudgeType.RUBRIC_AND_RESPONSE: """You are an expert evaluator. Evaluate the following response based on the rubric.

Rubric:
{rubric}

Response to evaluate:
{response}

Provide a detailed feedback based on the rubric, then end with your verdict in this EXACT format:
[RESULT] X where X is an integer between 1 and 5.""",

    JudgeType.RUBRIC_RESPONSE_AND_GOLD: """You are an expert evaluator. Compare the response with the reference answer.

Reference Answer:
{gold}

Response to evaluate:
{response}

IMPORTANT: Your response MUST be in this EXACT format:
[[true]] if the response is correct, or [[false]] if it is incorrect.""",

    JudgeType.RESPONSE_COMPARISON: """You are an expert evaluator. Compare the two responses and determine which one is better.

Response A:
{response}

Response B:
{response_b}

IMPORTANT: Your response MUST be in this EXACT format:
[[A]] if Response A is better, or [[B]] if Response B is better.

First provide your detailed comparison, then end with your verdict in the specified format."""
}
