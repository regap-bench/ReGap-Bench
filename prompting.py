from __future__ import annotations

from pathlib import Path

from data_loading import load_question, validate_task


PROMPT_TEMPLATES = {
    "shape_counting": """{question}

**Rules**:
- Rectangles include squares.
- Parallelograms include rectangles.
- Trapeziums include parallelograms.
- Isosceles triangles include equilateral triangles.
- Ellipses include circles.

**Instructions**:
You may think step-by-step and show your reasoning to carefully count the shapes.
Once you have finished your reasoning, you MUST enclose your final numerical integer count within <answer> tags.

Example format:
[Your reasoning and step-by-step counting here...]
<answer>7</answer>
""",
    "angle_ranking": """{question}

**Instructions**:
You are tasked with ranking the angles indicated by the arcs based on their visual size.
The angles are identified by numerical labels in parentheses, such as (1), (2), (3), etc.

You may think step-by-step and show your reasoning to carefully compare the angles.
Once you have finished your reasoning, you MUST enclose your final answer within <answer> tags as a standard array of integers representing the labels. Do not include the parentheses in your final array.

Example format:
[Your reasoning and step-by-step comparison here...]
<answer>[3, 1, 2, 4]</answer>
""",
    "size_ranking": """{question}

**Instructions**:
You are tasked with ranking the shapes based on their visual size.
The shapes are identified by numerical labels in parentheses, such as (1), (2), (3), etc.

You may think step-by-step and show your reasoning to carefully compare the sizes.
Once you have finished your reasoning, you MUST enclose your final answer within <answer> tags as a standard array of integers representing the labels. Do not include the parentheses in your final array.

Example format:
[Your reasoning and step-by-step comparison here...]
<answer>[3, 1, 2, 4]</answer>
""",
    "clock_ranking": """{question}

**Instructions**:
You are tasked with ranking the clocks based on the times shown.
The clocks are identified by numerical labels in parentheses, such as (1), (2), (3), etc.
Use the AM/PM marker when comparing times.

You may think step-by-step and show your reasoning to carefully compare the times.
Once you have finished your reasoning, you MUST enclose your final answer within <answer> tags as a standard array of integers representing the labels. Do not include the parentheses in your final array.

Example format:
[Your reasoning and step-by-step comparison here...]
<answer>[3, 1, 2, 4]</answer>
""",
    "transformation_prediction": """{question}

**Instructions**:
You may think step-by-step and show your reasoning to carefully apply the transformations in order.
Once you have finished your reasoning, you MUST enclose your final answer within <answer> tags as a single capital letter.

Example format:
[Your reasoning and step-by-step transformation here...]
<answer>B</answer>
""",
}


def get_question_from_file(question_file_path: str | Path) -> dict:
    return load_question(question_file_path)


def select_prompt(task: str, question_or_data: str | dict) -> str:
    task = validate_task(task)
    if isinstance(question_or_data, dict):
        question = question_or_data.get("question", "")
    else:
        question = question_or_data
    if not question:
        raise ValueError("Question text is empty.")
    return PROMPT_TEMPLATES[task].format(question=question)


def build_prompt(question: str, task: str) -> str:
    return select_prompt(task, question)
