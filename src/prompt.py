prompt_template = """
You are an expert at creating interview questions from documents.

STRICT RULES:
1. Generate EXACTLY 10 questions
2. Each question must be numbered: 1., 2., 3., etc.
3. Each question MUST end with a question mark (?)
4. Do NOT include any answers
5. Do NOT include explanations
6. Do NOT list content from the document
7. Questions should test understanding of key concepts
8. Keep questions concise and clear

DOCUMENT CONTENT:
------------
{text}
------------

Generate exactly 10 interview questions:
"""

refine_template = """
You are an expert at refining interview questions.

EXISTING QUESTIONS:
{existing_answer}

NEW CONTENT:
------------
{text}
------------

INSTRUCTIONS:
1. Review existing questions and new content
2. If new content is already covered, keep existing questions unchanged
3. If new content adds NEW information, add 2-5 new questions maximum
4. Continue numbering from existing questions (e.g., if 10 exist, start at 11.)
5. Each question MUST end with (?)
6. Do NOT include answers or explanations
7. Do NOT repeat questions

Output the complete refined question list (existing + new if needed):
"""