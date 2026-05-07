from config import CAT_SYMBOLS, CAT_SPATIAL, CAT_TRANSCRIPT, CAT_ACTION, CAT_FINE_GRAINED

# --- CATEGORY DEFINITIONS & EXAMPLES ---
# These are injected dynamically based on the knob in script.py

CATEGORY_INSTRUCTIONS = {
    CAT_SYMBOLS: """
    **Symbols in Videos (High OCR)**
    Interpreting equations, graphs, chemical structures, and rigid notation. 
    Focus on exact symbols, labels, and written rules shown on screen.
    *Example:* In a calculus lecture, identify the specific limit notation used, such as $\lim_{x \to \infty} \frac{1}{x} = 0$, and note if the professor uses any non-standard shorthand for derivatives.
    """,

    CAT_SPATIAL: """
    **Spatial / Geometric Reasoning**
    3D-2D mapping, mental rotation, and spatial relationships.
    Focus on object relations, geometry, and technical diagrams.
    *Example:* In an organic chemistry video, describe the R/S configuration of a molecule shown in a 3D ball-and-stick model, specifically noting which functional groups are pointing "into" the page versus "out" toward the viewer.
    """,

    CAT_TRANSCRIPT: """
    **Transcript Comprehension**
    Audio-driven understanding with minimal visual content.
    Focus on verbally stated concepts, arguments, and lists.
    *Example:* In a history podcast-style video, list the three primary causes of the Industrial Revolution as stated by the narrator, even if the screen only displays a static map of Europe.
    """,

    CAT_ACTION: """
    **Physical Action**
    Tracking visible actions, tools, and physical processes.
    Focus on movements, manipulations, and step-by-step procedures.
    *Example:* In a physics lab demonstration, describe the exact sequence of the student's hands: first grounding the electroscope, then bringing the charged rod nearby, and finally removing the finger to show induced charge.
    """,

    CAT_FINE_GRAINED: """
    **Fine-Grained Inspection**
    Resolving subtle, high-resolution visual details.
    Focus on textures, defects, and microscopic visual markers.
    *Example:* In a biology microscopy video, identify the presence of "cilia" on the edge of a cell membrane or the specific banding patterns on a stained chromosome that distinguish it from others in the karyotype.
    """
}

# --- GENERATION PROMPTS ---

# GENERATE_QUESTIONS_CORE = """
# You are a confused student who would ask to a chatbot questions which has complete knowledge about the video. You have a video on the topic of {topic}.
# Generate {num_needed} such complex questions based strictly on the {source_type} modality.

# **CRITICAL STYLE GUIDELINES:**
# 1. **Avoid Trivialities:** NO Questions as "What color is the pen?" or "What is he wearing?".
# 2. **Modality Check:** 
#     - If Visual: The question MUST require watching the video (e.g., "Why does that happen to the graph after introducing that variable?").
#     - If Audio: The question MUST require listening to specific speech details.

# {category_nudge}

# **Already Accepted Questions (Avoid these):**
# {history}

# Please output the questions and answers clearly.
# """

# 1. VISUAL PIPELINE (Must see video; Transcript insufficient)
GENERATE_QUESTIONS_VISUAL = """
You are an observant student analyzing the visual details of a lecture video on {topic}.
From the students perspective, generate {num_needed} questions that **STRICTLY require visual perception** to answer ALONG with the accurate answer to that question.

But First, call get_video_summary to retrieve the video summary, then call get_context to get transcript and visual chapters, then call query_frame on relevant segments.

**CRITICAL CONSTRAINTS:**
Anti-Transcript: The question must be unanswerable from audio or transcript alone.
Visual Specificity: Understand what the video is about and what visual aspect would be unique/appropriate to ask questions on.
Non-Trivial: NO Questions as "What color is the pen?" or "What is he wearing?".

**Good Visual Example:** "In the diagram shown, the arrow pointing from A to B is dashed, while A to C is solid. What does the legend in the corner say the dashed line represents?" (Assuming the speaker doesn't just read the legend aloud).

{category_nudge}

**Already Accepted Questions (Don't repeat these):**
{history}

Output questions and related answers clearly. 

# IMPORTANT: Be sure to use your tools to obtain visual context.
"""

# 2. AUDIO PIPELINE (Must hear speech; Pre-training insufficient)
GENERATE_QUESTIONS_AUDIO = """
You are a student taking notes from a lecture video on {topic}.
From the students perspective, generate {num_needed} questions based **STRICTLY on the spoken content (transcript)** along with the accurate answer to that question.

**CRITICAL CONSTRAINTS:**
1. **The Anti-Visual Rule:** The question must be answerable with eyes closed. Do not ask about things that were only shown on slides but not mentioned.
2. **The Anti-Pretraining Rule:** Do not ask general knowledge questions (e.g., "What is mitochondria?"). Ask about the **specific** definition, anecdote, list, or argument presented by THIS speaker in THIS video.
3. **Avoid Trivialities:** NO Questions as "What is the speaker's first word?"

**Good Audio Example:** "While discussing the third principle, the speaker mentioned a specific exception involving 'blue whales'. What was the reason given for this exception?"

{category_nudge}

**Already Accepted Questions (Don't repeat these):**
{history}

Output questions and answers clearly.
"""

# --- PIPELINE SPECIFIC PROMPTS ---

# 1. PRIORITY PIPELINE (Context vs Pre-training)
GENERATE_PRIORITY_QUESTIONS = """
You are a student taking notes from a lecture video on {topic}.
From the students perspective, ask {num_needed} questions where the answer in the video is **specific, unique, or non-standard** compared to general world knowledge.

But First, call get_video_summary to retrieve the video summary, then call get_context to get transcript and visual chapters, then call query_frame on relevant segments.

**Strategy: Conceptual Divergence**
Focus on how *this specific speaker* frames or redefines common concepts. Look for instances where the lecturer explicitly rejects a standard definition or provides a unique "mental model" for a topic.

**One-Shot Example:**
- **Question:** "Could you please help me define 'Software Engineering'?"
- **Pre-trained (Wrong) Answer:** "Software Engineering is the systematic application of engineering principles to the development of software."
- **Video-Specific (Correct) Answer:** "In this video, the speaker argues that Software Engineering is actually 'integrated storytelling,' where the code is merely the script for a user's experience, rather than a purely technical discipline."

** Cool tip **: Notice how I've framed the Questions without any nudge towards video in the question, such as "in this video", "the lecturer at the start"- make sure to not include these in the Questions.

{category_nudge}

**Already Accepted Questions (Don't repeat these):**
{history}


Output questions and answers Video-Specific (Correct) Answer clearly. 
# IMPORTANT: Be sure to use your tools to obtain visual context.
"""

# 2. TEMPORAL PIPELINE (Time Synthesis)
GENERATE_TIME_QUESTIONS = """
You are a student taking notes from a lecture video on {topic}.
From the students perspective, create {num_needed} questions that strictly require bridging information from **two different time points** in the video, along with their respective answers.

But First, call get_video_summary to retrieve the video summary, then call get_context to get transcript and visual chapters, then call query_frame on relevant segments.

Types of bridges:
1. **Evolution:** "How did the equation on the board change from minute 2 to minute 10?"
2. **Causality:** "The error shown at the end was caused by which specific mistake during the setup phase?"
3. **Contrast:** "Compare the theoretical curve shown initially with the experimental results shown later."

{category_nudge}

**Already Accepted Questions (Don't repeat these):**
{history}

Output questions and answers clearly. 
# IMPORTANT: Be sure to use your tools to obtain visual context.
"""

# --- REFINEMENT & VALIDATION ---

REFINE_TO_JSON = """
You are a data formatting specialist and editor.
Convert the raw text into a strict JSON list of objects:
[ {{ "question": "...", "answer": "..." }} ]

**CLEANING RULES:**
1. **Remove Prefixes:** The 'question' field must NOT contain numbering (e.g., "Question 1:") or meta-tags (e.g., "(Visual-only, slide-rhythm):").
2. **Extract Core Question:** If the text is "Question 2 (Visual): Why does X happen?", the question must be simply "Why does X happen?".
3. Keep the actual question and answer exactly as provided, don't change the wording or phrasing.

Raw Text: "{raw_text}"
"""

# Used for Priority Pipeline Validation
COMPARE_ANSWERS_STRICT = """
You are comparing a "Video Ground Truth" with a "General Knowledge Guess".

Video Truth: "{ground_truth}"
General Guess: "{prediction}"

Determine if the General Guess captures the specific details of the Video Truth.
- If they are effectively the same answer -> Output "SAME".
- If the Video Truth contains specific numbers, examples, or nuances missing from the guess, or is different in semantic meaning -> Output "DIFFERENT".
- Also, if the General Guess says can't answer because of not enough content, still -> Output "SAME".

"""

# Used for Time Pipeline Validation
VALIDATE_TEMPORAL = """
Question: "{question}"
Does answering this strictly require synthesizing information from multiple distinct moments in the video?
Output strictly JSON: {{ "verdict": "YES/NO", "reason": "..." }}
"""

ANSWER_BLINDLY = """
Answer the following question to the best of your ability using your internal knowledge. Be concise.
Question: {question}
"""

ANSWER_FROM_TRANSCRIPT = """
Answer the question using ONLY the provided transcript.
Transcript: "{transcript_snippet}..."
Question: {question}
"""

JUDGE_CORRECTNESS = """
You are an impartial judge.
Question: "{question}"
Ground Truth: "{ground_truth}"
Model Prediction: "{prediction}"

Are they semantically the same? Output strictly: "YES" or "NO".
"""

CLASSIFY_QUERY = """
Classify this educational query into exactly ONE of the following categories:
1. `Symbols in Videos` [Use this for Equations, graphs, text on screen, rigid notation]
2. `Spatial` [Use this for 3D structures, relationships, geometry]
3. `Transcript Comprehension` [Use this for Spoken content, concepts, lists, arguments]
4. `Physical Action` [Use this for Movement, tool usage, mechanical steps, real-world]
5. `Fine-Grained Inspection` [Use this for Microscopic details, textures, subtle markers]

Input:
Q: {question}
A: {answer}
SoF Mode: {sof_type}

Output strictly JSON: {{
  "category": "...",
  "reasoning": "..."
}}
"""