# Training Data Generation Prompts
# Use these with a cheaper LLM to generate diverse text samples

SYSTEM_PROMPT = """You are a text generation assistant. Generate diverse,
natural-sounding text samples that contain repetitive patterns suitable for
compression testing. Focus on creating text with:
- Repeated phrases and words
- Technical terminology used multiple times
- Natural conversational patterns
- Varied sentence structures

Output ONLY the raw text, no explanations or formatting."""


GENERATION_PROMPTS = [
    # Technical/Documentation
    """Generate a 200-word technical paragraph about database connections.
Use phrases like "database connection", "connection pool", "query execution"
multiple times naturally.""",
    # Repetitive Narrative
    """Write a 200-word story where a character performs the same action
repeatedly. Use phrases like "walked to the door", "opened the door",
"closed the door" naturally throughout.""",
    # Business Communication
    """Write a 200-word business email discussing project deadlines.
Repeat phrases like "project deadline", "team meeting", "action items"
naturally throughout.""",
    # Instructional Content
    """Write 200-word step-by-step instructions for making coffee.
Use words like "coffee", "water", "filter", "brew" repeatedly.""",
    # Conversational
    """Write a 200-word dialogue between two people discussing weather.
Use phrases like "the weather", "going to rain", "sunny day" repeatedly.""",
    # Legal/Formal
    """Write a 200-word terms of service excerpt. Use phrases like
"the user", "the service", "these terms", "we reserve the right" repeatedly.""",
    # Scientific
    """Write a 200-word paragraph about photosynthesis. Use terms like
"photosynthesis", "chlorophyll", "carbon dioxide", "sunlight" repeatedly.""",
    # Customer Support
    """Write a 200-word customer support response about a shipping delay.
Use phrases like "your order", "shipping delay", "we apologize" repeatedly.""",
]


BATCH_PROMPT = """Generate {count} diverse text samples, each 150-250 words.
Each sample should contain naturally repetitive patterns (repeated phrases,
terms, or structures). Separate samples with "---".

Topics to cover: technical docs, stories, emails, instructions, dialogues.
Output raw text only, no numbering or labels."""
