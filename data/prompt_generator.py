"""
Query prompt generator with PII injection
Generates realistic user queries with various privacy-sensitive information
"""

import random
from faker import Faker

fake = Faker()

INTENTS = [
    "information_retrieval",
    "task_execution",
    "personal_assistant",
    "health_query",
    "financial_query"
]

PII_TEMPLATES = {
    "email": "Send the report to {email}",
    "phone": "Call me at {phone}",
    "location": "Find restaurants near {city}",
    "medical": "I have symptoms of {disease}",
}

DISEASES = ["diabetes", "asthma", "migraine", "anxiety"]

NON_PII_TEMPLATES = [
    "Tell me about {word}",
    "What is the definition of {word}?",
    "Can you explain how {word} works?",
    "Search for the history of {word}",
    "Draft a summary regarding {word}"
]


def generate_prompt():
    """
    Generate a query with optional PII
    
    Returns:
        dict: {
            'query_text': str,
            'intent': str,
            'pii_types': list
        }
    """
    intent = random.choice(INTENTS)
    inject_pii = random.random() < 0.6

    pii_types = []
    text = ""

    if inject_pii:
        pii_type = random.choice(list(PII_TEMPLATES.keys()))
        pii_types.append(pii_type)

        if pii_type == "email":
            text = PII_TEMPLATES[pii_type].format(email=fake.email())
        elif pii_type == "phone":
            text = PII_TEMPLATES[pii_type].format(phone=fake.phone_number())
        elif pii_type == "location":
            text = PII_TEMPLATES[pii_type].format(city=fake.city())
        elif pii_type == "medical":
            text = PII_TEMPLATES[pii_type].format(disease=random.choice(DISEASES))
    else:
        text = random.choice(NON_PII_TEMPLATES).format(word=fake.word())

    return {
        "query_text": text,
        "intent": intent,
        "pii_types": pii_types
    }