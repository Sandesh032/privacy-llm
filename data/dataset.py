"""
Research-Grade Dataset Generator for Colab
Uses LLM paraphrasing for natural language generation
"""

import random
import json
import uuid
import os
import torch
from faker import Faker
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

fake = Faker()

# ============================================
# LLM-based Query Generator (Colab Optimized)
# ============================================

class QueryGenerator:
    def __init__(self, model_name="google/gemma-2b-it"):
        print(f"Loading {model_name} for query generation...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Use device_map="auto" to handle offloading automatically
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=32, # Single sample for 4GB GPU
            pad_token_id=self.tokenizer.pad_token_id
        )
        print(f"Model loaded successfully")

    def create_prompt(self, intent, pii_type=None, pii_value=None):
        """Create the prompt string without generating"""
        if pii_type and pii_value:
            return f"""Generate a realistic, casual user query for an AI chatbot.
Requirements:
- Intent: {intent}
- Must naturally include: {pii_type} = {pii_value}
- Make it conversational (use contractions, casual tone)
- 1-2 sentences maximum
- Output ONLY the query, nothing else

Query:"""
        else:
            return f"""Generate a realistic, casual user query for an AI chatbot.
Requirements:
- Intent: {intent}
- Must NOT contain any personal/private information
- Make it conversational
- 1-2 sentences maximum
- Output ONLY the query, nothing else

Query:"""

    def generate_batch(self, prompts):
        """Generate multiple queries at once with rate limiting"""
        import time
        results = []
        
        for i, prompt in enumerate(prompts):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=prompt,
                        config={
                            'temperature': 0.9,
                            'max_output_tokens': 50,
                        }
                    )
                    query = response.text.strip()
                    query = query.strip('"').strip("'").strip()
                    
                    if len(query) < 5 or len(query) > 300:
                        results.append(None)
                    else:
                        results.append(query)
                    
                    # IMPORTANT: Sleep 4 seconds between requests
                    # 60 seconds / 15 requests = 4 seconds per request
                    # This keeps us safely under 15 requests/minute
                    time.sleep(4)
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                        retry_count += 1
                        wait_time = 60  # Wait 1 minute before retrying
                        if retry_count < max_retries:
                            print(f"\nRate limit hit, waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                            time.sleep(wait_time)
                        else:
                            print(f"\nMax retries reached, skipping this request")
                            results.append(None)
                            break
                    else:
                        # Other errors, don't retry
                        results.append(None)
                        break
        return results

# ============================================
# Google Gemini API Query Generator (Free & Fast)
# ============================================

class GeminiAPIQueryGenerator:
    """Fast query generation using Google Gemini API (Free tier available)"""
    def __init__(self):
        try:
            from google import genai
            from google.genai import types
            
            api_key = os.getenv("GOOGLE_API_KEY")
            client = genai.Client(api_key=api_key)
            self.client = client
            
            # Try to find a working model
            available_models = [
                "models/gemini-2.5-flash",
                "models/gemini-2.0-flash",
                "models/gemini-flash-latest",
            ]
            
            self.model_id = None
            for model_name in available_models:
                try:
                    # Test if model works
                    test_response = client.models.generate_content(
                        model=model_name,
                        contents="test",
                        config={'max_output_tokens': 5}
                    )
                    self.model_id = model_name
                    print(f"Using Google Gemini API: {model_name} (Free tier)")
                    break
                except Exception as e:
                    continue
            
            if not self.model_id:
                raise Exception("No working Gemini model found. Please check API key and model availability.")
                
        except ImportError:
            raise ImportError("Please install: pip install google-genai")
        except Exception as e:
            raise Exception(f"Failed to initialize Gemini API: {e}")
    
    def create_prompt(self, intent, pii_type=None, pii_value=None):
        """Create the prompt string without generating"""
        if pii_type and pii_value:
            return f"Write one natural user question to an AI assistant. Task: {intent}. Must include {pii_type}: {pii_value}. Write conversationally, like texting a friend. Query:"
        else:
            return f"Write one natural user question to an AI assistant. Task: {intent}. No personal information. Write conversationally, like texting a friend. Query:"
    
    def generate_batch(self, prompts):
        """Generate multiple queries at once"""
        results = []
        for prompt in prompts:
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config={
                        'temperature': 0.9,
                        'max_output_tokens': 200
                    }
                )
                
                # Try to extract text
                query = None
                if hasattr(response, 'text') and response.text:
                    query = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content'):
                        if hasattr(candidate.content, 'parts'):
                            query = candidate.content.parts[0].text
                
                if not query:
                    print(f"Warning: Empty response from API")
                    results.append(None)
                    continue
                
                # Clean up
                query = query.strip().strip('"').strip("'").strip()
                
                if len(query) < 5 or len(query) > 300:
                    results.append(None)
                else:
                    results.append(query)
                    
            except Exception as e:
                print(f"API error: {e}")
                results.append(None)
        return results

# Initialize generator (do this once)
generator = None

def get_generator():
    global generator
    if generator is None:
        # Use Gemini API if key exists, otherwise local model
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and api_key.strip():
            try:
                generator = GeminiAPIQueryGenerator()
            except Exception as e:
                print(f"Failed to initialize Gemini API: {e}")
                print("Falling back to local model...")
                generator = QueryGenerator("google/gemma-2b-it")
        else:
            print("No GOOGLE_API_KEY found, using local model...")
            generator = QueryGenerator("google/gemma-2b-it")
    return generator

# ============================================
# PII Templates and Data
# ============================================

INTENTS = [
    # Information & Knowledge
    "explain a complex concept", "answer a trivia question", "define a technical term",
    "summarize recent news", "compare two products", "recommend a book/movie",
    "translate a phrase", "explain a math problem", "describe a historical event",
    
    # Productivity & Tasks
    "draft an email", "write a business letter", "create a meeting agenda",
    "generate code snippet", "debug programming error", "optimize SQL query",
    "format a resume", "write a cover letter", "create a presentation outline",
    
    # Creative & Personal
    "write a creative story", "compose a poem", "suggest gift ideas",
    "plan a vacation itinerary", "generate recipe ideas", "suggest workout routine",
    "write a social media post", "create a birthday message", "plan a date night",
    
    # Health & Wellness
    "explain medical symptoms", "suggest healthy meal plans", "describe mental health tips",
    "interpret lab test results", "explain medication side effects", "suggest stress relief techniques",
    
    # Financial & Business
    "explain investment strategies", "calculate loan payments", "interpret financial reports",
    "suggest budgeting tips", "explain tax deductions", "analyze market trends",
    "draft business proposal", "calculate ROI", "explain crypto concepts",
    
    # Education & Learning
    "explain scientific theory", "solve chemistry problem", "teach grammar rules",
    "explain physics concepts", "tutor math equations", "explain programming concepts",
    "summarize research paper", "explain statistical analysis", "teach language vocabulary",
    
    # Daily Life & Entertainment
    "suggest restaurants nearby", "find movie showtimes", "check weather forecast",
    "plan party activities", "suggest weekend activities", "recommend music playlists",
    "find local events", "suggest date ideas", "plan family activities"
]

PII_TEMPLATES = {
    "email": "Send the report to {email}",
    "phone": "Call me at {phone}",
    "location": "Find restaurants near {city}",
    "medical": "I have symptoms of {disease}",
}

DISEASES = [
    "diabetes", "hypertension", "asthma", "depression", "anxiety",
    "migraine", "arthritis", "covid19", "heart_disease", "cancer",
    "insomnia", "obesity", "epilepsy", "ptsd", "bipolar_disorder"
]

NON_PII_TEMPLATES = [
    "Tell me about {word}",
    "What is the definition of {word}?",
    "Can you explain how {word} works?",
    "Search for the history of {word}",
]

PII_SENSITIVITY = {
    "email": 0.5,
    "location": 0.55,
    "phone": 0.65,
    "medical": 0.75
}

NETWORK_MAP = {"wifi": 0, "4g": 1, "5g": 2}

# ============================================
# Hard Negatives
# ============================================

HARD_NEGATIVES = [
    # Technical queries that mention PII concepts but don't contain actual PII
    ("How do I validate email addresses in Python regex?", [], "explain programming concepts"),
    ("What's the best format for storing phone numbers in a database?", [], "explain programming concepts"),
    ("Explain how GPS location tracking works in mobile apps", [], "explain a technical term"),
    ("What is HIPAA compliance for medical records?", [], "explain a complex concept"),
    ("How does end-to-end email encryption protect privacy?", [], "explain a complex concept"),
    
    # Queries with indirect/vague PII (harder to classify)
    ("I'm at the downtown Starbucks, can you find parking nearby?", ["location"], "suggest restaurants nearby"),
    ("My doctor said my cholesterol is 240, is that concerning?", ["medical"], "interpret lab test results"),
    ("The apartment I'm renting is in postal code 90210, what's the crime rate?", ["location"], "answer a trivia question"),
    ("Someone with type 2 diabetes asked me for diet advice, what should I tell them?", [], "suggest healthy meal plans"),
    ("My test results show elevated liver enzymes, what could cause that?", ["medical"], "explain medical symptoms"),
    
    # Edge cases with context-dependent PII
    ("I live in California, what are the state tax rates?", ["location"], "explain tax deductions"),
    ("Draft an email to my manager asking for time off next week", [], "draft an email"),
    ("Write a cover letter for a software engineer position at Google", [], "write a cover letter"),
    ("My kid's school is in District 5, what are the vaccination requirements?", ["location"], "explain medical symptoms"),
    ("Calculate mortgage payment for a $500k house in my area", ["location"], "calculate loan payments"),
]

# ============================================
# Generation Functions
# ============================================

def generate_prompt(use_llm=True, llm_ratio=0.8):
    """Generate a query with optional PII"""

    intent = random.choice(INTENTS)
    inject_pii = random.random() < 0.6

    pii_types = []
    text = ""
    generation_method = "template"

    # Try LLM generation
    if use_llm and random.random() < llm_ratio:
        try:
            gen = get_generator()

            if inject_pii:
                pii_type = random.choice(list(PII_TEMPLATES.keys()))
                pii_types.append(pii_type)

                # Generate realistic PII value
                if pii_type == "email":
                    pii_value = fake.email()
                elif pii_type == "phone":
                    pii_value = fake.phone_number()
                elif pii_type == "location":
                    pii_value = fake.city()
                elif pii_type == "medical":
                    pii_value = random.choice(DISEASES).replace("_", " ")

                text = gen.generate(intent, pii_type, pii_value)
                if text:
                    generation_method = "llm"
            else:
                text = gen.generate(intent)
                if text:
                    generation_method = "llm"
        except Exception as e:
            print(f"LLM generation error: {e}")
            text = None

    # Fallback to templates
    if not text:
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

        generation_method = "template"

    return {
        "query_text": text,
        "intent": intent,
        "pii_types": pii_types,
        "generation_method": generation_method
    }

def compute_privacy_risk(pii_types):
    """Compute privacy risk score"""
    if not pii_types:
        return 0.05
    return sum(PII_SENSITIVITY[p] for p in pii_types) / len(pii_types)

def simulate_device():
    """Simulate device characteristics"""
    return {
        "battery_level": round(random.uniform(0.15, 1.0), 2),
        "cpu_load": round(random.uniform(0.1, 0.9), 2),
        "ram_mb": random.choice([2048, 4096, 8192]),
        "network": random.choice(["wifi", "4g", "5g"])
    }

def simulate_energy_latency(device):
    """Simulate energy and latency"""
    net = device["network"]
    latency = {
        "wifi": random.uniform(20, 60),
        "4g": random.uniform(60, 140),
        "5g": random.uniform(15, 40)
    }[net]

    return {
        "latency_ms": round(latency, 2),
        "tx_energy": round(0.2 + latency / 200, 3),
        "local_energy": round(0.3 + device["cpu_load"] * 0.4, 3)
    }

def evaluate_routes(pii_risk, energy):
    """Evaluate routing options with balanced competitiveness for all three classes"""
    return {
        "local": {
            "privacy_leakage": 0.12 + 0.10 * pii_risk,  # Low-medium privacy
            "energy_cost": energy["local_energy"] * 0.9,  # Some energy advantage
            "task_quality": min(0.70 + random.uniform(-0.03, 0.03), 1.0)  # Decent quality
        },
        "hybrid": {
            "privacy_leakage": 0.30 + 0.25 * pii_risk,  # Medium privacy
            "energy_cost": (energy["local_energy"] * 0.6 + energy["tx_energy"] * 0.4),  # Balanced energy
            "task_quality": min(0.88 + random.uniform(-0.02, 0.02), 1.0)  # Good quality
        },
        "cloud": {
            "privacy_leakage": 0.55 + 0.40 * pii_risk,  # Higher privacy cost
            "energy_cost": energy["tx_energy"] * 0.85,  # Network efficiency
            "task_quality": min(0.96 + random.uniform(-0.01, 0.01), 1.0)  # Best quality
        }
    }

def choose_best(routes, alpha=None, beta=None):
    """Choose optimal route with dynamic weights"""
    if alpha is None:
        alpha = random.uniform(0.3, 0.7)  # Privacy weight
    if beta is None:
        beta = random.uniform(0.3, 0.7)  # Energy weight
    
    best, best_score = None, -1e9
    for r, v in routes.items():
        score = v["task_quality"] - alpha * v["privacy_leakage"] - beta * v["energy_cost"]
        if score > best_score:
            best, best_score = r, score
    return best

# ============================================
# Targeted Device Generation for Balanced Dataset
# ============================================

def simulate_device_for_target(target_class):
    """Generate device conditions that favor a specific routing class"""
    
    if target_class == "local":
        # Local favored by: high privacy risk, good battery, low CPU load
        return {
            "battery_level": round(random.uniform(0.6, 1.0), 2),
            "cpu_load": round(random.uniform(0.1, 0.5), 2),
            "ram_mb": random.choice([4096, 8192]),
            "network": random.choice(["wifi", "5g"])
        }
    
    elif target_class == "hybrid":
        # Hybrid favored by: moderate everything
        return {
            "battery_level": round(random.uniform(0.4, 0.7), 2),
            "cpu_load": round(random.uniform(0.4, 0.7), 2),
            "ram_mb": random.choice([4096, 8192]),
            "network": random.choice(["4g", "5g"])
        }
    
    elif target_class == "cloud":
        # Cloud favored by: low privacy risk, low battery OR high CPU load
        return {
            "battery_level": round(random.uniform(0.15, 0.5), 2),
            "cpu_load": round(random.uniform(0.6, 0.9), 2),
            "ram_mb": random.choice([2048, 4096]),
            "network": random.choice(["wifi", "5g"])
        }
    
    else:
        # Fallback to random
        return simulate_device()

def compute_privacy_risk_for_target(pii_types, target_class):
    """Compute privacy risk biased toward target class"""
    base_risk = compute_privacy_risk(pii_types)
    
    # Adjust based on target
    if target_class == "local":
        # Local likes HIGH privacy risk
        return min(base_risk + random.uniform(0.1, 0.3), 1.0)
    elif target_class == "cloud":
        # Cloud likes LOW privacy risk
        return max(base_risk - random.uniform(0.1, 0.3), 0.0)
    else:
        # Hybrid likes MODERATE privacy risk
        return base_risk

# ============================================
# Main Dataset Generation
# ============================================

def generate_research_dataset(
    output_path="local_dataset.jsonl",
    target_per_class=5000,
    use_llm=True,
    llm_ratio=0.8,
    hard_neg_ratio=0.1
):
    print("="*80)
    print("BATCHED DATASET GENERATION STARTED (Targeted Balanced Approach)")
    print("="*80)
    
    counts = {"local": 0, "hybrid": 0, "cloud": 0}
    gen_methods = {"template": 0, "llm": 0, "hard_negative": 0}
    attempts = {"local": 0, "hybrid": 0, "cloud": 0}
    
    gen = get_generator()
    
    # Auto-detect batch size based on generator typeGeminiAPIQueryGenerator
    if isinstance(gen, GeminiAPIQueryGenerator):
        BATCH_SIZE = 2  # Gemini API can handle multiple in parallel
        print("Using Google Gemini API - Fast mode enabled (Free tier)")
    else:
        BATCH_SIZE = 1  # Local model on 4GB GPU
        print("Using local model")
    
    batch_buffer = [] 
    batch_prompts = []

    print(f"Target per class: {target_per_class:,}")
    print(f"Total samples: {target_per_class * 3:,}")
    print(f"LLM enabled: {use_llm} ({llm_ratio*100:.0f}% ratio)")
    print(f"Hard negatives: {hard_neg_ratio*100:.0f}%")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*80)
    
    total_batches = 0
    max_consecutive_failures = 10000  # Safety limit per class

    with open(output_path, "w") as f:
        pbar = tqdm(total=target_per_class * 3, desc="Generating dataset")
        
        while min(counts.values()) < target_per_class:
            
            # Find which class needs more samples
            needed_class = min(counts, key=counts.get)
            
            # Safety check
            if attempts[needed_class] > max_consecutive_failures:
                print(f"\nWARNING: Could not generate {needed_class} samples after {max_consecutive_failures} attempts")
                print(f"   Current counts: {counts}")
                print(f"   Try adjusting evaluate_routes() or reduce target_per_class")
                break
            
            # --- 1. Fill the Batch (targeting needed class) ---
            while len(batch_prompts) < BATCH_SIZE:
                attempts[needed_class] += 1
                
                # Adjust PII probability based on target class
                if needed_class == "local":
                    inject_pii = random.random() < 0.8  # High PII for local
                elif needed_class == "cloud":
                    inject_pii = random.random() < 0.2  # Low PII for cloud
                else:  # hybrid
                    inject_pii = random.random() < 0.5  # Moderate PII
                
                intent = random.choice(INTENTS)
                pii_types = []
                pii_value = None
                
                if inject_pii:
                    pii_type = random.choice(list(PII_TEMPLATES.keys()))
                    pii_types.append(pii_type)
                    if pii_type == "email": pii_value = fake.email()
                    elif pii_type == "phone": pii_value = fake.phone_number()
                    elif pii_type == "location": pii_value = fake.city()
                    elif pii_type == "medical": pii_value = random.choice(DISEASES).replace("_", " ")
                
                method = "template"
                prompt_text = None
                
                if use_llm and random.random() < llm_ratio:
                    method = "llm"
                    if inject_pii:
                        prompt_text = gen.create_prompt(intent, pii_types[0], pii_value)
                    else:
                        prompt_text = gen.create_prompt(intent)
                
                if method == "llm":
                    batch_prompts.append(prompt_text)
                    batch_buffer.append({
                        "intent": intent,
                        "pii_types": pii_types,
                        "generation_method": "llm",
                        "pii_value": pii_value,
                        "target_class": needed_class  # Track target
                    })
                else:
                    # Template generation
                    text = ""
                    if inject_pii:
                        pt = pii_types[0]
                        val = pii_value if pii_value else (fake.email() if pt=="email" else fake.phone_number() if pt=="phone" else fake.city() if pt=="location" else random.choice(DISEASES))
                        if pt == "medical": text = f"I have symptoms of {val}"
                        elif pt == "location": text = f"Find restaurants near {val}"
                        else: text = PII_TEMPLATES[pt].format(**{pt: val})
                    else:
                        text = random.choice(NON_PII_TEMPLATES).format(word=fake.word())
                    
                    # Process with targeted device generation
                    saved = process_item_targeted(text, intent, pii_types, "template", counts, 
                                                  target_per_class, f, pbar, gen_methods, 
                                                  needed_class, attempts)
                    if saved:
                        break  # Exit batch filling to check if we still need this class

            # --- 2. Process the Batch (GPU) ---
            if batch_prompts:
                total_batches += 1
                if total_batches % 10 == 0: # Log every 10 batches to avoid spam
                    print(f"\n[Batch {total_batches}] Processing {len(batch_prompts)} items on GPU...")
                    print(f"   Target: {needed_class}, Counts: {counts}")
                
                generated_texts = gen.generate_batch(batch_prompts)
                
                saved_count = 0
                for i, text in enumerate(generated_texts):
                    meta = batch_buffer[i]
                    final_text = text
                    
                    if not final_text:
                        meta["generation_method"] = "template"
                        final_text = f"Tell me about {meta['intent']}" 
                    
                    # Use targeted processing
                    saved = process_item_targeted(final_text, meta["intent"], meta["pii_types"], 
                                                 meta["generation_method"], counts, target_per_class, 
                                                 f, pbar, gen_methods, meta["target_class"], attempts)
                    if saved: 
                        saved_count += 1
                
                if total_batches % 10 == 0:
                     print(f"[Batch {total_batches}] Done. Saved {saved_count} new samples. Total: {sum(counts.values())}")
                     print(f"    Attempts so far: local={attempts['local']}, hybrid={attempts['hybrid']}, cloud={attempts['cloud']}")

                batch_prompts = []
                batch_buffer = []

    pbar.close()
    print(f"\nFinished! Total batches processed: {total_batches}")
    print(f"Final counts: {counts}")
    print(f"Generation methods: {gen_methods}")
    return counts, gen_methods

def process_item_targeted(text, intent, pii_types, method, counts, target, f, pbar, gen_methods, 
                          target_class, attempts):
    """Helper to calculate scores with targeted device generation. Returns True if saved."""
    # Use targeted device generation
    device = simulate_device_for_target(target_class)
    energy = simulate_energy_latency(device)
    
    # Use targeted privacy risk
    privacy_risk = compute_privacy_risk_for_target(pii_types, target_class)
    
    routes = evaluate_routes(privacy_risk, energy)
    
    # Calculate scores for all routes with dynamic alpha/beta
    alpha = random.uniform(0.3, 0.7)
    beta = random.uniform(0.3, 0.7)
    
    scores = {
        r: v["task_quality"] - alpha * v["privacy_leakage"] - beta * v["energy_cost"]
        for r, v in routes.items()
    }
    
    optimal_route = max(scores, key=scores.get)
    optimal_score = scores[optimal_route]
    second_best_score = sorted(scores.values(), reverse=True)[1]
    margin = optimal_score - second_best_score
    
    # DEBUG: Print first 20 attempts to diagnose issues
    total_samples = sum(counts.values())
    if total_samples < 20 or attempts[target_class] % 100 == 0:
        print(f"\n  [DEBUG] Attempt {attempts[target_class]} for {target_class}:")
        print(f"    Target: {target_class}, Got: {optimal_route}, Margin: {margin:.4f}")
        print(f"    Scores: local={scores['local']:.3f}, hybrid={scores['hybrid']:.3f}, cloud={scores['cloud']:.3f}")
        print(f"    Privacy risk: {privacy_risk:.3f}, Battery: {device['battery_level']:.2f}, CPU: {device['cpu_load']:.2f}")
        if margin < 0.01:
            print(f"    REJECTED: Margin {margin:.4f} < 0.01")
        elif optimal_route != target_class:
            print(f"    REJECTED: Got {optimal_route}, needed {target_class}")
        else:
            print(f"    ACCEPTED")
    
    # SKIP samples with too small margins (reduces label noise)
    if margin < 0.01:  # Lowered from 0.03 for better generation speed
        return False
    
    # Check if we got the target class
    if optimal_route == target_class and counts[optimal_route] < target:
        attempts[target_class] = 0  # Reset attempts on success
        
        record = {
            "id": str(uuid.uuid4()),
            "query_text": text,
            "intent": intent,
            "pii_types": pii_types,
            "privacy_risk": round(privacy_risk, 3),
            "generation_method": method,
            "device": device,
            "energy": energy,
            "routes": routes,
            "optimal_route": optimal_route,
            "label": ["local", "hybrid", "cloud"].index(optimal_route)
        }
        f.write(json.dumps(record) + "\n")
        counts[optimal_route] += 1
        gen_methods[method] += 1
        pbar.update(1)
        pbar.set_postfix({
            "local": counts["local"], 
            "hybrid": counts["hybrid"], 
            "cloud": counts["cloud"]
        })
        return True
    
    return False