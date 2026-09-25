import os
import re
import random
from typing import Dict, Optional, Tuple

def build_eval_prompt(question: str, options: Dict[str, str], cot: bool = False) -> str:
    """
    Constructs evaluation prompt for zero-shot or clinical Chain-of-Thought (Med-CoT).
    """
    opt_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
    if not cot:
        return f"""Evaluate this OCT scan and answer the following multiple choice question:
{question}

Options:
{opt_str}

Respond with exactly one character representing the correct option (A, B, C, or D). Do not write explanations, markdown, or multiple letters."""

    return f"""You are an expert ophthalmic retinal specialist interpreting an Optical Coherence Tomography (OCT) scan.
Analyze this scan step-by-step using clinical medical reasoning (Clinical Chain-of-Thought):

STEP 1: [Visual Perception] Describe optical reflectivity (hyper/hyporeflective), structural boundaries, or fluid pockets.
STEP 2: [Anatomical Localization] Map observations to exact retinal layers (e.g., ILM, OPL, IS/OS, RPE, or vitreoretinal/subretinal space).
STEP 3: [Clinical Correlation] Correlate the visual evidence with ophthalmic consensus guidelines and disease manifestations.
STEP 4: [Conclusion] State your final chosen option letter.

Question: {question}

Options:
{opt_str}

Format your output EXACTLY as follows:
<CLINICAL_THINKING>
1. Visual Perception: [your visual findings]
2. Anatomical Localization: [layer/region identified]
3. Clinical Correlation: [differential analysis]
</CLINICAL_THINKING>
FINAL_ANSWER: [A, B, C, or D]"""

def extract_letter_and_reasoning(raw_text: str, default: str = "A") -> Tuple[str, str]:
    """
    Extracts the predicted option letter (A-D) and the Clinical CoT reasoning text.
    """
    reasoning = ""
    if "<CLINICAL_THINKING>" in raw_text and "</CLINICAL_THINKING>" in raw_text:
        start = raw_text.find("<CLINICAL_THINKING>") + len("<CLINICAL_THINKING>")
        end = raw_text.find("</CLINICAL_THINKING>")
        reasoning = raw_text[start:end].strip()
    elif "Visual Perception:" in raw_text or "1." in raw_text:
        parts = re.split(r"FINAL_ANSWER\s*:?", raw_text, flags=re.IGNORECASE)
        reasoning = parts[0].strip() if len(parts) > 1 else raw_text.strip()
    
    # Locate FINAL_ANSWER: X
    match = re.search(r"FINAL_ANSWER\s*:?\s*([A-D])", raw_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), reasoning
        
    # Fallback to direct letter search in trailing text
    tail = raw_text[-30:].upper()
    for letter in ["A", "B", "C", "D"]:
        if f"OPTION {letter}" in tail or f"ANSWER IS {letter}" in tail:
            return letter, reasoning
    for letter in reversed(["A", "B", "C", "D"]):
        if letter in tail:
            return letter, reasoning
            
    return default, reasoning

class BaseModelEvaluator:
    """
    Abstract interface for model execution on the MCQ benchmark.
    Supports both Direct Zero-Shot and Clinical Chain-of-Thought (Med-CoT) inference.
    """
    def predict(self, image_path: str, question: str, options: Dict[str, str], cot: bool = False) -> str:
        letter, _ = self.predict_with_reasoning(image_path, question, options, cot=cot)
        return letter

    def predict_with_reasoning(self, image_path: str, question: str, options: Dict[str, str], cot: bool = False) -> Tuple[str, str]:
        raise NotImplementedError("Each model evaluator must implement predict_with_reasoning()")

class RandomBaselineEvaluator(BaseModelEvaluator):
    """
    Random guess baseline (25% chance of correct answer).
    Used as validation control.
    """
    def predict_with_reasoning(self, image_path: str, question: str, options: Dict[str, str], cot: bool = False) -> Tuple[str, str]:
        chosen = random.choice(["A", "B", "C", "D"])
        if not cot:
            return chosen, ""
            
        chosen_text = options.get(chosen, "selected feature")
        simulated_cot = (
            f"1. Visual Perception: Scan demonstrates focal reflectivity alterations and structural boundary disruption.\n"
            f"2. Anatomical Localization: Centered primarily along the neurosensory retinal layers and RPE interface.\n"
            f"3. Clinical Correlation: Morphological presentation corresponds with {chosen_text} per ophthalmic guidelines."
        )
        return chosen, simulated_cot

class OpenAIEvaluator(BaseModelEvaluator):
    """
    Evaluator for OpenAI GPT vision models.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def predict_with_reasoning(self, image_path: str, question: str, options: Dict[str, str], cot: bool = False) -> Tuple[str, str]:
        import base64
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        prompt = build_eval_prompt(question, options, cot=cot)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name, #for ope ai (gpt - 4o)
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=350 if cot else 5,
                temperature=0.0
            )
            raw_text = response.choices[0].message.content.strip()
            return extract_letter_and_reasoning(raw_text)
        except Exception as e:
            print(f"OpenAI prediction error: {e}")
            return "A", f"Error: {e}"

class GeminiEvaluator(BaseModelEvaluator):
    """
    Evaluator for Gemini vision models using google-genai.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client()
        return self._client

    def predict_with_reasoning(self, image_path: str, question: str, options: Dict[str, str], cot: bool = False) -> Tuple[str, str]:
        from PIL import Image
        img = Image.open(image_path)
        prompt = build_eval_prompt(question, options, cot=cot)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[img, prompt]
            )
            raw_text = response.text.strip()
            return extract_letter_and_reasoning(raw_text)
        except Exception as e:
            print(f"Gemini prediction error: {e}")
            return "A", f"Error: {e}"

class ClaudeEvaluator(BaseModelEvaluator):
    """
    Evaluator for Anthropic Claude Vision models (Claude 3.5 Sonnet, Claude 3 Haiku).
    """
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022"):
        self.model_name = model_name
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def predict_with_reasoning(self, image_path: str, question: str, options: Dict[str, str], cot: bool = False) -> Tuple[str, str]:
        import base64
        prompt = build_eval_prompt(question, options, cot=cot)

        try:
            with open(image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")

            try:
                from anthropic import Anthropic
                client = Anthropic()
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=350 if cot else 10,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_data}},
                            {"type": "text", "text": prompt}
                        ]
                    }]
                )
                raw_text = response.content[0].text.strip()
            except ImportError:
                import json
                import urllib.request
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=json.dumps({
                        "model": self.model_name,
                        "max_tokens": 350 if cot else 10,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_data}},
                                {"type": "text", "text": prompt}
                            ]
                        }]
                    }).encode("utf-8"),
                    headers={
                        "x-api-key": self.api_key or "",
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    }
                )
                with urllib.request.urlopen(req, timeout=30) as res:
                    data = json.loads(res.read().decode("utf-8"))
                    raw_text = data["content"][0]["text"].strip()

            return extract_letter_and_reasoning(raw_text)
        except Exception as e:
            print(f"Claude prediction error: {e}")
            return "A", f"Error: {e}"

class OllamaVisionEvaluator(BaseModelEvaluator):
    """
    Evaluator for local open-source Vision models via Ollama (e.g. LLaVA, LLaMA 3.2-Vision, Qwen2-VL).
    Completely offline and free - requires no cloud API keys.
    """
    def __init__(self, model_name: str = "llava", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host.rstrip("/")

    def predict_with_reasoning(self, image_path: str, question: str, options: Dict[str, str], cot: bool = False) -> Tuple[str, str]:
        import base64
        import json
        import urllib.request

        prompt = build_eval_prompt(question, options, cot=cot)

        try:
            with open(image_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [base64_data],
                "stream": False,
                "options": {"temperature": 0.0}
            }
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as res:
                data = json.loads(res.read().decode("utf-8"))
                raw_text = data.get("response", "").strip()

            return extract_letter_and_reasoning(raw_text)
        except Exception as e:
            print(f"Ollama ({self.model_name}) prediction error: {e}")
            return "A", f"Error: {e}"

class HuggingFaceVisionEvaluator(BaseModelEvaluator):
    """
    Evaluator for open-source Vision models using Hugging Face Inference API
    (e.g., Qwen/Qwen2-VL-7B-Instruct, llava-hf/llava-1.5-7b-hf).
    """
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.model_name = model_name
        self.token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    def predict_with_reasoning(self, image_path: str, question: str, options: Dict[str, str], cot: bool = False) -> Tuple[str, str]:
        import base64
        prompt = build_eval_prompt(question, options, cot=cot)

        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(api_key=self.token)
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }
            ]
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=350 if cot else 10
            )
            raw_text = response.choices[0].message.content.strip()
            return extract_letter_and_reasoning(raw_text)
        except Exception as e:
            print(f"HuggingFace ({self.model_name}) error: {e}")
            return "A", f"Error: {e}"

class GroqVisionEvaluator(BaseModelEvaluator):
    """
    Evaluator for ultra-fast LLaMA 3.2 Vision models hosted on Groq.
    """
    def __init__(self, model_name: str = "llama-3.2-11b-vision-preview"):
        self.model_name = model_name
        self.api_key = os.environ.get("GROQ_API_KEY")

    def predict_with_reasoning(self, image_path: str, question: str, options: Dict[str, str], cot: bool = False) -> Tuple[str, str]:
        import base64
        from openai import OpenAI
        prompt = build_eval_prompt(question, options, cot=cot)

        try:
            with open(image_path, "rb") as f:
                b64_img = base64.b64encode(f.read()).decode("utf-8")

            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                        ]
                    }
                ],
                max_tokens=350 if cot else 5,
                temperature=0.0
            )
            raw_text = response.choices[0].message.content.strip()
            return extract_letter_and_reasoning(raw_text)
        except Exception as e:
            print(f"Groq ({self.model_name}) error: {e}")
            return "A", f"Error: {e}"

def get_evaluator(name: str = "random", model_variant: Optional[str] = None) -> BaseModelEvaluator:
    """
    Factory helper to instantiate any model evaluator by name.
    Supported: 'openai', 'gemini', 'claude', 'ollama', 'llava', 'huggingface', 'qwen', 'groq', 'random'
    """
    key = name.lower().strip()
    if "openai" in key or "gpt" in key:
        return OpenAIEvaluator(model_name=model_variant or "gpt-4o")
    elif "gemini" in key:
        return GeminiEvaluator(model_name=model_variant or "gemini-2.5-flash")
    elif "claude" in key or "anthropic" in key:
        return ClaudeEvaluator(model_name=model_variant or "claude-3-5-sonnet-20241022")
    elif "groq" in key:
        return GroqVisionEvaluator(model_name=model_variant or "llama-3.2-11b-vision-preview")
    elif "ollama" in key or "llava" in key or "local" in key:
        return OllamaVisionEvaluator(model_name=model_variant or "llava")
    elif "hf" in key or "huggingface" in key or "qwen" in key:
        return HuggingFaceVisionEvaluator(model_name=model_variant or "Qwen/Qwen2-VL-7B-Instruct")
    else:
        return RandomBaselineEvaluator()
