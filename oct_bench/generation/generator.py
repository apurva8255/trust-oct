import os
import json
import random
from typing import List, Dict, Any, Optional
from oct_bench.config import config
from oct_bench.data.schemas import ImageSample, MCQ
from oct_bench.data.renderer import OCTRenderer
from oct_bench.knowledge.database import KnowledgeDatabase
from .prompts import get_generation_prompt, SYSTEM_PROMPT

class VQAGenerator:
    """
    Orchestrates the generation of visual question-answering MCQs.
    Fuses rendered images, clinical knowledge, and metadata, passing them to an LLM
    or using a rule-based mock generator if LLM credentials are not available.
    """
    def __init__(self):
        self.renderer = OCTRenderer(bbox_colors=config.visualization.get("bbox_colors"))
        self.kb = KnowledgeDatabase()
        
        # Configure output paths
        self.output_dir = config.paths.get("generated_qa_dir", "./data/generated_qa")
        self.rendered_images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.rendered_images_dir, exist_ok=True)

        # Retrieve API configuration
        self.provider = config.generation.get("provider", "openai")
        self.model = config.generation.get("model", "gpt-4o")
        self.temperature = config.generation.get("temperature", 0.2)

        # Initialize clients if environment keys are present
        self.openai_client = None
        self.gemini_client = None
        
        if self.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            self.openai_client = OpenAI()
        elif self.provider == "gemini" and os.environ.get("GEMINI_API_KEY"):
            # Import google-genai client
            try:
                from google import genai
                self.gemini_client = genai.Client()
            except ImportError:
                print("Warning: google-genai library missing, fallback to mock generation.")

    def _call_llm(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Sends the system instruction and context prompt to the configured LLM API.
        """
        if self.provider == "openai" and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.temperature
                )
                res_text = response.choices[0].message.content
                return json.loads(res_text)
            except Exception as e:
                print(f"Error calling OpenAI API: {e}. Falling back to mock generator.")
                
        elif self.provider == "gemini" and self.gemini_client:
            try:
                # Use standard client.models.generate_content
                response = self.gemini_client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "response_mime_type": "application/json",
                        "temperature": self.temperature
                    }
                )
                return json.loads(response.text)
            except Exception as e:
                print(f"Error calling Gemini API: {e}. Falling back to mock generator.")
                
        return None

    def generate_mock_mcq(self, task_id: str, sample: ImageSample, rendered_img_path: str) -> Dict[str, Any]:
        """
        Deterministic rule-based mock MCQ generator used as a fallback if API keys are missing.
        """
        options = {}
        correct_answer = "D"
        explanation = ""
        question = ""
        disease = sample.disease_label
        
        # Build mock QAs based on Task IDs
        if task_id == "T01":
            question = "What medical imaging modality is demonstrated in the provided scan?"
            options = {"A": "MRI Scan", "B": "Ultrasound", "C": "CT Scan", "D": "Optical Coherence Tomography (OCT)"}
            correct_answer = "D"
            explanation = "The image presents high-resolution cross-sectional retinal layer bands, which is characteristic of an Optical Coherence Tomography (OCT) scan."
            
        elif task_id == "T06":
            count = len(sample.bboxes)
            question = "How many distinct pathological features are highlighted with annotation boxes in this OCT image?"
            options = {
                "A": str(max(0, count - 2)),
                "B": str(count + 2),
                "C": str(count + 1),
                "D": str(count)
            }
            correct_answer = "D"
            explanation = f"There are exactly {count} bounding boxes drawn, indicating the annotated lesions."

        elif task_id == "T09":
            # Retinal region is highlighted in our renderer test
            question = "Which anatomical region is highlighted in red in this OCT image?"
            options = {"A": "Vitreous cavity", "B": "Cornea", "C": "Choroid layer", "D": "Retina"}
            correct_answer = "D"
            explanation = "The highlighted red area covers the neural retinal band bounded by the ILM above and the RPE line below."

        elif task_id == "T12":
            lesion = "NORMAL"
            if sample.bboxes:
                lesion = sample.bboxes[0].label
            question = f"What type of retinal lesion is outlined by the bounding box in this image?"
            options = {"A": "Intraretinal Fluid (IRF)", "B": "Subretinal Fluid (SRF)", "C": "Subretinal Hyperreflective Material (SHRM)", "D": "Pigment Epithelial Detachment (PED)" if lesion == "PED" else lesion}
            # Auto-assign correct answer
            for k, v in options.items():
                if lesion in v:
                    correct_answer = k
            explanation = f"The lesion displays characteristic features matching {lesion} based on boundary separation and reflectivity."

        elif task_id == "T17":
            question = "Based on the structural changes observed in this OCT scan, what is the most likely clinical diagnosis?"
            options = {"A": "Healthy Retina", "B": "Diabetic Macular Edema (DME)", "C": "Central Serous Chorioretinopathy (CSC)", "D": f"Macular pathology ({disease})"}
            correct_answer = "D"
            explanation = f"The presence of structural anomalies and liquid accumulation matches criteria for {disease}."
            
        else:
            # Generic fallback
            question = f"Regarding the OCT scan with image ID {sample.image_id}, which statement is clinically valid?"
            options = {
                "A": "Vitreous detachment is complete.",
                "B": "The retina displays extensive hemorrhage.",
                "C": "Visual acuity is unaffected.",
                "D": f"The pathology matches criteria for {disease}."
            }
            correct_answer = "D"
            explanation = f"Features point towards neovascular leakage or structural changes characteristic of {disease}."

        return {
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "explanation": explanation
        }

    def generate_question(self, task_id: str, sample: ImageSample) -> MCQ:
        """
        Renders the annotated image required for the task, compiles prompt context, 
        calls LLM (or mock fallback), and returns an MCQ object.
        """
        # Create output image path specific to task (e.g. highlighted/numbered)
        filename = f"{sample.image_id}_{task_id}.png"
        rendered_img_path = os.path.join(self.rendered_images_dir, filename)
        
        # 1. Apply visual markup to the image matching task requirements
        if task_id in ["T06", "T07", "T08"]:
            # Draw numbered bounding boxes for counting/comparisons
            self.renderer.draw_numbered_bboxes(sample.image_path, [b.model_dump() for b in sample.bboxes], rendered_img_path)
        elif task_id == "T09":
            # Highlight Retina region in red
            self.renderer.highlight_anatomical_region(sample.image_path, "retina", rendered_img_path, color_rgb=(255, 0, 0), alpha=80)
        elif task_id == "T10":
            # Highlight RPE layer line
            self.renderer.highlight_retinal_layer(sample.image_path, "rpe", rendered_img_path, color_rgb=(0, 255, 0))
        else:
            # Standard annotation drawing
            self.renderer.draw_bboxes(sample.image_path, [b.model_dump() for b in sample.bboxes], rendered_img_path)
            
        # 2. Retrieve guidelines matching disease label
        rules = self.kb.get_knowledge_for_disease(sample.disease_label or "NORMAL")
        
        # 3. Create prompt
        prompt = get_generation_prompt(task_id, sample.model_dump(), rules)
        
        # 4. Generate QA content
        qa_data = self._call_llm(prompt)
        if not qa_data:
            # Fall back to mock generation
            qa_data = self.generate_mock_mcq(task_id, sample, rendered_img_path)
            
        # Save output image path (relative to repo root for webapp portability)
        qa_data["rendered_image_path"] = rendered_img_path.replace("\\", "/")
        
        # 5. Wrap in MCQ schema
        return MCQ(
            question_id=f"Q_{sample.image_id}_{task_id}",
            task_id=task_id,
            image_id=sample.image_id,
            question=qa_data["question"],
            options=qa_data["options"],
            correct_answer=qa_data["correct_answer"],
            explanation=qa_data.get("explanation"),
            metadata={
                "disease_label": sample.disease_label,
                "rendered_image_path": qa_data["rendered_image_path"],
                "dataset_source": sample.dataset_source,
                "original_image_path": sample.image_path.replace("\\", "/")
            }
        )

    def generate_benchmark(self, samples: List[ImageSample], tasks: List[str]) -> List[MCQ]:
        """
        Batch generates MCQs across multiple tasks and image samples, exporting the output dataset.
        """
        generated_mcqs = []
        for sample in samples:
            for task_id in tasks:
                print(f"Generating MCQ for {sample.image_id} - Task {task_id}...")
                mcq = self.generate_question(task_id, sample)
                generated_mcqs.append(mcq)
                
        # Save complete dataset
        out_path = os.path.join(self.output_dir, "dataset.json")
        with open(out_path, "w") as f:
            json.dump([q.model_dump() for q in generated_mcqs], f, indent=2)
            
        print(f"Completed! Generated {len(generated_mcqs)} questions. Saved to {out_path}")
        return generated_mcqs
