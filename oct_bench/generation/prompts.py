# Prompts and instructions for generating OCT-Bench task questions (T01 - T20)

SYSTEM_PROMPT = """You are a clinical expert in ophthalmology and optical coherence tomography (OCT) imaging.
Your goal is to generate a high-quality, clinically accurate multiple-choice question (MCQ) based on the provided OCT image context, annotation metadata, and clinical knowledge.

You MUST output your response in valid JSON format matching this schema:
{
  "question": "Clear and concise question text based on the image/annotations.",
  "options": {
    "A": "Option A text",
    "B": "Option B text",
    "C": "Option C text",
    "D": "Option D text"
  },
  "correct_answer": "A", // Must be one of: A, B, C, or D
  "explanation": "Detailed explanation of why the correct option is correct and why other options are incorrect based on the image context and clinical guidelines."
}

Rules:
1. Provide exactly 4 options (A, B, C, D).
2. Ensure there is EXACTLY ONE clearly correct answer. The other three options must be plausible clinical distractors but clearly incorrect based on the provided image evidence and clinical facts.
3. The question must be answerable by viewing the annotated image (grounded). Avoid generating questions that can be answered through pure textual knowledge alone unless it is a clinical reasoning question that integrates the visual findings.
4. Do not include any language cues or shortcuts that reveal the correct answer.
"""

TASK_PROMPTS = {
    # PERCEPTION TASKS
    "T01": {
        "name": "Modality Perception",
        "instruction": "Ask the user to identify the imaging modality shown in the image. Correct answer is D: OCT (Optical Coherence Tomography). Distractors should be other common medical imaging modalities (e.g., MRI, CT, Ultrasound, Fundus Photography)."
    },
    "T02": {
        "name": "Annotation Recognition",
        "instruction": "Ask a question about the properties of the annotations drawn on the image (e.g., box count, colors, or annotation markers). Refer to the color of the bounding boxes provided in the image metadata."
    },
    "T03": {
        "name": "Morphological Description",
        "instruction": "Describe the morphological shape pattern of the lesion located inside the annotation box (e.g., dot-like, mass-like, elevated, flat, dome-shaped). Check the annotation description for morphologic clues."
    },
    "T04": {
        "name": "Boundary Feature Recognition",
        "instruction": "Analyze the boundary characteristics of the lesion inside the annotation box (e.g., smooth, irregular, jagged, poorly defined, absent). Ask about the border characteristic."
    },
    "T05": {
        "name": "Reflectivity Analysis",
        "instruction": "Describe the reflectivity pattern of the highlighted region or lesion inside the box. Ground it in OCT reflectivity terms: hyperreflective (bright), hyporeflective (dark), iso-reflective (similar to surrounding tissue), or heterogeneous."
    },
    "T06": {
        "name": "Quantity Estimation",
        "instruction": "Ask a quantitative question about the annotations in the image. For example: 'How many annotated regions are shown in this image?' Correct answer must match the actual bounding box count."
    },
    "T07": {
        "name": "Scale Perception",
        "instruction": "Compare the size, width, or height of different numbered annotation boxes. For example: 'Which annotation box is larger, Box 1 or Box 2?' Or 'What is the approximate size relationship between Box 1 and Box 2?'"
    },
    "T08": {
        "name": "Spatial Orientation Recognition",
        "instruction": "Determine the relative spatial position of one numbered annotation box relative to another. Example: 'Where is Box 1 located relative to Box 2?' (A) Upper-Left, (B) Upper-Right, (C) Lower-Left, (D) Lower-Right."
    },
    
    # COGNITION TASKS
    "T09": {
        "name": "Region Identification",
        "instruction": "Identify which macro-anatomical region is highlighted by the color mask in the image. Options must include: Vitreous, Retina, Choroid, Cornea."
    },
    "T10": {
        "name": "Layer Identification",
        "instruction": "Identify which specific fine retinal layer is highlighted by the colored line. Options must include: ILM (Inner Limiting Membrane), OPL (Outer Plexiform Layer), IS/OS (Inner/Outer Segment junction), RPE (Retinal Pigment Epithelium)."
    },
    "T11": {
        "name": "Inter-layer Relationship",
        "instruction": "Ask about the relative spatial order of retinal layers. Example: 'Which retinal layer lies immediately below the highlighted layer line?' Options: ILM, OPL, IS/OS, RPE."
    },
    "T12": {
        "name": "Lesion Classification",
        "instruction": "Classify the specific type of lesion highlighted in the image. Options must cover clinical labels: IRF (Intraretinal Fluid), SRF (Subretinal Fluid), PED (Pigment Epithelial Detachment), SHRM (Subretinal Hyperreflective Material)."
    },
    "T13": {
        "name": "Structural Status Assessment",
        "instruction": "Evaluate the structural integrity of the macula/retina shown in the image. Options: Intact, Mildly irregular, Locally disrupted, Severely disrupted."
    },
    "T14": {
        "name": "Lesion Localization",
        "instruction": "Determine the anatomical localization of the lesion in the bounding box relative to retinal boundaries. Options: Inner retina, Outer retina, Subretinal space, Vitreous cavity, Sub-RPE space."
    },
    "T15": {
        "name": "Disease Association",
        "instruction": "Link the observed pathological findings (e.g. cystic spaces or dome-shaped RPE elevation) to its most typically associated disease condition. Options: DME (Diabetic Macular Edema), CSC (Central Serous Chorioretinopathy), Wet AMD, Macular Hole."
    },
    "T16": {
        "name": "Functional Impact Assessment",
        "instruction": "Given the pathology shown (e.g. macular hole, subretinal fluid at fovea), ask what functional vision abnormality is most likely caused by this lesion. Options: Reduced central vision, Corneal opacity, Peripheral field loss, Increased intraocular pressure."
    },
    
    # REASONING TASKS
    "T17": {
        "name": "Disease Diagnosis",
        "instruction": "Establish a clinical diagnosis based on the overall findings in the OCT image. Options: AMD, CSC, DME, MH."
    },
    "T18": {
        "name": "Stage Classification",
        "instruction": "Determine the clinical stage or severity of the disease demonstrated in the image (e.g., macular hole stage 1 to 4, or dry vs wet AMD). Ground the correct option in the clinical guidelines provided."
    },
    "T19": {
        "name": "Treatment Planning",
        "instruction": "Select the appropriate treatment strategy or diagnosis-treatment pair matching the visual findings. Options must incorporate anti-VEGF therapy, PDT (Photodynamic Therapy), vitrectomy + gas tamponade + ILM peeling, or conservative observation."
    },
    "T20": {
        "name": "Follow-up Adjustment",
        "instruction": "Formulate a clinical follow-up adjustment based on the persistent fluid presence and patient symptoms described. Connect visual persistence to management strategies (e.g. active intervention with PDT vs switching anti-VEGF agent vs urgent vitrectomy)."
    }
}

def get_generation_prompt(task_id: str, image_metadata: dict, clinical_rules: dict) -> str:
    """
    Constructs the prompt for the MLLM by combining task instructions, image annotations, 
    and clinical knowledge constraints.
    """
    task = TASK_PROMPTS.get(task_id)
    if not task:
        raise ValueError(f"Unknown task ID: {task_id}")
        
    prompt = f"""
### TASK DEFINITION
Task ID: {task_id}
Task Name: {task['name']}
Instruction: {task['instruction']}

### IMAGE CONTEXT & METADATA
Image ID: {image_metadata.get('image_id')}
Source Dataset: {image_metadata.get('dataset_source')}
Image Dimensions: {image_metadata.get('width')}x{image_metadata.get('height')}
Disease Label: {image_metadata.get('disease_label')}
Clinical Metadata: {image_metadata.get('clinical_attributes', {})}

Annotations:
"""
    
    bboxes = image_metadata.get("bboxes", [])
    if bboxes:
        for idx, box in enumerate(bboxes):
            prompt += f"- Box {idx + 1}: label={box.get('label')}, coordinates={box.get('coords')}, desc={box.get('description')}\n"
    else:
        prompt += "- No bounding box annotations.\n"
        
    masks = image_metadata.get("masks", [])
    if masks:
        for mask in masks:
            prompt += f"- Mask: label={mask.get('label')}, path={mask.get('mask_path')}\n"
            
    if clinical_rules:
        prompt += f"""
### CLINICAL KNOWLEDGE CONSTRAINTS
Disease Description: {clinical_rules.get('description', '')}
Management Guidelines: {clinical_rules.get('guidelines', '')}
Key Features: {', '.join(clinical_rules.get('key_features', []))}
"""
        if "stages" in clinical_rules:
            prompt += "Clinical Staging Criteria:\n"
            for stage, desc in clinical_rules["stages"].items():
                prompt += f"- {stage}: {desc}\n"
                
    prompt += "\nGenerate the multiple-choice question in JSON format following the schema exactly."
    return prompt
