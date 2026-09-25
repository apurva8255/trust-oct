import json
from typing import Dict, Any, Optional

CLINICAL_KNOWLEDGE = {
    "NORMAL": {
        "description": "Healthy retina structure with no pathology.",
        "guidelines": "No treatment required. Normal visual acuity and layer integrity.",
        "key_features": ["Intact retinal layers", "No fluid accumulation", "No RPE elevation"]
    },
    "AMD": {
        "description": "Age-Related Macular Degeneration (AMD). Specifically, Wet (Neovascular) AMD is characterized by choroidal neovascularization leading to fluid leakage, hemorrhage, and scarring.",
        "guidelines": "First-line therapy consists of intravitreal anti-VEGF injections (e.g., Aflibercept, Ranibizumab). Rapid intervention is necessary to prevent irreversible central vision loss.",
        "key_features": ["Pigment Epithelial Detachment (PED)", "Subretinal Hyperreflective Material (SHRM)", "Subretinal Fluid (SRF)"],
        "stages": {
            "Early": "Small drusen, no pigmentary abnormalities.",
            "Intermediate": "Medium/large drusen, pigmentary changes.",
            "Late": "Geographic atrophy or neovascularization (wet AMD)."
        }
    },
    "CSC": {
        "description": "Central Serous Chorioretinopathy (CSC) is characterized by localized serous detachment of the neurosensory retina due to choroidal hyperpermeability.",
        "guidelines": "Usually managed conservatively with observation for 3 months, as many cases resolve spontaneously. If fluid persists and progressive visual acuity decline occurs, consider Photodynamic Therapy (PDT) with verteporfin or focal laser photocoagulation.",
        "key_features": ["Subretinal Fluid (SRF) accumulation", "Intact inner retinal structure", "Thickened choroid"],
        "stages": {
            "Acute": "Focal fluid accumulation, duration <6 months, high chance of spontaneous resolution.",
            "Chronic": "Persistent fluid >6 months, diffuse RPE alterations, risk of permanent vision loss."
        }
    },
    "DME": {
        "description": "Diabetic Macular Edema (DME) is a complication of diabetic retinopathy characterized by retinal vascular leakage leading to fluid accumulation within the macula.",
        "guidelines": "Standard of care is intravitreal anti-VEGF injections. For refractory cases, intravitreal corticosteroid implants or focal laser photocoagulation may be considered.",
        "key_features": ["Intraretinal Fluid (IRF) cystoid spaces", "Subretinal Fluid (SRF) in severe cases", "Retinal thickening"],
        "stages": {
            "Mild": "Retinal thickening far from macula center.",
            "Moderate": "Retinal thickening close to macula center but not involving center.",
            "Severe": "Retinal thickening involving the center of the macula."
        }
    },
    "MH": {
        "description": "Macular Hole (MH) is a full-thickness defect in the neurosensory retina at the fovea.",
        "guidelines": "Surgical repair via Pars Plana Vitrectomy (PPV) combined with Internal Limiting Membrane (ILM) peeling and gas tamponade (e.g., SF6 or C3F8) is the standard treatment.",
        "key_features": ["Full-thickness tissue gap", "Cystic changes in hole edges", "Vitreomacular traction"],
        "stages": {
            "Stage 1": "Impending macular hole (foveolar detachment or loss of foveal depression).",
            "Stage 2": "Small full-thickness foveal defect (<400 micrometers in diameter).",
            "Stage 3": "Large full-thickness foveal defect (>=400 micrometers) with partial vitreous separation (posterior hyaloid remains attached to optic nerve).",
            "Stage 4": "Full-thickness macular hole with complete posterior vitreous detachment (indicated by Weiss ring)."
        }
    }
}

class KnowledgeDatabase:
    """
    Local clinical guidelines and disease metadata repository.
    Used to supply context for medical cognition and clinical reasoning generation.
    """
    def __init__(self):
        self.db = CLINICAL_KNOWLEDGE

    def get_knowledge_for_disease(self, disease_label: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves clinical guidelines and description for a disease label.
        """
        return self.db.get(disease_label.upper())

    def get_stage_description(self, disease_label: str, stage_name: str) -> Optional[str]:
        """
        Retrieves detailed clinical definition of a specific disease stage.
        """
        disease_info = self.get_knowledge_for_disease(disease_label)
        if disease_info and "stages" in disease_info:
            # Try exact match or fuzzy match
            stages = disease_info["stages"]
            for s_key, s_val in stages.items():
                if stage_name.lower() in s_key.lower() or s_key.lower() in stage_name.lower():
                    return s_val
        return None

    def list_diseases(self) -> list:
        return list(self.db.keys())
