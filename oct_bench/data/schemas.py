from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AnnotationBox(BaseModel):
    coords: List[int] = Field(..., description="[x1, y1, x2, y2] coordinates")
    label: str = Field(..., description="Class label, e.g. 'PED', 'SRF', 'IRF'")
    color: Optional[str] = Field("yellow", description="Render color name or hex code")
    description: Optional[str] = Field(None, description="Optional textual detail")

class AnnotationMask(BaseModel):
    mask_path: str = Field(..., description="Relative or absolute path to segmentation mask image")
    label: str = Field(..., description="Retinal layer/region label, e.g. 'ILM', 'RPE'")
    color: Optional[str] = Field(None, description="Display color code")

class ImageSample(BaseModel):
    image_id: str = Field(..., description="Unique image identifier")
    image_path: str = Field(..., description="Path to the original OCT scan image")
    dataset_source: str = Field(..., description="Source dataset, e.g. 'OCT5k', 'AMD-SD'")
    width: int
    height: int
    disease_label: Optional[str] = Field(None, description="Overall disease classification")
    bboxes: List[AnnotationBox] = Field(default_factory=list)
    masks: List[AnnotationMask] = Field(default_factory=list)
    clinical_attributes: Dict[str, Any] = Field(default_factory=dict, description="Metadata like age, eye side, severity")

class MCQ(BaseModel):
    question_id: str = Field(..., description="Unique question identifier")
    task_id: str = Field(..., description="Fine-grained task ID (T01-T20)")
    image_id: str = Field(..., description="Reference image ID")
    question: str = Field(..., description="Multiple choice question text")
    options: Dict[str, str] = Field(..., description="A, B, C, D option mapping")
    correct_answer: str = Field(..., description="Correct option character (A, B, C, or D)")
    explanation: Optional[str] = Field(None, description="Clinical/Visual explanation of correct option")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Task context, metadata, and difficulty info")
