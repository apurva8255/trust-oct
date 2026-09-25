::: {align="center"}
# 👁️ TrustOCT

### A Hallucination-Aware Vision-Language Framework for Trustworthy OCT Interpretation

```{=html}
<p>
```
`<strong>`{=html}Multimodal AI • Medical Image Analysis • RAG •
Trustworthy AI • Explainable AI`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p>
```
`<img src="https://img.shields.io/badge/Domain-Medical%20AI-blue" alt="Medical AI"/>`{=html}
`<img src="https://img.shields.io/badge/Imaging-OCT-purple" alt="OCT"/>`{=html}
`<img src="https://img.shields.io/badge/AI-Multimodal%20LLM-orange" alt="Multimodal LLM"/>`{=html}
`<img src="https://img.shields.io/badge/RAG-Evidence%20Retrieval-green" alt="RAG"/>`{=html}
`<img src="https://img.shields.io/badge/SDG-3%20Good%20Health-red" alt="SDG 3"/>`{=html}
`<img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow" alt="Research Prototype"/>`{=html}
```{=html}
</p>
```
```{=html}
<p>
```
`<em>`{=html}Moving from "What did the model predict?" to "Can we verify
why it predicted it?"`</em>`{=html}
```{=html}
</p>
```
:::📌 Overview
TrustOCT is an AI/ML research project focused on improving the
reliability of Multimodal Large Language Models (MLLMs) for
Optical Coherence Tomography (OCT) interpretation.
Rather than accepting only the final disease prediction, TrustOCT aims
to inspect and verify the model's intermediate clinical reasoning.
::: {align="center"}
### Perception → Cognition → Reasoning → Verification → Final Output
:::<table>
<tr>
<td>
<b>{=html}Project Area</b>{=html}
</td>
<td>
AI/ML, Multimodal AI, Medical Image Analysis, RAG, Trustworthy AI
</td>
</tr>
<tr>
<td>
<b>{=html}Primary Goal</b>{=html}
</td>
<td>
Verify image-grounded and clinical claims before accepting an
AI-generated OCT interpretation
</td>
</tr>
<tr>
<td>
<b>{=html}SDG Alignment</b>{=html}
</td>
<td>
UN SDG 3 --- Good Health and Well-being
</td>
</tr>
<tr>
<td>
<b>{=html}Project Type</b>{=html}
</td>
<td>
Academic Research / EDI Prototype
</td>
</tr>
</table>
🚨 Problem Statement
Modern vision-language models can analyze medical images and generate
clinical interpretations. However, a highly confident response is not
necessarily clinically correct.
A model may:
- Produce a confident but incorrect diagnosis.
- Reach the correct final answer using incorrect intermediate
  reasoning.
- Misidentify retinal layers, lesions, or fluid locations.
- Generate explanations that are not adequately supported by the OCT
  image.
- Rely on internal model knowledge without providing verifiable
  clinical evidence.
Traditional disease-classification systems largely follow:
OCT Image → Disease Label
TrustOCT instead investigates:
What did the model see? → Where is the abnormality? → What does it
clinically imply? → Is the conclusion supported by evidence?
🎯 Objectives
The main objectives of TrustOCT are to:
1. Analyze OCT scans using multimodal vision-language models.
2. Structure OCT interpretation into Perception, Cognition, and
   Reasoning stages.
3. Extract individual clinical claims from model-generated
   interpretations.
4. Retrieve relevant clinical knowledge using RAG.
5. Verify whether generated clinical claims are supported, uncertain,
   or contradicted.
6. Detect unsupported or hallucinated clinical statements.
7. Trigger re-evaluation when the initial reasoning is inconsistent
   with available evidence.
8. Evaluate whether verification improves reliability compared with a
   baseline MLLM.
🧠 Cognitive Reasoning Framework
TrustOCT follows a hierarchical OCT interpretation process.
1. Perception --- What is visible?
The system examines low-level visual information such as:
- OCT modality and image characteristics
- Lesion morphology
- Boundaries
- Optical reflectivity
- Quantity and size of abnormalities
- Spatial orientation
2. Cognition --- Where is the abnormality?
The system interprets the observed findings anatomically and
pathologically:
- Retinal region identification
- Retinal layer identification
- Inter-layer relationships
- Lesion classification
- Structural integrity
- Lesion localization
- Disease-pathology association
3. Reasoning --- What does it clinically mean?
The system uses the visual and anatomical findings for higher-level
tasks such as:
- Disease diagnosis
- Disease-stage classification
- Treatment-related reasoning
- Follow-up reasoning
🏗️ Proposed TrustOCT Architecture
                         OCT Image
                             |
                             v
                  Vision-Language Model
                             |
                             v
                  Initial Interpretation
                             |
                             v
                    Claim Extraction
                             |
              +--------------+--------------+
              |                             |
              v                             v
       Image/Anatomical              Clinical Knowledge
          Evidence                    Retrieval (RAG)
              |                             |
              +--------------+--------------+
                             |
                             v
                     Claim Verification
                             |
               +-------------+-------------+
               |             |             |
               v             v             v
           Supported      Uncertain     Contradicted
               |             |             |
               +-------------+-------------+
                             |
                             v
                    Re-reason if needed
                             |
                             v
                 Verified / Abstained Output
🔍 Claim-Level Verification
A central idea of TrustOCT is that the entire generated response should
not be accepted or rejected as a single block.
For example, suppose an MLLM generates:
"The OCT indicates diabetic macular edema with intraretinal fluid."

TrustOCT can decompose this into claims:
1. Fluid is present.
2. The fluid is intraretinal.
3. The observed findings are consistent with diabetic macular edema.
Each claim can then be checked independently.
Image Evidence
Used to answer:
Is the claimed feature actually visible in this OCT scan?
Possible sources include:
- OCT image annotations
- Bounding boxes
- Segmentation masks
- Disease labels
- Anatomical metadata
Clinical Evidence
Used to answer:
Does established clinical knowledge support the relationship claimed
by the model?
The project knowledge base is intended to use trusted ophthalmology
material such as:
- American Academy of Ophthalmology (AAO) Preferred Practice Pattern
  guidelines
- Consensus ophthalmology literature
- Established ophthalmology references such as Ryan's Retina
📚 Retrieval-Augmented Generation (RAG)
Clinical documents can be converted into a searchable knowledge base:
Clinical Guidelines / Literature
              |
              v
         Text Extraction
              |
              v
           Chunking
              |
              v
          Embeddings
              |
              v
      FAISS / ChromaDB
              |
              v
       Evidence Retrieval
              |
              v
      Clinical Verification
RAG is used for clinical knowledge verification. It does not by
itself prove that a visual feature exists in a particular OCT image;
image-grounded verification must be handled separately.
🩺 Clinical Reasoning / Med-CoT
Rather than forcing the MLLM to directly output a disease or answer,
structured clinical reasoning can guide it through:
Visual Perception
       ↓
Anatomical Localization
       ↓
Clinical Correlation
       ↓
Final Conclusion
TrustOCT extends this idea by asking an additional question:
Is the generated reasoning actually supported?
Structured reasoning generates an explanation; TrustOCT's verification
layer is intended to evaluate that explanation.
🔬 Relationship with OCT-Bench
OCT-Bench and TrustOCT serve different purposes.
  OCT-Bench                          TrustOCT
  Evaluates MLLMs                    Builds a reliability/verification
                                     framework
  Measures model performance         Checks generated clinical claims
  Identifies failure points          Attempts to detect unsupported
                                     reasoning
  Perception → Cognition → Reasoning Perception → Cognition → Reasoning
                                     → Verification
  Benchmark-focused                  Trustworthiness-focused
TrustOCT is inspired by limitations exposed through hierarchical OCT
evaluation; it should not be described as a modification of OCT-Bench
unless the benchmark itself is directly extended.
📊 Evaluation Strategy
The project can compare:
Baseline
OCT Image → MLLM → Final Answer
TrustOCT
OCT Image
   ↓
MLLM
   ↓
Structured Reasoning
   ↓
Claim Extraction
   ↓
Evidence Retrieval + Image Verification
   ↓
Claim Verification
   ↓
Re-reason / Abstain when necessary
   ↓
Final Output
Potential evaluation measures include:
- Final-answer accuracy
- Unsupported-claim rate
- Hallucination rate
- Evidence consistency
- Performance across Perception, Cognition, and Reasoning
- Confidence/calibration behavior
- Abstention behavior
- Additional inference latency
The exact hallucination and evidence-consistency metrics should be
finalized only after defining a validated ground-truth protocol.
👁️ Target OCT Conditions
Depending on the final dataset, the system may evaluate retinal
conditions such as:
- Age-Related Macular Degeneration (AMD)
- Diabetic Macular Edema (DME)
- Central Serous Chorioretinopathy (CSC)
- Retinal Vein Occlusion (RVO)
- Macular Hole (MH)
Only conditions represented and validated in the final experimental
dataset should be reported as supported by the implemented system.
🛠️ Proposed Technology Stack
  Component            Technology
  Programming          Python
  Image Processing     OpenCV, Pillow
  ML / Deep Learning   PyTorch, Hugging Face
  Multimodal AI        Vision-Language Models
  RAG                  FAISS / ChromaDB
  Data Validation      Pydantic
  Interface            Streamlit
  Data Handling        Pandas / NumPy
  Evaluation           Scikit-learn / custom metrics
The exact MLLM provider(s) should be documented once the experimental
setup is finalized.
👥 Team Module Distribution
Module 1 --- Dataset & Computer Vision
- OCT dataset preparation
- Data normalization
- Image preprocessing
- Bounding boxes and segmentation masks
- Retinal-layer and lesion visualization
- Image-grounded evidence preparation
Module 2 --- Clinical Knowledge & RAG
- Clinical knowledge-base preparation
- Guideline processing
- Document chunking and embeddings
- Vector database
- Evidence retrieval
- VQA/question generation where required
Module 3 --- MLLM & Clinical Reasoning
- Vision-language model integration
- OCT image + question inference
- Structured clinical reasoning
- Perception/Cognition/Reasoning analysis
- Baseline model experiments
- Model-response parsing
Module 4 --- Trustworthiness & Evaluation
- Claim extraction
- Claim-level verification
- Contradiction detection
- Hallucination analysis
- Evaluation metrics
- Dashboard and system integration
💡 Expected Contribution
The intended contribution of TrustOCT is not simply to combine several
AI models.
The project investigates whether claim-level verification of an OCT
reasoning chain can make multimodal AI outputs more reliable.
The central research question is:
Can evidence-aware verification identify and reduce unsupported
clinical reasoning in multimodal OCT interpretation?

⚠️ Limitations
- RAG quality depends on the quality and coverage of the indexed
  clinical sources.
- Clinical evidence cannot independently verify whether an abnormality
  is actually visible in an OCT image.
- Model-generated confidence is not equivalent to calibrated clinical
  confidence.
- Expert/clinician validation may be required for strong medical
  claims.
- Results from a limited OCT dataset should not be generalized to
  clinical deployment.
- TrustOCT is a research and evaluation project, not a replacement for
  an ophthalmologist.
🚀 Future Scope
- Adaptive verification based on uncertainty
- Multi-agent evidence verification
- Visual grounding and saliency analysis
- 3D volumetric OCT analysis
- Risk-weighted clinical evaluation
- Improved uncertainty calibration and abstention
- Extension to other medical imaging modalities
🌍 SDG Alignment
SDG 3 --- Good Health and Well-being
TrustOCT aligns with SDG 3 through research into safer and more reliable
AI-assisted medical image interpretation.
⚕️ Disclaimer
TrustOCT is intended for academic research and evaluation. It is not a
certified medical device and should not be used for independent clinical
diagnosis, treatment selection, or patient management.
