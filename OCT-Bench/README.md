---
license: apache-2.0
task_categories:
- image-text-to-text
tags:
- multimodal
- ophthalmology
- OCT
- benchmark
- medical
- visual question answering
---

# 🤗 OCT-Bench

[Paper](https://huggingface.co/papers/2607.16609) | [GitHub](https://github.com/baochenfu/OCT-Bench)

![Overview](overview.png)

We introduce OCT-Bench, a comprehensive benchmark for evaluating Multimodal Large Language Models (MLLMs) on optical coherence tomography (OCT) image understanding. OCT-Bench comprises 10,076 expert-verified multiple-choice questions from 4,137 OCT images across seven public datasets and evaluates 3 capability dimensions, 9 capability groups, and 20 fine-grained tasks covering perception, cognition, and clinical reasoning. We benchmark 20 representative MLLMs, including proprietary, open-source, and medical-domain models, providing a comprehensive assessment of OCT understanding capabilities.

For detailed usage and instructions, please refer to the [GitHub page](https://github.com/baochenfu/OCT-Bench).

You can download **OCT-Bench**. The expected directory structure is:

```
OCT-Bench
├── images
│   ├── OCT5K
│   ├── OCTDL
│   └── ...
└── VQA
    ├── T01_VQA.jsonl
    ├── T02_VQA.jsonl
    └── ...
```