# 2024-2025 TCD MSc. Business Analytics Dissertation 

Trinity Business School Corporate Project  

Design and Implementation of a Local Flask Application Using LLaMA for Automated Extraction and Matching of Indian Undergraduate Institutions

A short demonstration video to showcase the key functionalities of the Institution Ranking Checker application: https://drive.google.com/file/d/1RmV_-e_zZY2Qck4l6uvciu-j3QvQhfCu/view

## Abstract
This thesis details the design, development, and evaluation of the Institution Ranking Checker, a privacy-preserving, locally deployed application aimed at supporting the Trinity Business School (TBS) in the efficient screening of undergraduate applicants from Indian institutions. The tool automates the extraction and identification of institutions mentioned in admissions documents and cross-references them against a structured Excel-based institutional ranking sheet to support fair and consistent admissions decision-making.

Built using the Flask web framework and powered by the LLaMA large language model via the Ollama runtime, the system combines AI-driven semantic extraction with RapidFuzz string matching to handle both structured and unstructured documents. Functional testing was conducted using real-world applicant files. The results demonstrate that the LLaMA model outperforms the fuzzy matching method in both accuracy and contextual understanding, particularly when handling low-quality or scanned documents. The entire process runs offline, ensuring full compliance with GDPR and Trinity College’s data governance policies.

By integrating intelligent automation with a user-friendly interface and a human-in-the-loop design, the Institution Ranking Checker offers a scalable and efficient solution to a key administrative bottleneck in the admissions process. It improves processing speed, reduces human error, and maintains full control over applicant data within a trusted local environment.
