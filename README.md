# Comparing-Text-Similarity-Between-PDFs
This project proposes a novel methodology for measuring semantic similarity between unstructured PDF documents by combining LDA for topic modeling and BERT for contextual embeddings. It significantly outperforms traditional similarity measures such as TF-IDF and cosine similarity—especially in tasks like plagiarism detection and legal document analysis.

📌 Key Features
Preprocessing of unstructured documents

Topic extraction using LDA

Contextual embedding using BERT

Cosine similarity computation

Heatmap visualization of similarity scores

Let me know if you'd like help customizing it further with badges, installation steps, or a live demo link.








You said:
add all feature in detail 


ChatGPT said:
Absolutely! Here's a complete and well-structured README.md section that includes all project features in detail, tailored to your IEEE-published research:

📄 IEEE Research Publication
This project has been published in IEEE Xplore as part of the
2024 5th International Conference on Intelligent Computing and Control Systems (ICICCS).

Paper Title: An Approach for Measuring Unstructured Text Document Similarity Using LDA-BERT Embedding Model
📚 Published by: IEEE
👨‍💻 Authors: Ishan Ramkrishna Chinchghare, Akhilesh Dhoot, Pranay Rahangdale, Prof. Ashwini V. Zadgaonkar, Gaurav Iyer, Prateek B
🔗 Read the IEEE Paper

🧠 Project Overview
This project introduces a hybrid approach to measuring semantic similarity between unstructured PDF documents by combining the strengths of Latent Dirichlet Allocation (LDA) and BERT (Bidirectional Encoder Representations from Transformers). It improves upon traditional text similarity methods by capturing both topical and contextual information.

🚀 Features
🔍 1. Text Preprocessing
Tokenization and lowercasing of text

Removal of stopwords

Filtering out non-alphabetic characters

Lemmatization using NLTK

Outputs clean, uniform text ready for topic modeling and embedding

📚 2. Topic Modeling with LDA
Uses the Gensim library to build an LDA model from preprocessed text

Extracts dominant topics and distributions per document

Represents documents as topic vectors

Enables coarse-level similarity understanding

🧠 3. BERT Embedding Generation
Uses Hugging Face Transformers to extract BERT embeddings

Focuses on [CLS] token representation for capturing the entire document meaning

Converts text into high-dimensional vectors with deep semantic understanding

📈 4. Similarity Score Calculation
Calculates cosine similarity between document pairs using:

LDA topic distributions

BERT embeddings

Combines both similarity scores into a weighted average:

0.2 for abstract

0.3 for introduction

0.2 for keywords

Customizable based on document sections

🔥 5. Visualization Tools
Generates a heatmap (square matrix) using Seaborn to show document-to-document similarity scores

Provides a circle graph for visual comparison and clustering insights

Color-coded similarity representation (e.g., 80% = high similarity, 30% = low similarity)

🧪 6. Evaluation and Testing
Tested on PDF datasets from multiple domains:

Research papers

Legal documents

News articles

Demonstrated robust performance on paraphrased or semantically reworded documents

Outperforms traditional methods (TF-IDF, Jaccard, plain Cosine Similarity)

🛠 Tech Stack
Python 🐍

NLTK

Gensim

Hugging Face Transformers

NumPy / SciPy

Matplotlib & Seaborn

PyMuPDF or PDFMiner (for PDF text extraction)

💡 Potential Applications
Plagiarism Detection

Legal Document Comparison

Research Literature Review Tools

News Article Deduplication

Semantic Document Clustering

Knowledge Graph Building
