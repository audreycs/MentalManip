# MentalManip Dataset

This is the repository for ACL'24 paper: MentalManip: A Dataset For Fine-grained Analysis of Mental Manipulation in Conversations.

### 1. Environment Requirement
We recommend installing the following packages and versions before running the code:
- Pytorch: 2.1.2
- Transformers: 4.36.2
- Tokenizers: 0.15.0
- Openai: 1.6.1
- Scipy: 1.11.4
- Seaborn: 0.12.2
- Sentence-transformers: 2.3.0
- tqdm: 4.65.0
- Pandas: 2.1.4
- scikit-learn: 1.2.2
- peft: 0.7.1
- trl: 0.7.7
  

### 2. File Structure of This Repository
```
MentalManip/
├── README.md
├── dataset/  # contains the final MentalManip dataset
├── experiments/  # Code for all the experiments
│   ├── datasets/  # Datasets for the experiments
│   ├── manipulation_detection/  # Code for the manipulation detection task
│   ├── technique_vulnerability/  # Code for the technique and vulnerability classification task
├── statistic_analysis/  # Code for generating statistical figures in the paper
```
