import argparse
import logging

# Imports
import torch, json, tqdm, sys, random, time
import torch.nn.functional as F
import numpy as np
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
sent_checkpoints = ['distilbert-base-uncased-finetuned-sst-2-english',
                    'bhadresh-savani/distilbert-base-uncased-emotion',
                    'textattack/albert-base-v2-yelp-polarity',
                    'textattack/albert-base-v2-imdb']
# Possible other checkpoints:
# Amazon Reviews: fabriceyhc/bert-base-uncased-amazon_polarity
# Yelp: textattack/albert-base-v2-yelp-polarity
# IMDb: textattack/albert-base-v2-imdb
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def logit_to_single_score(logits: torch.Tensor):
    # Label 0 is negative, label 1 is positive
    logits = logits[0]
    logits = F.softmax(logits, dim=0)
    logits = logits.cpu().detach().numpy()
    neg_score = -1 * float(logits[0])
    pos_score = float(logits[1])
    return pos_score + neg_score

def get_question_logits(question: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = []
    for checkpoint in sent_checkpoints:
        sent_model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
        sent_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        inputs = sent_tokenizer(question, return_tensors="pt").to(device)
        outputs = sent_model(**inputs)
        logits.append(outputs.logits)
    return logits

def compute_cosine(tensor_1: torch.Tensor, tensor_2: torch.Tensor, softmax: bool = False):
    # Compute the cosine between 2 tensors by computing their magnitudes and dot product
    if len(tensor_1.shape) == 2:
        tensor_1 = torch.squeeze(tensor_1)
    if len(tensor_2.shape) == 2:
        tensor_2 = torch.squeeze(tensor_2)
    if softmax:
        tensor_1 = F.softmax(tensor_1, dim=0)
        tensor_2 = F.softmax(tensor_2, dim=0)
    dot_prod = torch.dot(tensor_1, tensor_2)
    mag_1 = torch.linalg.norm(tensor_1)
    mag_2 = torch.linalg.norm(tensor_2)
    cosine = (dot_prod / (mag_1 * mag_2)).detach().cpu().numpy()
    return cosine

def get_sent_score(q_logits: list[torch.Tensor], phrase: str, debug: bool = False, softmax: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = []
    # Compute and compare logits for each model
    for i, checkpoint in enumerate(sent_checkpoints):
        question_logits = q_logits[i]
        sent_model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
        sent_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        inputs = sent_tokenizer(phrase, return_tensors="pt").to(device)
        outputs = sent_model(**inputs)
        scores.append(compute_cosine(outputs.logits, question_logits, softmax))
    
    if debug:
        print(scores)

    # Shift the mean score by the variance of the distribution
    score = np.mean(scores)
    if not softmax:
        if score > 0:
            score = max(0, score - np.std(scores)**2)
        else:
            score = min(0, score + np.std(scores)**2)
    else:
        if score > 0.5:
            score = max(0.5, score - np.std(scores)**2)
        else:
            score = min(0.5, score + np.std(scores)**2)

    return score

def batch_sent_score(q_logits: list[torch.Tensor], responses: list[str], logger: logging.Logger, debug: bool = False, softmax: bool = False, var_shift: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = None
    # Compute tokenized batch
    # BUG: Batch size 1
    pbar = tqdm.tqdm(sent_checkpoints)
    for i, checkpoint in enumerate(pbar):
        # Fetch information on current checkpoint logits for the question
        pbar.set_description(f"Now loading {checkpoint}")
        curr_q = torch.squeeze(q_logits[i]) # M
        curr_q = F.softmax(curr_q, dim=0)
        curr_q_norm = torch.linalg.norm(curr_q)

        # Load and pass through the model
        sent_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        sent_model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
        inputs = sent_tokenizer(responses, padding=True, return_tensors="pt").to(device)

        outputs = sent_model(**inputs)
        logits = outputs.logits # B x M
        #logger.info(f"logits: {logits[0]}")
        #logger.info(f"logits shape: {logits.shape}")

        # Compute dot product
        if softmax:
            logits = F.softmax(logits, dim=1)
            #logger.info(f"softmax logits: {logits[0]}")
            #logger.info(f"softmax shape: {logits.shape}")
        logits_norm = torch.squeeze(torch.linalg.norm(logits, dim=1))
        scale_factor = curr_q_norm * logits_norm
        #logger.info(f"scale factor shape: {scale_factor.shape}")
        dot_prods = torch.squeeze(torch.mv(logits, curr_q))
        # Batch size 1 compensation
        if len(responses) == 1:
            dot_prods = torch.tensor([dot_prods], dtype=logits.dtype, device=device)
        #logger.info(f"dot products: {dot_prods}")
        #logger.info(f"dot product shape: {dot_prods.shape}")
        if scores is None:
            scores = (dot_prods / scale_factor).unsqueeze(1)
        else:
            scores = torch.cat([scores, (dot_prods / scale_factor).unsqueeze(1)], dim=1)
        #logger.info(f"scores first response: {scores[0]}")
        #logger.info(f"scores shape: {scores.shape}")
    
    # Shift the mean score by the variance of the distribution
    #logger.info(f"means: {torch.mean(scores, dim=1)}")
    #logger.info(f"means shape: {torch.mean(scores, dim=1).shape}")
    #logger.info(f"stdevs: {torch.std(scores, dim=1)}")
    #logger.info(f"stdev shape: {torch.std(scores, dim=1).shape}")
    stdevs = torch.std(scores, dim=1).detach().cpu().numpy()
    scores = torch.mean(scores, dim=1).detach().cpu().numpy()
    final_scores = []

    for avg, std in zip(scores, stdevs):
        if var_shift:
            if not softmax:
                if avg > 0:
                    avg = max(0, avg - std**2)
                else:
                    avg = min(0, avg + std**2)
            else:
                if avg > 0.5:
                    avg = max(0.5, avg - std**2)
                else:
                    avg = min(0.5, avg + std**2)
        final_scores.append(float(avg))

    return final_scores

def check_ngram_overlap(base: str, target: list[int], n: int = 2):
    """
    Returns true if there is an n-gram overlap (split by space) between the base and target string
    Args:
        target: Already tokenized target
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    base = tokenizer.encode(base.lower())
    target = tokenizer.encode(target.lower())

    for i in range(len(base) - n):
        gram = base[i: i + n]
        for j in range(len(target) - n):
            target_gram = target[j: j + n]
            overlap = True
            for k in range(len(gram)):
                if gram[k] != target_gram[k]:
                    overlap = False
            
            if overlap:
                return True
    return False

def main():
    pass

if __name__ == "__main__":
    main()