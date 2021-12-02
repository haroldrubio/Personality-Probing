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

def batch_sent_score(q_logits: list[torch.Tensor], responses: list[str], logger: logging.Logger, debug: bool = False, softmax: bool = False, var_shift: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = None
    # Compute tokenized batch
    # BUG: Batch size 1
    pbar = tqdm.tqdm(sent_checkpoints)
    for i, checkpoint in enumerate(pbar):
        # Fetch information on current checkpoint logits for the question
        pbar.set_description(f"Now loading {checkpoint}")
        curr_q = torch.squeeze(q_logits[i]) # M
        curr_q_norm = torch.linalg.norm(curr_q)

        # Load and pass through the model
        sent_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        sent_model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
        inputs = sent_tokenizer(responses, padding=True, return_tensors="pt").to(device)

        outputs = sent_model(**inputs)
        logits = outputs.logits # B x M
        # Batch size 1 compensation
        if len(responses) == 1:
            logits = logits.unsqueeze(0)
        logger.info(f"logits: {logits[0]}")
        logger.info(f"logits shape: {logits.shape}")

        # Compute dot product
        if softmax:
            logits = F.softmax(logits, dim=1)
            logger.info(f"softmax logits: {logits[0]}")
            logger.info(f"softmax shape: {logits.shape}")
        logits_norm = torch.squeeze(torch.linalg.norm(logits, dim=1))
        scale_factor = curr_q_norm * logits_norm
        logger.info(f"scale factor shape: {scale_factor.shape}")
        dot_prods = torch.squeeze(torch.mv(logits, curr_q))
        logger.info(f"dot products: {dot_prods}")
        logger.info(f"dot product shape: {dot_prods.shape}")
        if scores is None:
            scores = (dot_prods / scale_factor).unsqueeze(1)
        else:
            scores = torch.cat([scores, (dot_prods / scale_factor).unsqueeze(1)], dim=1)
        logger.info(f"scores first response: {scores[0]}")
        logger.info(f"scores shape: {scores.shape}")
    
    # Shift the mean score by the variance of the distribution
    logger.info(f"means: {torch.mean(scores, dim=1)}")
    logger.info(f"means shape: {torch.mean(scores, dim=1).shape}")
    logger.info(f"stdevs: {torch.std(scores, dim=1)}")
    logger.info(f"stdev shape: {torch.std(scores, dim=1).shape}")
    stdevs = torch.std(scores, dim=1).detach().cpu().numpy()
    scores = torch.mean(scores, dim=1).detach().cpu().numpy()
    final_scores = []
    if not var_shift:
        return scores

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

def check_ngram_overlap(base: str, target: str, n: int = 2):
    """
    Returns true if there is an n-gram overlap (split by space) between the base and target string
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default='gpt2',
        type=str,
        required=True,
        help="Model type selected in the list: gpt2/-medium/-large/-xl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The name of the JSON file outputted"
    )
    parser.add_argument("--num_generation", type=int, default=5)
    parser.add_argument("--start_at", type=int, default=1)
    parser.add_argument("--end_at", type=int, default=120)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to use only 1 question",
    )
    parser.add_argument(
        "--softmax",
        action="store_true",
        help="Uses softmax probability distributions instead of logits",
    )
    args = parser.parse_args()

    set_seed(int(time.time()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.warning(f"device: {device}")

    model_name = args.model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    num_generations = args.num_generation
    start_at = args.start_at
    end_at = args.end_at
    filename = args.output_file

    # Load the filtered questions
    logger.info(f"loading questions")
    with open('./data/questions.json', 'r') as f:
        personality_test = json.load(f)
    questions = personality_test['questions']
    prompt = personality_test['prompt']
    output_dict = {}

    # DEBUG: limit to single item
    if args.debug:
        logger.info(f"running debug mode")
        questions = questions[0:1]
    questions = questions[start_at - 1: end_at]

    for i, question in enumerate(tqdm.tqdm(questions, desc=f"Question Number")):
        # Initialize output structures
        output_dict[i] = {}
        output_dict[i]['question'] = question['text'].strip()
        output_dict[i]['responses'] = []

        logger.info(f"fetching question logits")
        logits = get_question_logits(question['text'].strip())

        input_ids = tokenizer.encode(question['question'], return_tensors='pt').to(device)
        while len(output_dict[i]['responses']) < num_generations:
            # Generate only as much as needed to fill out the list
            logger.info(f"generating responses for question {i + 1} - making {model_name}")
            model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
            sample_outputs = model.generate(
                input_ids,
                do_sample=True, 
                max_length=len(input_ids[0]) + 16, 
                top_k=200, 
                top_p=0.95,
                num_return_sequences=num_generations - len(output_dict[i]['responses'])
            )
            logger.info(f"responses complete - freeing {model_name}")

            # Perform scoring and storing outputs
            # Aggregate responses that pass the ngram overlap
            # TODO: Aggregate valid responses then send in one shot
            logger.info(f"scoring responses for question {i + 1}")
            scoring_batch = []
            for sample_output in tqdm.tqdm(sample_outputs, desc="Response Number"):
                response_dict = {}
                # Get everything after the question
                out_str = tokenizer.decode(sample_output, skip_special_tokens=True)[len(question['question']) - 1:]
                out_str = out_str.split('\n')[0]
                # Continue if it began to generate the prompt
                if check_ngram_overlap(prompt, out_str, n=2):
                    continue
                # And break out if reached limit
                if len( output_dict[i]['responses']) >= num_generations:
                    break

                # If legal, add metadata information to list of responses and add it to the batch
                response_dict['text'] = out_str
                scoring_batch.append(out_str)
                response_dict['facet'] = question['facet']
                response_dict['domain'] = question['domain']
                response_dict['reverse_score'] = question['reverse_score']
                output_dict[i]['responses'].append(response_dict)

            # Compute and set batch scores
            if len(scoring_batch) >= 1:
                batch_scores = batch_sent_score(logits, scoring_batch, logger, debug=args.debug, softmax=args.softmax)
                assert len(scoring_batch) == len(batch_scores), f"{len(scoring_batch)} =/= {len(batch_scores)}"
                for k, score in enumerate(batch_scores):
                    total = len(output_dict[i]['responses'])
                    response = output_dict[i]['responses'][total - len(batch_scores) + k]
                    response['score'] = score

        logger.info(f"dumping to file")
        with open(f"./data/{filename}.json", 'w') as f:
            json.dump(output_dict, f, indent=4)

if __name__ == "__main__":
    main()