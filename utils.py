"""
All function has to be
input1 : sentence | type : str | sentence_generated
input2 : sentence | type : str | sentence_gold

output : score | type float

"""
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
import torch, gc
import torch.nn as nn

class EmbeddingModelWrapper():
    DEFAULT_MODEL="sentence-transformers/all-mpnet-base-v2"

    def __init__(self, model_path=None, bs=8):
        if model_path is None: model_path = self.DEFAULT_MODEL
        self.model, self.tokenizer = self.load_model(model_path)
        self.bs = bs
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def load_model(self, model_path):
        model = AutoModel.from_pretrained(
            model_path,
        ).cuda()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
             model_path,
        )
        return model, tokenizer

    def emb_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences):
        embeddings=torch.tensor([],device="cuda")
        
        if self.bs is None:
            batches=[sentences]
        else:
            batches = [sentences[i:i + self.bs] for i in range(0, len(sentences), self.bs)]  
            
        for sentences in batches:
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to("cuda")
            with torch.no_grad():
                model_output = self.model(**encoded_input)        
            batch_embeddings=self.emb_mean_pooling(model_output, encoded_input['attention_mask'])
            
            embeddings=torch.cat( (embeddings, batch_embeddings), dim=0 )

        return embeddings

    def get_similarities(self, x, y=None):
        if y is None:
            num_samples=x.shape[0]
            similarities = [[0 for i in range(num_samples)] for f in range(num_samples)]
            for row in tqdm(range(num_samples)):
                # similarities[row][0:row+1]=em.cos(x[row].repeat(row+1,1), x[0:row+1]).tolist()
                similarities[row][0:row+1] = self.cos(x[row].repeat(row+1, 1), x[0:row+1]).tolist()

            return similarities
        else:            
            return self.cos(x,y).tolist()

class ModelPredictionGenerator:
    def __init__(self, model, tokenizer, eval_dataset, use_accelerate=False, bs=8, generation_config=None):
        self.model=model
        self.tokenizer=tokenizer
        self.bs=bs
        self.eval_prompts=self.messages_to_prompts( eval_dataset )
        self.use_accelerate=use_accelerate
        self.accelerator = Accelerator()

        assert tokenizer.eos_token_id is not None
        assert tokenizer.chat_template is not None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # llama-precise
        if generation_config is None:            
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.1,
                "repetition_penalty": 1.18,
                "top_k": 40,
                "do_sample": True,
                "max_new_tokens": 100,
                "pad_token_id": tokenizer.pad_token_id
            }
        else:
            self.generation_config = generation_config

    def clear_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def messages_to_prompts(self, ds):
        prompts=[]
        for conversation in ds["messages"]:
            for i,msg in enumerate(conversation):
                if msg["role"]=="user":
                    prompts.append(
                        dict (
                            # prompt: format current messages up to the current user message and add a generation prompt
                            prompt=self.tokenizer.apply_chat_template(conversation[:i+1], add_generation_prompt=True, tokenize=False),
                            answer_ref=conversation[i+1]["content"]
                        )
                    )
        return prompts

    def get_batches(self, dataset, batch_size):
        return [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]  

    def tokenize_batch(self, batch):
        pad_side=self.tokenizer.padding_side
        self.tokenizer.padding_side="left"     # left pad for inference
        
        prompts=[ item["prompt"] for item in batch ]   
        prompts_tok=self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding='longest', 
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_length=True,
            pad_to_multiple_of=8,
            add_special_tokens=False
        ).to(self.model.device)
        self.tokenizer.padding_side=pad_side   # restore orig. padding side
        
        return prompts_tok

    def generate_batch(self, batch_tok):
        with torch.no_grad():
            outputs_tok=self.model.generate(
                input_ids=batch_tok["input_ids"],
                attention_mask=batch_tok["attention_mask"],
                **self.generation_config
            ).to("cpu")
        outputs=[
            # cut prompt from output
            self.tokenizer.decode(
                outputs_tok[i][outputs_tok[i] != self.tokenizer.pad_token_id][batch_tok["length"][i]:], 
                spaces_between_special_tokens=False,
                skip_special_tokens=True
                ).strip()
            for i,t in enumerate(outputs_tok) ]

        return outputs

    def run(self):
        self.model.eval()
        self.clear_cache()
    
        if self.use_accelerate:
            with self.accelerator.split_between_processes(list(range(len(self.eval_prompts)))) as eval_prompts_local_idcs:
                eval_prompts_local = [self.eval_prompts[i] for i in eval_prompts_local_idcs]
        else:
            eval_prompts_local = self.eval_prompts

        for batch in tqdm( self.get_batches(eval_prompts_local, self.bs) ):
            batch_tok = self.tokenize_batch( batch )
            answers = self.generate_batch( batch_tok )   
    
            for i in range(len(batch)):
                batch[i]["answer_pred"]=answers[i]
                batch[i]["GPU"]=self.accelerator.process_index
            
        if self.use_accelerate:
            return gather_object(eval_prompts_local)
        else:
            return eval_prompts_local


import torch
from bert_score import score
import evaluate
from evaluate import load
import time
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
from nltk.translate.bleu_score import sentence_bleu
from openai import OpenAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
openaiKey = open("openaiKey.txt",'r').readline()
# print(openaiKey)

def send_gpt_geval(cur_prompt):
    client = OpenAI(api_key=openaiKey)
    score = 0
    try:
        _response = client.chat.completions.create(model="gpt-4-0613",
        messages=[{"role": "system", "content": cur_prompt}],
        temperature=2,
        max_tokens=5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=20)
        time.sleep(0.5)

        all_responses = [_response.choices[i].message.content for i in
                            range(len(_response.choices))]
        scores = [float(response) for response in all_responses if response.replace('.', '', 1).isdigit()]
        if scores:
            score = sum(scores) / len(scores)
        else:
            print(f"No valid scores returned")
    except Exception as e:
        print(e)
        if ("limit" in str(e)):
            time.sleep(2)
        else:
            print('ignored')
    return score

def geval_score(sentence_generated, sentence_gold):
    coh_prompt = open("./prompt/coh_detailed.txt").read() # /5
    con_prompt = open("./prompt/con_detailed.txt").read() # /5
    flu_prompt = open("./prompt/flu_detailed.txt").read() # /3
    rel_prompt = open("./prompt/rel_detailed.txt").read() # /5
    
    coh_prompt = coh_prompt.replace('{{Document}}', sentence_generated).replace('{{Summary}}', sentence_gold)
    con_prompt = con_prompt.replace('{{Document}}', sentence_generated).replace('{{Summary}}', sentence_gold)
    flu_prompt = flu_prompt.replace('{{Document}}', sentence_generated).replace('{{Summary}}', sentence_gold)
    rel_prompt = rel_prompt.replace('{{Document}}', sentence_generated).replace('{{Summary}}', sentence_gold)
    
    coh_score = send_gpt_geval(cur_prompt=coh_prompt)
    con_score = send_gpt_geval(cur_prompt=con_prompt)
    flu_score = send_gpt_geval(cur_prompt=flu_prompt)
    rel_score = send_gpt_geval(cur_prompt=rel_prompt)
    print("coh_score, con_score, flu_score, rel_score:", coh_score, con_score, flu_score, rel_score)
    return coh_score, con_score, flu_score, rel_score


def bert_score(sentence_generated,sentence_gold):
    cands = [sentence_generated]
    refs = [sentence_gold]
    (P, R, F), hashname = score(cands, refs, lang="en", return_hash=True)
    return P.mean().item()


def selfcheck_nli_score(sentence_generated,sentence_gold):
    selfcheck_nli = SelfCheckNLI(device=device)

    sent_scores_nli = selfcheck_nli.predict(
    sentences = [sentence_gold],                          
    sampled_passages = [sentence_generated],
    )
    return normalize_selfcheck_score(sent_scores_nli[0])

def normalize_selfcheck_score(score):
    return 1 - score

def bleu_score(sentence_generated,sentence_gold):
    references = [sentence_gold.split()]
    candidate = sentence_generated.split()
    score = sentence_bleu(references, candidate)
    return score

def rougue_score(sentence_generated,sentence_gold):
    references = [[sentence_gold]]
    candidate = [sentence_generated]
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=candidate, references=references)
    
    score = results['rougeL']
    return score
    

def semscore_score(sentence_generated, sentence_gold):
    sentences = [sentence_gold, sentence_generated]
    em = EmbeddingModelWrapper()
    sentence_embeddings = em.get_embeddings(sentences)
    similarities = em.get_similarities(sentence_embeddings.cuda())
    return similarities[1][0]