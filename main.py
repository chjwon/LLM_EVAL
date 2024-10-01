from utils import bleu_score, rougue_score, bert_score, selfcheck_nli_score, semscore_score, deberta_cos_score
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")


def sentence_bert_score(mock_gen, mock_gold):
    sentences = [mock_gen, mock_gold]
    embeddings = model.encode(sentences)
    similarities = model.similarity(embeddings, embeddings)
    return similarities[0][1]

    



mock_gold = 'Trump is good and Biden is bad.'
mock_generated_1 = 'Biden is good and Trump is bad.' # opposite+order
mock_generated_2 = 'Biden is bad and Trump is good.' # order
mock_generated_3 = 'Trump is bad and Biden is good.' # opposite
mock_generated_4 = 'Trump card is good and Biden is bad.' # typo
mock_generated_5 = 'Trump card is bad and Biden is good.' # typo+opposite
mock_generated_6 = 'Trmup is good and Biden is bad.' # typo
mock_generated_7 = 'Trmup is bad and Biden is good.' # typo


score_board = {
    "dataset":["opposite meaning, reorder","same meaning, reorder","opposite meaning","different word","different word, opposite meaning","typo","typo, oppoiste meaning"],

    "bleu_score":[],
    "rougue_score":[],
    "bert_score":[],
    "sentence_bert_score":[],
    "semscore_score":[],
    "selfcheck_nli_score":[],
    "deberta_cos_sim":[]
}




mock_gen = [mock_generated_1, mock_generated_2, mock_generated_3, mock_generated_4, mock_generated_5, mock_generated_6,mock_generated_7]
for i in range(len(mock_gen)):
    bleu_score_value = bleu_score(mock_gen[i], mock_gold)
    score_board["bleu_score"].append(bleu_score_value)
    
    rogue_score_value = rougue_score(mock_gen[i], mock_gold)
    score_board["rougue_score"].append(rogue_score_value)
    
    bert_score_value = bert_score(mock_gen[i], mock_gold)
    score_board["bert_score"].append(bert_score_value)
    
    sentence_bert_score_value = sentence_bert_score(mock_gen[i], mock_gold)
    score_board["sentence_bert_score"].append(sentence_bert_score_value)
    
    semscore_score_value = semscore_score(mock_gen[i], mock_gold)
    score_board["semscore_score"].append(semscore_score_value)
    
    selfcheck_nli_score_value = selfcheck_nli_score(mock_gen[i], mock_gold)
    score_board["selfcheck_nli_score"].append(selfcheck_nli_score_value)
    
    deberta_score = deberta_cos_score(mock_gen[i],mock_gold)
    score_board["deberta_cos_sim"].append(deberta_score)
    
    
    
df = pd.DataFrame.from_dict(score_board)
print(df)
df.to_csv("./mock_eval_sample.csv")
