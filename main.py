from utils import bleu_score, rougue_score, bert_score, selfcheck_nli_score, semscore_score
import warnings
warnings.filterwarnings("ignore")




mock_gold = 'Trump is good and Biden is bad.'
mock_generated_1 = 'Biden is good and Trump is bad.' # opposite+order
mock_generated_2 = 'Biden is bad and Trump is good.' # order
mock_generated_3 = 'Trump is bad and Biden is good.' # opposite
mock_generated_4 = 'Trump card is good and Biden is bad.' # typo
mock_generated_5 = 'Trump card is bad and Biden is good.' # typo+opposite
mock_generated_6 = 'Trmup is good and Biden is bad.' # typo



bleu_score_value = bleu_score(mock_generated_2, mock_gold)
print("bleu_score:",bleu_score_value)


rogue_score_value = rougue_score(mock_generated_2, mock_gold)
print("rougue_score:",rogue_score_value)

bert_score_value = bert_score(mock_generated_2, mock_gold)
print("bert_score:",bert_score_value)

selfcheck_nli_score_value = selfcheck_nli_score(mock_generated_2, mock_gold)
print("selfcheck_nli_score:",selfcheck_nli_score_value)


semscore_score_value = semscore_score(mock_generated_2, mock_gold)
print("semscore:", semscore_score_value)