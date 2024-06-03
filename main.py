from utils import bleu_score, rougue_score, bert_score, selfcheck_nli_score
import warnings
warnings.filterwarnings("ignore")




mock_gold =        'A black and white photo of President Obama with a blue tie.'

mock_generated =   'A black and white photo of President Obama with a blue tie.'
mock_generated_2 = 'Black and white photo of President Obama with a blue tie.'

bleu_score_value = bleu_score(mock_generated, mock_gold)
print("bleu_score:",bleu_score_value)


rogue_score_value = rougue_score(mock_generated, mock_gold)
print("rougue_score:",rogue_score_value)

bert_score_value = bert_score(mock_generated, mock_gold)
print("bert_score:",bert_score_value)

selfcheck_nli_score_value = selfcheck_nli_score(mock_generated, mock_gold)
print("selfcheck_nli_score:",selfcheck_nli_score_value)


