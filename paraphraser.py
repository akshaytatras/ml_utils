from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

# to get reproducable paraphrase generations
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)

def paraphrase_sentence(sentence, use_gpu=False, diversity_ranker="euclidean", do_diverse=True,
                        max_return_phrases=5, max_length=32, adequacy_threshold = 0.90, fluency_threshold = 0.90):

    para_phrases = parrot.augment(input_phrase=sentence, 
                                  use_gpu=use_gpu,
                                  diversity_ranker=diversity_ranker,
                                  do_diverse=do_diverse, 
                                  max_return_phrases=max_return_phrases, 
                                  max_length=max_length, 
                                  adequacy_threshold=adequacy_threshold, # if the generated text conveys the same meaning as the original context
                                  fluency_threshold=fluency_threshold) #  if the text is fluent / grammatically correct english
    return para_phrases

if __name__ == '__main__':

    
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
    sentence = 'So let me have the conversation with them tomorrow and.'
    paraphrases = paraphrase_sentence(sentence)
    print('Original Phrase - ')
    print(sentence)
    print('---------------------------------')
    print('Generated Paraphrases - ')
    print(paraphrases)