# utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F    
import math


class Score:
    def __init__(self, bert_model=None, bert_tokenizer=None, elmo_model=None, elmo_tokenizer=None, gpt2_model=None, gpt2_tokenizer=None):
        """
        Initialize the Score class with BERT 、ELMo 、 gpt-2 models and tokenizers.
        """
        self.bert_model = bert_model
        self.elmo_model = elmo_model
        self.bert_tokenizer = bert_tokenizer
        self.elmo_tokenizer = elmo_tokenizer
        self.gpt2_model = gpt2_model
        self.gpt2_tokenizer = gpt2_tokenizer

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def bert_score(self, sentence, bert_model=None, bert_tokenizer=None):
        """
        Calculate the score of a sentence based on logits(of vocab_size) output.
        
        :param sentence: str, input sentence
        
        :return: score of the sentence
        """
        if bert_model is None:
            bert_model = self.bert_model
        if bert_tokenizer is None:
            bert_tokenizer = self.bert_tokenizer
        bert_model.to(self.device)
        bert_model.eval()
        
        # map to ids
        tokens = bert_tokenizer.tokenize(sentence)
        token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [bert_tokenizer.cls_token_id] + token_ids + [bert_tokenizer.sep_token_id]
        seq_len = len(input_ids)
        # input_ids_tensor = torch.tensor([input_ids]).to(self.device)

        # log score = -1/n * sum(log p_wi)
        log_probs = []
        mask_token_id = bert_tokenizer.mask_token_id

        for i in range(1, seq_len-1):
            # mask the i-th token(id w_i)
            masked_input_ids = input_ids.copy()
            masked_input_ids[i] = mask_token_id
            masked_input_ids_tensor = torch.tensor([masked_input_ids]).to(self.device)

            with torch.no_grad():
                outputs = bert_model(masked_input_ids_tensor)
                logits = outputs.logits # (batch_size, seq_len, vocab_size)

            # (vocab_size,)
            mask_logits = logits[0, i] 
            probs = F.softmax(mask_logits, dim=-1)

            w_i = input_ids[i]
            p_wi = probs[w_i]
            log_probs.append(torch.log(p_wi + 1e-10))  # avoid log(0)

        log_score = -torch.tensor(log_probs).mean()
        score = torch.exp(log_score)
        
        return score.item()

    def elmo_score(self, sentence, elmo_model=None, elmo_tokenizer=None, elmo_hidden_dim=1024, elmo_vocab_size=20000):
        """
        Calculate the ELMo score of a sentence based on bidirectional info(forward_hidden, backward_hidden).
        Add a linear layer to map hidden states to vocab size.
        
        :param sentence: str, input sentence
        
        :return: score of the sentence
        """
        if elmo_model is None:
            elmo_model = self.elmo_model
        if elmo_tokenizer is None:
            elmo_tokenizer = self.elmo_tokenizer
        elmo_model.to(self.device)
        elmo_model.eval()

        # map to ids
        tokens = elmo_tokenizer.tokenize(sentence)
        token_ids = elmo_tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [elmo_tokenizer.cls_token_id] + token_ids + [elmo_tokenizer.sep_token_id]
        seq_len = len(input_ids)
        input_ids_tensor = torch.tensor([input_ids]).to(self.device)

        # log score = -1/n * sum(log p_wi)
        log_probs = []

        # map hidden states to vocab size (for both forward and backward)
        linear_layer = nn.Linear(elmo_hidden_dim, elmo_vocab_size).to(self.device)

        with torch.no_grad():
            # assuming elmo_model returns a tuple (forward_hidden, backward_hidden)
            # (batch_size, seq_len, hidden_dim)
            forward_hidden, backward_hidden = elmo_model(input_ids_tensor)

            forward_logits = linear_layer(forward_hidden)
            backward_logits = linear_layer(backward_hidden)

            # (batch_size, seq_len, vocab_size)
            forward_probs = F.softmax(forward_logits, dim=-1) 
            backward_probs = F.softmax(backward_logits, dim=-1)
            
            # approx bidirectional prob P(w_i | w_1, ..., w_{i-1}, w_{i+1}, ..., w_n) = P(w_i | w_1, ..., w_{i-1}) * P(w_i | w_{i+1}, ..., w_n)
            for i in range(1, seq_len-1): # Skip [CLS] and [SEP]
                forward_p_wi = forward_probs[0, i, input_ids[i]]
                backward_p_wi = backward_probs[0, i, input_ids[i]]
                joint_p_wi = forward_p_wi * backward_p_wi

                norm_factor = (forward_probs[0, i] * backward_probs[0, i]).sum()
                p_wi = joint_p_wi / (norm_factor + 1e-10)  # avoid zero

                log_probs.append(torch.log(p_wi + 1e-10))  # avoid log(0)

        log_score = -torch.tensor(log_probs).mean()
        score = torch.exp(log_score)
        
        return score.item()
    
    def gpt2_score(self, sentence, gpt2_model=None, gpt2_tokenizer=None):
        """
        Compute the GPT-2 score of a sentence by evaluating token-wise conditional probabilities.
        :param sentence: str, input sentence
        :return: perplexity-style score (lower = more likely)
        """

        if gpt2_model is None:
            gpt2_model = self.gpt2_model
        if gpt2_tokenizer is None:
            gpt2_tokenizer = self.gpt2_tokenizer

        gpt2_model.to(self.device)
        gpt2_model.eval()

        # Tokenize and encode sentence
        tokens = gpt2_tokenizer.tokenize(sentence)
        token_ids = gpt2_tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([token_ids]).to(self.device)  # shape: (1, seq_len)

        log_probs = []

        with torch.no_grad():
            for i in range(1, input_ids.size(1)):
                # Get prefix up to i tokens (i.e., w_0 ... w_{i-1})
                prefix = input_ids[:, :i]  # shape: (1, i)

                # Predict the next token distribution (i.e., P(w_i | w_<i))
                outputs = gpt2_model(prefix)
                logits = outputs.logits  # shape: (1, i, vocab_size)
                last_token_logits = logits[0, -1]  # shape: (vocab_size,)

                probs = F.softmax(last_token_logits, dim=-1)
                target_token = input_ids[0, i]  # actual w_i
                p_wi = probs[target_token]

                log_probs.append(torch.log(p_wi + 1e-10))  # add epsilon to avoid log(0)

        # 计算平均负 log 概率并指数化（类似困惑度）
        log_score = -torch.stack(log_probs).mean()
        score = torch.exp(log_score)

        return score.item()


    def select_best_sentence(self, sentences, model_type):
        """
        Choose the best sentence based on the model type.
        
        :param sentences: list of sentences to evaluate
        :param model_type: str, type of model to use ('bert', 'elmo', 'ft_elmo', 'gpt2')
        
        :return: index of the best sentence
        """
        if model_type not in ['bert', 'elmo', 'ft_elmo', 'gpt2']:
            raise ValueError("model_type must be 'bert', 'elmo', 'ft_elmo', or 'gpt2'")

        scores = []
        if model_type == 'bert':
            compute_fn = self.bert_score
        elif model_type == 'elmo':
            compute_fn = self.elmo_score
        elif model_type == 'ft_elmo':
            compute_fn = self.ft_elmo_score
        else:
            compute_fn = self.gpt2_score

        for sentence in sentences:
            score = compute_fn(sentence)
            scores.append(score)

        best_idx = scores.index(min(scores))
        # best_sentence = sentences[best_idx]
        # best_score = scores[best_idx]

        return best_idx


if __name__ == "__main__":
    scorer = Score()

    sentences = [
        "He put a turkey into the fridge",
        "He put an elephant into the fridge",
        "He put a book into the fridge"
    ]

    print("Using BERT:")
    best_sentence = scorer.select_best_sentence(sentences, model_type='bert')
    print(f"Best sentence: {best_sentence}")
    print("\nUsing ELMo:")
    best_sentence = scorer.select_best_sentence(sentences, model_type='elmo')
    print(f"Best sentence: {best_sentence}")
    print("\nUsing gpt2:")
    best_sentence = scorer.select_best_sentence(sentences, model_type='gpt2')
    print(f"Best sentence: {best_sentence}")