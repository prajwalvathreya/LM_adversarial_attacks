from transformers import GPT2LMHeadModel, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import jiwer

def load_target_model():
    target_model = AutoModelForSequenceClassification.from_pretrained("hipnologo/gpt2-imdb-finetune")
    target_model.config.pad_token_id = target_model.config.eos_token_id
    return target_model

def load_attacker_model():
    attacker_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2', output_hidden_states = True)
    return attacker_model

def process_dataset(tokenizer, text, max_length = 256):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors = "pt")
    return input

def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * nn.functional.log_softmax(shift_logits, dim=-1)).sum(-1).mean()

def calculate_word_error_rate(reference, hypothesis):
    reference = " ".join(["%d" % i for i in reference])
    hypothesis = " ".join(["%d" % i for i in hypothesis])
    return jiwer.wer(reference, hypothesis)

def load_embeddings(tokenizer, device, target_model, attacker_model):
    with torch.no_grad():
        tm_embeddings = target_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().to(device))
        am_embeddings = attacker_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().to(device))
    return tm_embeddings, am_embeddings

def train(target_model, attacker_model, input_ids, label, am_embeddings, tm_embeddings, device, tokenizer):
    
    # out variables
    adversarial_pred = None
    adversarial_text = None
    original_text = None
    original_pred = None
    
    # set up
    number_of_iterations = 100
    batch_size = 1
    number_of_gumbel_samples = 5
    loss_margin = 5
    adversarial_log_coeffs, original_text, adversarial_text = [], [], []
    
    token_errors = []

    lambda_similarity = 20
    lambda_perp = 1
    
    print("device", device)

    # for idx in range(start_index, end_index):
    print(input_ids)
    target_model_logits = target_model(input_ids = input_ids.to(device)).logits

    predicted_class = target_model_logits.argmax()
    print(f"1. The sentiment predicted by the model is: {'Positive' if predicted_class == 1 else 'Negative'}, Actual: {label}")

    with torch.no_grad():
        # get the o/p from the last layer
        original_output = attacker_model(torch.LongTensor(input_ids).to(device)).hidden_states[-1]

        # adv parameter distribution
        adv_log_coeffs = torch.zeros(len(input_ids[0]), tm_embeddings.size(0))
        print(len(input_ids[0]), adv_log_coeffs)
        indices = torch.arange(adv_log_coeffs.size(0)).long()
        adv_log_coeffs[indices, torch.LongTensor(input_ids)] = 15
        adv_log_coeffs = adv_log_coeffs.to(device)
        adv_log_coeffs.requires_grad = True

    optimizer = torch.optim.Adam([adv_log_coeffs], lr = 3e-1)
    for i in range(number_of_iterations):
        optimizer.zero_grad()
        coeffs = nn.functional.gumbel_softmax(adv_log_coeffs.unsqueeze(0).repeat(batch_size, 1, 1), hard = False)
        tm_inputs_embeds = (coeffs @ tm_embeddings[None, :, :])
        tm_pred = target_model(input_ids = None, inputs_embeds = tm_inputs_embeds).logits

        top_preds = tm_pred.sort(descending=True)[1]
        correct_preds = (top_preds[:, 0] == label).long()
        indices = top_preds.gather(1, correct_preds.view(-1, 1))
        loss = (tm_pred[:, label] - tm_pred.gather(1, indices).squeeze() + loss_margin).clamp(min=0).mean()

        # Add similarity constraint
        am_input_embeds = (coeffs @ am_embeddings[None, :, :])
        am_pred = attacker_model(inputs_embeds = am_input_embeds)
        am_output = am_pred.hidden_states[-1]
        cosine_similarity = (am_output * original_output).sum(1) / am_output.norm(2, 1) / original_output.norm(2, 1)
        ref_loss = -lambda_similarity  * cosine_similarity.mean()

        # Add the fluency constraint
        perp_loss = lambda_perp * log_perplexity(am_pred.logits, coeffs)

        total_loss = loss + ref_loss + perp_loss
        total_loss.backward()

        # Gradient step
        optimizer.step()

    # Original text, logits
    text = tokenizer.decode(input_ids.squeeze(0))
    original_text = text
    original_pred = target_model_logits

    # Adversarial text, logits
    with torch.no_grad():
        for j in range(number_of_gumbel_samples):
            adversarial_ids = nn.functional.gumbel_softmax(adv_log_coeffs, hard = True).argmax(1).tolist()
            adversarial_text = tokenizer.decode(adversarial_ids)
            x = tokenizer(adversarial_text, max_length = 256, truncation = True, return_tensors = "pt")
            token_errors.append(calculate_word_error_rate(adversarial_ids, x['input_ids'][0]))

            # Attack the model
            adversarial_logit = target_model(input_ids = x['input_ids'].to(device))
            adversarial_logit = adversarial_logit.logits
            adversarial_pred = adversarial_logit.argmax()
            
            # check the label given by the model
            if adversarial_pred != label:
                break

        adversarial_log_coeffs.append(adv_log_coeffs)
    
    return original_pred, original_text, adversarial_text, adversarial_pred











