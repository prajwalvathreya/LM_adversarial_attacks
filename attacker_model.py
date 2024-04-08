from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import jiwer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
tokenizer.model_max_length = 256
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

def load_target_model():
    target_model = AutoModelForSequenceClassification.from_pretrained("hipnologo/gpt2-imdb-finetune")
    target_model.config.pad_token_id = target_model.config.eos_token_id
    return target_model

def load_attacker_model():
    attacker_model = GPT2LMHeadModel.from_pretrained('openai-community/gpt2', output_hidden_states = True)
    return attacker_model

def process_dataset(dataset, max_length = 256):
    preprocess_function = lambda examples: tokenizer(examples['text'], max_length = max_length, truncation=True, padding='max_length')
    processed_dataset = dataset.map(preprocess_function, batched = True)
    return dataset, processed_dataset

def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * nn.functional.log_softmax(shift_logits, dim=-1)).sum(-1).mean()

def calculate_word_error_rate(reference, hypothesis):
    reference = " ".join(["%d" % i for i in reference])
    hypothesis = " ".join(["%d" % i for i in hypothesis])
    return jiwer.wer(reference, hypothesis)

def load_embeddings():
    with torch.no_grad():
        tm_embeddings = target_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().to(device))
        am_embeddings = attacker_model.get_input_embeddings()(torch.arange(0, tokenizer.vocab_size).long().to(device))
    return tm_embeddings, am_embeddings

def train(target_model, attacker_model, inputs, labels, am_embeddings, tm_embeddings):
    # set up
    start_index = 1
    number_of_samples = 1
    number_of_iterations = 100
    end_index = min(start_index + number_of_samples, len(inputs))
    adv_losses, ref_losses, perp_losses = torch.zeros(end_index - start_index, number_of_iterations), torch.zeros(end_index - start_index, number_of_iterations),torch.zeros(end_index - start_index, number_of_iterations)
    batch_size = 1
    number_of_gumbel_samples = 5
    loss_margin = 5
    adversarial_log_coeffs, original_text, adversarial_text = [], [], []
    original_logits = []
    adversarial_logits = []
    token_errors = []

    lambda_similarity = 20
    lambda_perp = 1

    for idx in range(start_index, end_index):
        input_ids = inputs[idx]
        label = labels[idx]
        target_model_logits = target_model(input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)).logits.cpu()

        predicted_class = target_model_logits.argmax()
        print(f"1. The sentiment predicted by the model is: {'Positive' if predicted_class == 1 else 'Negative'}, Actual: {label}")

        with torch.no_grad():
            # get the o/p from the last layer
            original_output = attacker_model(torch.LongTensor(input_ids).to(device).unsqueeze(0)).hidden_states[-1]

            # adv parameter distribution
            adv_log_coeffs = torch.zeros(len(input_ids), tm_embeddings.size(0))
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

            if i % 5 == 0:
                print('Index: %d, Iteration %d: loss = %.4f, adv_loss = %.4f, ref_loss = %.4f, perp_loss = %.4f' % (idx + 1, i+1, total_loss.item(), loss.item(), ref_loss.item(), perp_loss.item()))

            # Gradient step
            optimizer.step()

            adv_losses[idx - start_index, i] = loss.detach().item()
            ref_losses[idx - start_index, i] = ref_loss.detach().item()
            perp_losses[idx - start_index, i] = perp_loss.detach().item()

        # Original text, logits
        text = tokenizer.decode(input_ids)
        original_text.append(text)
        original_logits.append(target_model_logits)

        # Adversarial text, logits
        with torch.no_grad():
            for j in range(number_of_gumbel_samples):
                adversarial_ids = nn.functional.gumbel_softmax(adv_log_coeffs, hard = True).argmax(1).cpu().tolist()
                adv_text = tokenizer.decode(adversarial_ids)
                x = tokenizer(adv_text, max_length = 256, truncation = True, return_tensors = "pt")
                token_errors.append(calculate_word_error_rate(adversarial_ids, x['input_ids'][0]))

                # Attack the model
                adversarial_logit = target_model(input_ids = x['input_ids'].to(device))
                adversarial_logit = adversarial_logit.logits

                # check the label given by the model
                model_label = adversarial_logit.argmax()
                if model_label != label:
                    adversarial_text.append(adv_text)
                    print("Adversarial Text:", adv_text)
                    print("Adversarial pred:", adversarial_logit.argmax(), adversarial_logit)
                    print("Original text:", text)
                    print("Original pred:", target_model_logits.argmax(), target_model_logits)
                    adversarial_logits.append(adversarial_logit)
                    break

            adversarial_log_coeffs.append(adv_log_coeffs.cpu())
            print("--------\n")
    output_file_path = "output.txt"
    # TODO: Save
    return original_text, adversarial_text
    # torch.save({
    #     'adv_log_coeffs': adversarial_log_coeffs,
    #     'adv_logits': torch.cat(adversarial_logits, 0), # size N x C
    #     'adv_losses': adv_losses,
    #     'adv_texts': adversarial_text,
    #     'clean_logits': torch.cat(original_logits, 0),
    #     'clean_texts': original_text,
    #     'labels': list(processed_dataset['test']['label'][start_index: end_index]),
    #     'perp_losses': perp_losses,
    #     'ref_losses': ref_losses,
    #     'token_error': token_errors,
    # }, output_file_path)



if __name__ == '__main__':
    # TODO: Accept model params as args
    target_model = load_target_model()
    attacker_model = load_attacker_model()
    dataset = load_dataset('imdb', split = 'test')
    processed_dataset = process_dataset(dataset)[1]
    tm_embeddings, am_embeddings = load_embeddings()
    original_text, adversarial_text = train(target_model, attacker_model, inputs = processed_dataset['input_ids'], labels = processed_dataset['label'], am_embeddings  = am_embeddings, tm_embeddings = tm_embeddings)
    print(original_text, adversarial_text)













