
from edits_as_preferences.dataset import counterfact_testing_record
from edits_as_preferences.easy_edit_evaluate import compute_edit_quality


class Hparams:
    alg_name = "DPO"
    beta = 0.1
    max_length = 50


def evaluate_model(model, model_name, edit_sample, tokenizer, device):
    print("Evaluating model")
    test_model = model.to(device)
    test_model.eval()
    request = counterfact_testing_record(edit_sample)
    hparams = Hparams()
    performance = compute_edit_quality(
        test_model, model_name, hparams, tokenizer, request, device
    )
    ## test generations
    print("Generating from model")
    out = test_model.generate(
        **tokenizer(request['prompt'], return_tensors="pt").to(device),
        max_length=50,
        num_return_sequences=1
    )
    rewrite_out = tokenizer.decode(out[0], skip_special_tokens=True)
    print(rewrite_out)
    performance['rewrite_acc_gen'] = 1 if request['target_new'].lower() in rewrite_out.lower() else 0
    performance['rewrite_gt_acc_gen'] = 1 if request['ground_truth'].lower() in rewrite_out.lower() else 0
    outs = test_model.generate(
        **tokenizer(request['rephrase_prompt'], padding=True, return_tensors="pt").to(device),
        max_length=50,
        num_return_sequences=1
    )
    rephrase_outs = tokenizer.batch_decode(outs, skip_special_tokens=True)
    print(rephrase_outs)
    performance['rephrase_acc_gen'] = sum([1 if request['target_new'].lower() in rephrase_out.lower() else 0 for rephrase_out in rephrase_outs]) / len(rephrase_outs)
    performance['rephrase_gt_acc_gen'] = sum([1 if request['ground_truth'].lower() in rephrase_out.lower() else 0 for rephrase_out in rephrase_outs]) / len(rephrase_outs)
    outs = test_model.generate(
        **tokenizer(request['locality']["neighborhood"]["prompt"], return_tensors="pt", padding=True).to(device),
        max_length=50,
        num_return_sequences=1
    )
    locality_outs = tokenizer.batch_decode(outs, skip_special_tokens=True)
    print(locality_outs)
    performance['locality_new_acc_gen'] = sum([1 if request['target_new'].lower() in locality_out.lower() else 0 for locality_out in locality_outs]) / len(locality_outs)
    performance['locality_gt_acc_gen'] = sum([1 if request['ground_truth'].lower() in locality_out.lower() else 0 for locality_out in locality_outs]) / len(locality_outs)
    print(performance)
    print("Done")
    # save performance
    return performance
