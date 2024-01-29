from datasets import Dataset

def counterfact_preferences_ds(edit_samples, setting=['single']):
    ds = {
        "prompt": [

        ],
        "chosen": [

        ],
        "rejected": [

        ]
    }
    new_target_prompts, new_targets, old_targets = [], [], []
    for sample in edit_samples:
        new_target = " " + sample["requested_rewrite"]['target_new']['str']
        old_target = " " + sample["requested_rewrite"]['target_true']['str']
        original_prompt = sample["requested_rewrite"]["prompt"].format(
            sample["requested_rewrite"]['subject']
        )
        new_target_prompts.append(original_prompt)
        new_targets.append(new_target)
        old_targets.append(old_target)
        # extend setting to include paraphrases
        # generations or attribute edits or neightbourhood prompts
        # these can be used to test sample efficiency and locality control
        if 'paraphrases' in setting:
            new_target_prompts.extend(sample['paraphrase_prompts'])
            new_targets.extend([new_target] * len(sample['paraphrase_prompts']))
            old_targets.extend([old_target] * len(sample['paraphrase_prompts']))
        if 'with_generations' in setting:
            new_target_prompts.extend(sample['generated_prompts'])
            new_targets.extend([new_target] * len(sample['generated_prompts']))
            old_targets.extend([old_target] * len(sample['generated_prompts']))
        if 'with_locality_control' in setting:
            new_target_prompts.append(sample['attribute_prompts'])
            # keep old target
            new_targets.extend([old_target] * len(sample['attribute_prompts']))
            old_targets.extend([new_target] * len(sample['attribute_prompts']))

    ds["prompt"].extend(new_target_prompts)
    ds["chosen"].extend([new_target] * len(new_target_prompts))
    ds["rejected"].extend([old_target] * len(new_target_prompts))
    return Dataset.from_dict(
        ds
    )


def counterfact_testing_record(sample):
    request = {
        'prompt': [sample["requested_rewrite"]["prompt"].format(
            sample["requested_rewrite"]['subject']
        )],
        'target_new': sample["requested_rewrite"]['target_new']['str'],
        'ground_truth': sample["requested_rewrite"]['target_true']['str'],
        'rephrase_prompt': sample["paraphrase_prompts"],
        'locality': {
            "neighborhood": {
                'prompt': sample["attribute_prompts"],
                'ground_truth': sample["requested_rewrite"]['target_true']['str']
            }
        }
    }
    return request
