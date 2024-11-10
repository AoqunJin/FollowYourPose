from datasets import load_dataset, load_from_disk
from datasets import concatenate_datasets


def load_hf_data(from_saved=None, seed=42):
    if from_saved:
        dataset = load_from_disk(from_saved).shuffle(seed=seed).select(range(10000))
        return {"train": dataset}
    
    dataset = load_dataset("imagefolder", data_dir='/path/to/dataset', split='train')

    Set_A = set([
        'button-press-topdown-v2-goal-observable',     
        'button-press-wall-v2-goal-observable', 
        'reach-v2-goal-observable', 
        'push-wall-v2-goal-observable',
        'pick-place-v2-goal-observable',
        'assembly-v2-goal-observable', 
        'door-close-v2-goal-observable', 
        'door-lock-v2-goal-observable', 
        'drawer-open-v2-goal-observable',
        'faucet-open-v2-goal-observable', 
        'plate-slide-v2-goal-observable', 
        'plate-slide-back-side-v2-goal-observable',
        'window-open-v2-goal-observable', 
    ])

    Set_B = set([
        'button-press-topdown-wall-v2-goal-observable', 
        'button-press-v2-goal-observable', 
        'reach-wall-v2-goal-observable', 
        'push-v2-goal-observable', 
        'pick-place-wall-v2-goal-observable',
        'disassemble-v2-goal-observable',
        'door-open-v2-goal-observable',
        'door-unlock-v2-goal-observable',
        'drawer-close-v2-goal-observable',
        'faucet-close-v2-goal-observable',
        'plate-slide-back-v2-goal-observable',
        'plate-slide-side-v2-goal-observable', 
        'window-close-v2-goal-observable'
    ])

    # def filter_by_additional_feature(example):
    #     print(example.keys())
    #     return any(item in example['file_name'] for item in Set_A)

    # filtered_dataset = dataset.filter(filter_by_additional_feature)

    filtered_dataset = dataset
    filtered_dataset.shuffle(seed=seed)

    return {"train": filtered_dataset}


if __name__ == "__main__":
    dataset = load_hf_data("/path/to/dataset")
    print(dataset["train"])
    dataset["train"].save_to_disk("/path/to/dataset")
    