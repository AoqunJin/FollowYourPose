from datasets import load_dataset, load_from_disk
from datasets import concatenate_datasets


def load_hf_data(from_saved=None, seed=42):
    if from_saved:
        dataset = load_from_disk(from_saved)
        return {"train": dataset}
    
    # TODO 
    # 12150 / 6000 (50%)
    dataset1 = load_dataset("imagefolder", data_dir='/home/ao/workspace/fs/diffusers/hf_data_p', split='train').shuffle(seed=seed).select(range(3000))
    # 344985 / 6000 (5%)
    ## dataset2 = load_dataset("imagefolder", data_dir='/home/ao/workspace/fs/diffusers/hf_data_r', split='train').shuffle(seed=seed).select(range(6000))
    # 2080
    dataset2 = load_dataset("imagefolder", data_dir='/home/ao/workspace/fs/diffusers/hf_data_rl', split='train')
    # 10321 / 6000 (50%)
    # dataset2 = load_dataset("imagefolder", data_dir='/home/ao/workspace/fs/diffusers/hf_data_r_2', split='train').shuffle(seed=seed).select(range(3000))
    # 1601
    dataset3 = load_dataset("imagefolder", data_dir='/home/ao/workspace/fs/diffusers/hf_data_f', split='train')
    # 641
    dataset4 = load_dataset("imagefolder", data_dir='/home/ao/workspace/fs/diffusers/hf_data_l', split='train')

    dataset = concatenate_datasets([dataset1, dataset2, dataset3, dataset4])
    # dataset = concatenate_datasets([dataset1, dataset2])

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

    def filter_by_additional_feature(example):
        return (example['additional_feature'] in Set_A)

    filtered_dataset = dataset.filter(filter_by_additional_feature)

    filtered_dataset.shuffle(seed=42)

    return {"train": filtered_dataset}


if __name__ == "__main__":
    dataset = load_hf_data()
    dataset["train"].save_to_disk("/home/ao/workspace/fs/diffusers/hf_data_tmp")
    