from datasets import load_from_disk,concatenate_datasets


PATH_1="dataset_packed/train_1"
PATH_2="dataset_packed/train_2"
PATH_3="dataset_packed/train_3"
PACKED_PATH="dataset_packed/train"
dataset_part_1=load_from_disk(PATH_1)
dataset_part_2=load_from_disk(PATH_2)
dataset_part_3=load_from_disk(PATH_3)

dataset=concatenate_datasets([dataset_part_1,dataset_part_2,dataset_part_3])

dataset.save_to_disk(PACKED_PATH,max_shard_size="3GB")