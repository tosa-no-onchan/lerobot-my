"""
This script demonstrates how to train ACT Policy on a real-world dataset.

1. this code: lerobot-my/AlohaTransferCube/act_training_example.py
   original code: lerobot/examples/tutorial/act/act_training_example.py

2. dataset_id: lerobot/aloha_sim_transfer_cube_human

3. function: lerobot/aloha_sim_transfer_cube_human を入力データにした、ACT model の学習 コード

$ export PYTHONPATH=$PYTHONPATH:/home/yuor-id/local/git-download/lerobot/src
$ python act_training_example.py

"""
from pathlib import Path
import torch
from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]

def main(cont_f=False):
    output_directory = Path("output/robot_learning_tutorial/act")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    #device = torch.device("mps")  # or "cuda" or "cpu"
    device = torch.device("cuda")  # or "cuda" or "cpu"

    dataset_id = "lerobot/svla_so101_pickplace"
    #dataset_id = "lerobot/aloha_sim_transfer_cube_human"

    # This specifies the inputs the model will be expecting and the outputs it will produce
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    #print('cfg:',cfg)
    #cfg: ACTConfig(n_obs_steps=1,
    # input_features={'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(6,)), 
    #   'observation.images.up': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640)), 
    #   'observation.images.side': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 480, 640))}, 
    # output_features={'action': PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>, shape=(6,))},
    # device='cuda',
    # use_amp=False, 
    # use_peft=False, 
    # push_to_hub=True, 
    # repo_id=None, 
    # private=None, 
    # tags=None, 
    # license=None, 
    # pretrained_path=None, 
    # chunk_size=100, 
    # n_action_steps=100, 
    # normalization_mapping={'VISUAL': <NormalizationMode.MEAN_STD: 'MEAN_STD'>, 'STATE': <NormalizationMode.MEAN_STD: 'MEAN_STD'>, 'ACTION': <NormalizationMode.MEAN_STD: 'MEAN_STD'>}, 
    # vision_backbone='resnet18', 
    # pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1', 
    # replace_final_stride_with_dilation=False, 
    # pre_norm=False, 
    # dim_model=512, 
    # n_heads=8, 
    # dim_feedforward=3200, 
    # feedforward_activation='relu', 
    # n_encoder_layers=4, 
    # n_decoder_layers=1, 
    # use_vae=True, 
    # latent_dim=32, 
    # n_vae_encoder_layers=4, 
    # temporal_ensemble_coeff=None, 
    # dropout=0.1, 
    # kl_weight=10.0, 
    # optimizer_lr=1e-05, 
    # optimizer_weight_decay=0.0001, 
    # optimizer_lr_backbone=1e-05)


    if cont_f:
        model_id = "output/robot_learning_tutorial/act"
        policy = ACTPolicy.from_pretrained(
            model_id,
            local_files_only=True
        ).to(device)
    else:
        policy = ACTPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)
    # To perform action chunking, ACT expects a given number of actions as targets
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }
    # Instantiate the dataset
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

    # Create the optimizer and dataloader for offline training
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    #print('optimizer:',optimizer)
    #optimizer: AdamW (
    # Parameter Group 0
    # amsgrad: False
    # betas: (0.9, 0.999)
    #capturable: False
    #decoupled_weight_decay: True
    #differentiable: False
    #eps: 1e-08
    #foreach: None
    #fused: None
    #lr: 1e-05
    #maximize: False
    #weight_decay: 0.0001
    #)

    #batch_size = 32
    #batch_size = 8
    batch_size = 4

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    # Number of training steps and logging frequency
    #training_steps = 5
    training_steps = 3000
    #training_steps = 10000
    #training_steps = 15000
    #training_steps = 25000   # batch=8
    #log_freq = 1
    #log_freq = 100
    log_freq = 200

    # Run training loop
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break
    # Save the policy checkpoint, alongside the pre/post processors
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    # Save all assets to the Hub
    #policy.push_to_hub("<user>/robot_learning_tutorial_act")
    #preprocessor.push_to_hub("<user>/robot_learning_tutorial_act")
    #postprocessor.push_to_hub("<user>/robot_learning_tutorial_act")

if __name__ == "__main__":
    cont_f=False
    main(cont_f=cont_f)
