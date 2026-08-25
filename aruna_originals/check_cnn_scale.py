import torch
import s3fs

bucket_out = "walkky-ml"
base_prefix = "aruna-files/vqvae_final_12lead_vqenc/vqvae"
# base_prefix = "aruna-files/vqvae_12lead_vqenc_4h5/vqvae"
in_channels = 12

s3 = s3fs.S3FileSystem()
key = f"{bucket_out}/{base_prefix}/bert_pretraining/bert_model_nleads_{in_channels}.pt"

with s3.open(key, "rb") as f:
    state_dict = torch.load(f, map_location="cpu")

print(state_dict["embedding.cnn_scale"])

# Result: tensor(2.5235)
# tensor(1.1455)