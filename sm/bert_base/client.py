import numpy as np
import tritonclient.http
from scipy.special import softmax
from transformers import BertModel, BertTokenizer, TensorType

def topK(x, k, axis=0):
    idx = np.argpartition(x, -k)[:, -k:]
    indices = idx[:, np.argsort((-x)[:, idx][0])][0]
    return indices

#BertTokenizer.from_pretrained("bert-base-uncased").save_pretrained("bert-cache")
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('./bert-cache')

model_name = "bert_base"
url = "127.0.0.1:8000"
model_version = "1"
batch_size = 1

text = "The goal of life is [MASK]."
tokens = tokenizer(text=text, return_tensors=TensorType.NUMPY)

triton_client = tritonclient.http.InferenceServerClient(url=url, verbose=False)
assert triton_client.is_model_ready(
    model_name=model_name, model_version=model_version
), f"model {model_name} not yet ready"

input_ids = tritonclient.http.InferInput(name="input_ids", shape=(batch_size, 9), datatype="INT64")
token_type_ids = tritonclient.http.InferInput(name="token_type_ids", shape=(batch_size, 9), datatype="INT64")
attention = tritonclient.http.InferInput(name="attention_mask", shape=(batch_size, 9), datatype="INT64")
model_output = tritonclient.http.InferRequestedOutput(name="output", binary_data=False)

input_ids.set_data_from_numpy(tokens['input_ids'] * batch_size)
token_type_ids.set_data_from_numpy(tokens['token_type_ids'] * batch_size)
attention.set_data_from_numpy(tokens['attention_mask'] * batch_size)

response = triton_client.infer(
    model_name=model_name,
    model_version=model_version,
    inputs=[input_ids, token_type_ids, attention],
    outputs=[model_output],
)

token_logits = response.as_numpy("output")
mask_token_index = np.where(tokens['input_ids'] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
mask_token_logits = softmax(mask_token_logits, axis=1)

top_5_indices = topK(mask_token_logits, 5, axis=1)
top_5_values = mask_token_logits[:, top_5_indices][0]

top_5_tokens = zip(top_5_indices[0].tolist(), top_5_values[0].tolist())

lista=[]
for token, score in top_5_tokens:
    print(text.replace(tokenizer.mask_token, tokenizer.decode([token])), f"(score: {score})")
