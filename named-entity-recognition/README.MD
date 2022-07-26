# Named Entity Recognition with T5

[![Open in Layer](https://app.layer.ai/assets/badge.svg)](https://app.layer.ai/layer/named-entity-recognition) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/named-entity-recognition/named_entity_recognition.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/named-entity-recognition)

This is a named entity recognition example based off of [T5](https://github.com/google-research/text-to-text-transfer-transformer).

## How to use

Make sure you have the latest version of Layer
```
!pip install layer -q
```

Then, you can fetch the trained model from Layer and start segmenting images

```python
import layer
import torch
from torch import cuda

model = layer.get_model('layer/named-entity-recognition/models/t5-named-entity-recognition').get_train().to('cuda')
t5_tokenizer = layer.get_model("layer/t5/models/t5-small-tokenizer").get_train()

def get_entities(sentence):
  input_ids = t5_tokenizer.encode(f"recognize named entities: {sentence}")
  test = torch.LongTensor([input_ids]).to("cuda")
  outputs = model.generate(test,  max_length=1024)
  return t5_tokenizer.decode(outputs[0].tolist())

print(get_entities("The capital city of France is Paris"))
```
