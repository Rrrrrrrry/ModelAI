import allennlp
import numpy as np
from allennlp.modules.elmo import Elmo, batch_to_ids

# 初始化ELMo模型
options_file = r"D:\python_program\ModelAI\algorithms\neural_networks\models\elmo/options.json"
weight_file = r"D:\python_program\ModelAI\algorithms\neural_networks\models\elmo/lm_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0)

# 准备输入文本
sentences = [['I', 'am', 'learning', 'ELMo', 'embeddings', 'so'], ['AllenNLP', 'provides', 'powerful', 'tools', '.']]
# sentence = "The dog chased the cat.I LOVE IT"
# sentences = [s.strip().split(' ') for s in sentence.split('.') if s.strip()]
print(f"sentences{sentences}")
# 将文本转换为字符ID
character_ids = batch_to_ids(sentences)
# print(f"character_ids{character_ids}")
# 生成 ELMo 嵌入
embeddings = elmo(character_ids)

elmo_embeddings = embeddings['elmo_representations'][0].detach().numpy()
print(f"elmo_embeddings{elmo_embeddings.shape}")
