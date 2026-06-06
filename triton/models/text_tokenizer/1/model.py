import os
import numpy as np
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args:dict) -> None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
        local_tokenizer_dir = os.path.join(model_dir, 'tokenizer')
        model_name = "intfloat/multilingual-e5-small"

        if os.path.isdir(local_tokenizer_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_tokenizer_dir, local_files_only=True
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.max_length = 64

    def execute(self, requests: list) -> list:
        responses = []

        for request in requests:
            raw_texts = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            texts = [t.decode("utf-8") for t in raw_texts.flatten()]

            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='np'
            )

            input_ids = encoded['input_ids'].astype(np.int64)
            attention_mask = encoded['attention_mask'].astype(np.int64)

            out_tensors = [
                pb_utils.Tensor('input_ids', input_ids),
                pb_utils.Tensor('attention_mask', attention_mask)
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))
        
        return responses