


import gradio as gr
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from download_from_drive import download_file
from uuid import uuid4

# === Download Required Files from Google Drive ===
# download_file("1LauVdBy41kZvldZqK3dQIDApKdLqXiXs", "DNAEncoder.py")
# download_file("12kRAe9nmU-8k20Q32VZYFXkLTiUXFEZT", "Preprocessor.py")
# download_file("18erKly_wBTw_wu0y4eRfTEKtOL0Hs92k", "PretrainedBERT.py")
# download_file("1_bvtrRupabYwHSoXPOwChL-jsJJEj-vV", "inference.py")
# download_file("1BSmhgZr394cNMyvoij1zCNDcwe0QwAsn", "model.pt")



# === Import your modules ===
from DNAEncoder import ConvertDNALabelEncoder
from Preprocessor import PreprocessLLMData
from PretrainedBERT import initialize_pretrained_bert
from inference import inference

# === Model Loading ===
bert_classifier, optimizer = initialize_pretrained_bert(2)
bert_classifier.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
bert_classifier.eval()

# === DNA Encoder ===
convertDNALabelEncoder = ConvertDNALabelEncoder()

# === Prediction Function ===
def predict_dna(seq: str):
    try:
        seq = seq.replace('\n', '')
        genome_length = len(seq)
        chunk_size = 250
        chunks = []

        # Slice genome into 250bp chunks
        for i in range(0, genome_length, chunk_size):
            chunk_seq = seq[i:i+chunk_size]

            # Create temp CSV for chunk
            temp_csv = f"temp_input.csv"
            df = pd.DataFrame({'seq': [chunk_seq], 'label': [0]})
            df.to_csv(temp_csv, index=False)

            # Encode sequence
            df_encoded, y = convertDNALabelEncoder.convert_dna_string_to_dna_labelencoder(
                temp_csv, 'seq', 'label'
            )
            os.remove(temp_csv)

            # Preprocess
            preprocessor = PreprocessLLMData(df_encoded[0], y[0])
            _, _, y_list, test_dataloader = preprocessor.preprocess()

            # Inference
            probs, labels_list, _ = inference(bert_classifier, test_dataloader, device='cpu')
            predicted_class = int(probs >= 0.5)

            # Append chunk prediction
            chunks.append({
                "start": i,
                "end": min(i + chunk_size, genome_length),
                "score": float(probs),
                "label": "pathogenic" if predicted_class == 1 else "nonpathogenic"
            })

        # Global summary (aggregate prediction)
        avg_score = float(np.mean([c["score"] for c in chunks]))
        majority_class = max(set([c["label"] for c in chunks]),
                             key=[c["label"] for c in chunks].count)

        # Example GFF (replace with real annotations if available)
        annotations_gff = (
            "##gff-version 3\n"
            "virus . gene 1 500 . + . ID=gene1;Name=ORF1\n"
            "virus . gene 800 1500 . - . ID=gene2;Name=ORF2"
        )
        print(chunks[0])

        # ✅ Flat JSON (like old version, but extended)
        return {
            "input_sequence": seq[:30]+"...",
            "confidence": avg_score,
            "label_name": "Human-Pathogenic" if majority_class == "pathogenic" else "Non-Human",
            "chunks": chunks,
            "genome_length": genome_length,
            "annotations_gff": annotations_gff,
            "actual": majority_class,
            "predicted": majority_class,
        }

    except Exception as e:
        return {"error": str(e)}


# === Gradio Interface ===
api = gr.Interface(
    fn=predict_dna,
    inputs=gr.Textbox(label="DNA Sequence"),
    outputs="json",
    api_name="/predict_dna"
   # allow_flagging="never"
)
api.queue(api_open=True)
# with gr.Blocks() as demo:
#     def fn(a: int, b: int, c: list[str]) -> tuple[int, str]:
#         return a + b, c[a:b]
#     gr.api(fn, api_name="add_and_slice")
if __name__ == "__main__":
    api.launch(    share=True,
    debug=True,
    show_api=True,
    ssr_mode=False) #share=True, debug=True, show_api=True) show_api=True,share=True,ssr_mode=False











# def add_and_slice(a: int, b: int, c: list[int]) -> tuple[int, list[int]]:
#     return a + b, c[a:b]

# # === Gradio App ===
# with gr.Blocks() as demo:
#     # Visual UI for predict_dna
#     gr.Interface(
#         fn=predict_dna,
#         inputs=gr.Textbox(label="DNA Sequence"),
#         outputs="json",
#         allow_flagging="never"
#     )

#     # REST API for add_and_slice
#     demo.api(fn=add_and_slice, 
#              inputs=[gr.Number(), gr.Number(), gr.Textbox()], 
#              outputs=["number", "json"], 
#              api_name="/add_and_slice")

# if __name__ == "__main__":
#     demo.launch(share=True, show_api=True)

    

# import gradio as gr
# import torch
# import pandas as pd
# import numpy as np

# from download_from_drive import download_file

# # === Download Required Files from Google Drive ===
# download_file("1LauVdBy41kZvldZqK3dQIDApKdLqXiXs", "DNAEncoder.py")
# download_file("1C_c5Zf074PEh0YD3srurZt8FKz-tAgUB", "Preprocessor.py")
# download_file("18erKly_wBTw_wu0y4eRfTEKtOL0Hs92k", "PretrainedBERT.py")
# download_file("1q4NBD4dfx2xoZQgyOyLfMsOgv2ByUv-y", "inference.py")
# download_file("1BSmhgZr394cNMyvoij1zCNDcwe0QwAsn", "model.pt")

# # === Import your modules ===
# from DNAEncoder import ConvertDNALabelEncoder
# from Preprocessor import PreprocessLLMData
# from PretrainedBERT import initialize_pretrained_bert
# from inference import inference

# # === Model Loading ===
# bert_classifier, optimizer = initialize_pretrained_bert(2)
# bert_classifier.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")), strict=False)
# bert_classifier.eval()

# # === DNA Encoder ===
# convertDNALabelEncoder = ConvertDNALabelEncoder()

# # === Prediction Function ===
# def predict_dna(seq):
#     try:
#         # Create DataFrame for input sequence
#         df = pd.DataFrame({'seq': [seq], 'label': [0]})  # dummy label

#         # Apply Label Encoding
#         df_encoded, y = convertDNALabelEncoder.convert_dna_string_to_dna_labelencoder(df, 'seq', 'label')

#         # Preprocess
#         preprocessor = PreprocessLLMData(df_encoded[0], y[0])
#         inputs_list, masks_list, y_list, test_dataloader = preprocessor.preprocess()

#         # Run inference
#         probs, labels_list, logits = inference(bert_classifier, test_dataloader, device='cpu')

#         predicted_class = int(probs >= 0.5)
#         actual_class = int(np.argmax(y_list[0]))
#         probability = float(probs)

#         return {
#             "input_sequence": seq,
#             "actual": actual_class,
#             "predicted": predicted_class,
#             "confidence": probability,
#             "label_name": "Human-Pathogenic" if predicted_class == 1 else "Non-Human"
#         }

#     except Exception as e:
#         return {"error": str(e)}

# # === Gradio Interface ===
# api = gr.Interface(
#     fn=predict_dna,
#     inputs=gr.Textbox(label="DNA Sequence"),
#     outputs="json",
# )

# if __name__ == "__main__":
#     api.launch(server_name="0.0.0.0", server_port=7860, show_api=True)
