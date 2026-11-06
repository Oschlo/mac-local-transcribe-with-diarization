"""Konverter en Whisper-modell i HF-format til mlx-whisper, fra originalrepoet.

    .venv/bin/python bench/hf2mlx.py KBLab/kb-whisper-large work/models/kb-whisper-large-mlx

Finnes fordi tredjepartsportene ikke oppgir kilde. `load_model` bruker
`model.update`, som ikke er streng: en vekt med feil navn blir stående på
initverdien uten feil. Derfor sjekkes nøkler og former mot en tom modell før
noe skrives.
"""
import json, os, re, sys

import mlx.core as mx
from mlx.utils import tree_flatten
from huggingface_hub import hf_hub_download
from mlx_whisper import whisper

RENAME = [  # rekkefølgen betyr noe: final_layer_norm i blokk før encoder.layer_norm
    (r"^model\.", ""),
    (r"\.layers\.", ".blocks."),
    (r"\.self_attn_layer_norm\.", ".attn_ln."),
    (r"\.encoder_attn_layer_norm\.", ".cross_attn_ln."),
    (r"\.self_attn\.", ".attn."),
    (r"\.encoder_attn\.", ".cross_attn."),
    (r"\.k_proj\.", ".key."), (r"\.q_proj\.", ".query."),
    (r"\.v_proj\.", ".value."), (r"\.out_proj\.", ".out."),
    (r"\.fc1\.", ".mlp1."), (r"\.fc2\.", ".mlp2."),
    (r"(blocks\.\d+)\.final_layer_norm\.", r"\1.mlp_ln."),
    (r"^encoder\.layer_norm\.", "encoder.ln_post."),
    (r"^decoder\.layer_norm\.", "decoder.ln."),
    (r"^decoder\.embed_tokens\.", "decoder.token_embedding."),
    (r"^decoder\.embed_positions\.weight$", "decoder.positional_embedding"),
]
DROP = ("encoder.embed_positions.weight",  # sinusoider, regnes ut i mlx
        "proj_out.weight")                 # bundet til token_embedding


def main(repo, out):
    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    dims = dict(n_mels=cfg["num_mel_bins"], n_audio_ctx=cfg["max_source_positions"],
                n_audio_state=cfg["d_model"], n_audio_head=cfg["encoder_attention_heads"],
                n_audio_layer=cfg["encoder_layers"], n_vocab=cfg["vocab_size"],
                n_text_ctx=cfg["max_target_positions"], n_text_state=cfg["d_model"],
                n_text_head=cfg["decoder_attention_heads"], n_text_layer=cfg["decoder_layers"])
    src = mx.load(hf_hub_download(repo, "model.safetensors"))
    w = {}
    for k, v in src.items():
        k = k.removeprefix("model.") if k.startswith("model.") else k
        if k in DROP:
            continue
        for a, b in RENAME:
            k = re.sub(a, b, k)
        if k.endswith(("conv1.weight", "conv2.weight")):
            v = v.transpose(0, 2, 1)  # torch (ut, inn, k) → mlx (ut, k, inn)
        w[k] = v.astype(mx.float16)
    want = dict(tree_flatten(whisper.Whisper(whisper.ModelDimensions(**dims), mx.float16).parameters()))
    want.pop("alignment_heads")  # bygges av modellen selv, ikke en vekt
    missing, extra = want.keys() - w.keys(), w.keys() - want.keys()
    bad = [k for k in want.keys() & w.keys() if want[k].shape != w[k].shape]
    if missing or extra or bad:
        sys.exit(f"mangler {sorted(missing)[:5]}, ekstra {sorted(extra)[:5]}, feil form {bad[:5]}")
    os.makedirs(out, exist_ok=True)
    mx.save_safetensors(os.path.join(out, "weights.safetensors"), w)
    json.dump({**dims, "model_type": "whisper"}, open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"{len(w)} tensorer, alle nøkler og former stemmer → {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
