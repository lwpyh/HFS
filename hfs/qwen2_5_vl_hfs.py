"""
Qwen2.5-VL model wrapper with HFS (Hierarchical Frame Selection).

HFS selects 128 informative frames from 512 candidates via UCB budget allocation
and MMR per-segment frame selection, guided by CLIP text-image similarity.

Usage (after running setup.sh):
    cd hfs/lmms-eval
    PYTHONPATH="$(pwd)/../.." accelerate launch -m lmms_eval \\
        --model qwen2_5_vl_hfs \\
        --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,\\
max_pixels=6422528,attn_implementation=flash_attention_2,method=hfs \\
        --tasks videomme --batch_size 1
"""

import re
import time
from typing import List, Optional, Union

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL as Qwen2_5_VLSimple
from lmms_eval.protocol import ChatMessages

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("qwen_vl_utils not found. Install via `pip install qwen-vl-utils`")


def _load_method(method: str):
    from hfs.frame_selection import METHOD_REGISTRY

    if method not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{method}'. Available: " + ", ".join(METHOD_REGISTRY.keys())
        )
    return METHOD_REGISTRY[method]


def _extract_letter(text: str) -> str:
    for pat in [r"\(([A-D])\)", r"answer is\s+([A-D])\b", r"^([A-D])[.\s]", r"\b([A-D])\b"]:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return ""


def _extract_text(chat_messages: ChatMessages) -> str:
    parts = []
    for msg in chat_messages.messages:
        for content in msg.content:
            if content.type == "text" and content.text:
                parts.append(content.text.strip())
    return "\n".join(parts)


def _merge_frames_into_messages(
    official_msgs: list,
    hfs_out: dict,
    max_pixels: int,
    min_pixels: int,
) -> list:
    """
    Merge HFS-selected PIL frames into the official lmms-eval pipeline messages.

    Replaces the official video token with PIL frames (preserving 3D-RoPE temporal
    encoding). The official benchmark prompt text is kept unchanged.
    """
    pil_frames = []
    for msg in hfs_out.get("messages", []):
        if msg.get("role") != "user":
            continue
        for item in msg.get("content", []):
            if item.get("type") == "image":
                img = item.get("image")
                if hasattr(img, "mode"):  # PIL image
                    pil_frames.append(img)

    result = []
    for msg in official_msgs:
        if msg["role"] != "user":
            result.append(msg)
            continue

        new_content = []
        video_inserted = False
        for item in msg["content"]:
            item_type = item.get("type")

            if item_type in ("video", "image") and not video_inserted:
                if pil_frames:
                    new_content.append(
                        {
                            "type": "video",
                            "video": pil_frames,
                            "max_pixels": max_pixels,
                            "min_pixels": min_pixels,
                        }
                    )
                else:
                    new_content.append(item)
                video_inserted = True
            else:
                new_content.append(item)

        result.append({"role": "user", "content": new_content})

    return result


@register_model("qwen2_5_vl_hfs")
class Qwen2_5_VL_HFS(Qwen2_5_VLSimple):
    """
    Qwen2.5-VL with HFS (Hierarchical Frame Selection) for long video benchmarks.

    Builds on the official lmms-eval pipeline:
    - Video tokenisation: official qwen_vl_utils (full resolution, 3D-RoPE)
    - Sampling override: HFS provides 128 PIL frames via UCB+MMR selection
    - Prompt: official benchmark prompt is preserved unchanged

    Extra model_args:
        method (str): key in hfs.frame_selection.METHOD_REGISTRY — use "hfs" (default)
    """

    is_simple = False

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = "flash_attention_2",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        method: str = "hfs",
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained=pretrained,
            device=device,
            device_map=device_map,
            batch_size=batch_size,
            use_cache=use_cache,
            attn_implementation=attn_implementation,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_num_frames=max_num_frames,
            fps=fps,
            system_prompt=system_prompt,
            interleave_visuals=interleave_visuals,
            reasoning_prompt=reasoning_prompt,
            **kwargs,
        )
        self.method = method
        self._hfs = _load_method(method)
        self._running_correct = 0
        self._running_total = 0
        eval_logger.info(f"[qwen2_5_vl_hfs] method={method}")

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = (
            len(requests) // self.batch_size
            if len(requests) % self.batch_size == 0
            else len(requests) // self.batch_size + 1
        )
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)
            chat_messages: List[ChatMessages] = [
                ChatMessages(
                    messages=doc_to_messages[idx](self.task_dict[task[idx]][split[idx]][ids])
                )
                for idx, ids in enumerate(doc_id)
            ]
            gen_kwargs = all_gen_kwargs[0]

            video_kwargs = {
                "max_pixels": self.max_pixels,
                "min_pixels": self.min_pixels,
                "nframes": self.max_num_frames,
            }

            batched_messages = []
            for cm_idx, cm in enumerate(chat_messages):
                _, videos, _ = cm.extract_media()
                official_msgs = cm.to_hf_messages(video_kwargs=video_kwargs)

                if not videos:
                    batched_messages.append(official_msgs)
                    continue

                vpath = videos[0]

                try:
                    doc = self.task_dict[task[cm_idx]][split[cm_idx]][doc_id[cm_idx]]
                    if isinstance(doc, dict) and "question" in doc:
                        question = doc["question"] + "\n" + "\n".join(doc.get("options", []))
                    else:
                        question = _extract_text(cm)
                except Exception:
                    question = _extract_text(cm)

                try:
                    hfs_out = self._hfs.run(vpath, question)
                    hf_msgs = _merge_frames_into_messages(
                        official_msgs, hfs_out, self.max_pixels, self.min_pixels
                    )
                except Exception as e:
                    eval_logger.warning(f"HFS '{self.method}' failed: {e}. Falling back to official pipeline.")
                    hf_msgs = official_msgs

                batched_messages.append(hf_msgs)

            texts = self.processor.apply_chat_template(
                batched_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(batched_messages)

            padding_side = "left" if self.batch_size > 1 else "right"
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                padding_side=padding_side,
                return_tensors="pt",
            )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            default_gen_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
            }
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None
                current_gen_kwargs.pop("top_k", None)

            start_time = time.time()
            cont = self.model.generate(
                **inputs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=current_gen_kwargs["do_sample"],
                temperature=current_gen_kwargs["temperature"],
                top_p=current_gen_kwargs["top_p"],
                num_beams=current_gen_kwargs["num_beams"],
                max_new_tokens=current_gen_kwargs["max_new_tokens"],
                use_cache=self.use_cache,
            )
            end_time = time.time()

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)
            ]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                res.append(
                    GenerationResult(
                        text=ans,
                        token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i])),
                    )
                )
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)

                try:
                    doc = self.task_dict[task[i]][split[i]][doc_id[i]]
                    gt = doc.get("answer", "").strip().upper() if isinstance(doc, dict) else ""
                except Exception:
                    gt = ""
                pred = _extract_letter(ans)
                if gt:
                    self._running_correct += int(pred == gt)
                    self._running_total += 1
                mark = "✓" if (gt and pred == gt) else ("✗" if gt else "?")
                eval_logger.info(
                    f"[{self._running_total}/{len(requests)}] {mark} "
                    f"pred={pred or '?'} gt={gt or '?'}  "
                    f"acc={self._running_correct / max(self._running_total, 1):.1%}  "
                    f"method={self.method}"
                )

            pbar.update(1)

        res = re_ords.get_original(res)

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )
        pbar.close()
        return res
