import os
import sys
from app.core.config import settings
if "npu" in settings.device and ":" in settings.device:
    physical_id = settings.device.split(":")[-1]
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = physical_id
    print(f"[System] 强制隔离 NPU，仅可见物理卡: {physical_id}")
    # 物理卡变了，逻辑卡号必须归零
    TARGET_DEVICE = "npu:0"
    # 或者使用 transfer_to_npu 补丁，写成 "cuda:0"
    # settings.device = "cuda:0"
from torch_npu.contrib import transfer_to_npu # 自动迁移：将cudaAPI映射为npuAPI
import asyncio
import torch
import torch_npu
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
import whisper
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from funasr import AutoModel
from modelscope.pipelines.audio.funasr_pipeline import FunASRPipeline
from app.utils.feature_utils import id2label

# 单例缓存
_model_asr = None
_model_emotion = None
_model_online = None
_model_whisper = None
_model_speaker = None
_punct_pipeline = None
# 五何
_model_bert = None
_tokenizer = None
_bert_device = None

# print("PyTorch版本:", torch.__version__)
# print("NPU设备数量:", torch_npu.npu.device_count())
# print("当前NPU设备:", torch_npu.npu.get_device_name(0))
# print("Cuda available:", torch.cuda.is_available())
#
# print("开始")

# 线程锁
_model_lock = asyncio.Lock()


def device() -> torch.device:
    if torch.npu.is_available():
        # print(f"检测到华为 NPU 设备: {torch.npu.get_device_name(0)}")
        return torch.device(TARGET_DEVICE)
    elif torch.cuda.is_available():
        # print(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        # print("未检测到 GPU/NPU 设备，使用 CPU 进行推理")
        return torch.device("cpu")


def _resolve_bert_device() -> torch.device:
    cfg = (settings.bert_device or "").strip().lower()
    if cfg in ("", "auto"):
        return device()
    if cfg == "cpu":
        return torch.device("cpu")
    if cfg.startswith("npu"):
        target = globals().get("TARGET_DEVICE", cfg)
        return torch.device(target)
    return torch.device(cfg)


async def load_models_if_needed():
    """
    根据配置开关懒加载模型。
    """
    global _model_asr, _model_emotion, _model_online, _model_whisper, _model_speaker, _punct_pipeline

    async with _model_lock:
        if settings.open_spk and _model_asr is None:
            pass
            _model_asr = AutoModel(
                model=settings.asr_model_dir,
                punc_model=settings.punc_model_dir,
                vad_model=settings.vad_model_dir,
                spk_model=settings.spk_model_dir,
                vad_kwargs={"max_single_segment_time": 30000, "max_end_silence_time": 800},
                sentence_timestamp=True,
                ngpu=settings.ngpu,
                device = TARGET_DEVICE,
                # batch_size = 16,
                disable_update = True,
                disable_pbar = True
            )

        if settings.open_emotion and settings.open_spk and _model_emotion is None:
            _model_emotion = AutoModel(
                model=settings.emotion_model_dir,
                device=TARGET_DEVICE,
                ngpu=settings.ngpu,
                quantize=True,
                disable_update=True,
                disable_pbar=True
            )

        if settings.open_online and _model_online is None:
            _model_online = AutoModel(
                model=settings.asr_online_model_dir,
                device=TARGET_DEVICE,
                ngpu=settings.ngpu,
                quantize=True,
                disable_update=True,
                disable_pbar=True
            )

        if settings.open_online and _punct_pipeline is None:
            print(f"🚀 [Punctuation] 正在加载标点模型 (Direct Class Mode)...")

            # 优先使用直接类实例化 (最稳健)
            if FunASRPipeline is not None:
                _punct_pipeline = FunASRPipeline(
                    model=settings.asr_online_punc_model_dir,
                    device=f"cuda:{TARGET_DEVICE.split(':')[-1]}",  # AssertionError: device should be either cpu, cuda, gpu, gpu:X or cuda:X where X is the ordinal for gpu device.
                    disable_update=True,
                    disable_pbar=True
                )
            # 兜底
            else:
                _punct_pipeline = pipeline(
                    task=Tasks.punctuation,
                    model=settings.asr_online_punc_model_dir,
                    device=TARGET_DEVICE,
                    disable_update=True,
                    disable_pbar=True
                )

        if settings.open_mul_lang and _model_whisper is None:
            whisper_model_path = os.path.join(settings.whisper_model_dir, "large-v3-turbo.pt")
            _model_whisper = whisper.load_model(
                name=whisper_model_path,
                device=TARGET_DEVICE,
            )


def get_asr_model():
    return _model_asr


def get_emotion_model():
    return _model_emotion


def get_online_model():
    return _model_online


def get_whisper_model():
    return _model_whisper


def get_speaker_model():
    return _model_speaker


def get_punct_pipeline():
    return _punct_pipeline


# ---------- 五何分类 ----------
def _ensure_bert_loaded():
    global _model_bert, _tokenizer, _bert_device
    if _model_bert is None or _tokenizer is None:
        _bert_device = _resolve_bert_device()
        _model_bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=settings.bert_model_dir
        ).to(_bert_device).eval()
        _tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=settings.bert_model_tokenizer
        )


def predict_fivewh(text: str) -> tuple[str, int, float]:
    """
    教师提问5何（是何、为何、若何、由何、如何、非提问） bert预测（中文）
    """
    _ensure_bert_loaded()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(_bert_device)
    with torch.no_grad():
        logits = _model_bert(**inputs).logits
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
    return id2label[predicted.item()], predicted.item(), confidence.item()
