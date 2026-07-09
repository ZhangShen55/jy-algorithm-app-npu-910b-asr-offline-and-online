import os
import re
import time
import uuid
import tempfile
import logging
import asyncio
import torchaudio
from copy import deepcopy
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from app.entity.data import AsrRequestParams, get_asr_params
from app.core.config import settings
from app.core.models import (
    get_asr_model, get_emotion_model
)
from app.core.concurrency import acquire_gpu_slot, generate_with_gpu_lock
from app.utils.audio_utils import preprocess_audio, write_audio_bytes_to_temp_file, crop_audio
from app.utils.feature_utils import (
    extract_features,
    identify_teacher,
    convert_role_ids,
    calculate_speech_rate,
    calculate_speed_info,
)
from app.utils.asr_stats import update_stat, update_fail_task

logger = logging.getLogger(__name__)

router = APIRouter()

EMOTION_LABEL_MAP = {
    "中立/neutral": "平淡",
    "其他/other": "平淡",
    "<unk>": "平淡",
    "开心/happy": "积极",
    "吃惊/surprised": "兴奋",
    "生气/angry": "强调",
    "难过/sad": "思考",
    "厌恶/disgusted": "思考",
    "恐惧/fearful": "疑问",
}


def _generate_task_id() -> str:
    return str(uuid.uuid4())


def _map_emotion_label(label: str | None) -> str:
    if not label:
        return "平淡"
    return EMOTION_LABEL_MAP.get(label, "平淡")


def _select_top_emotion_label(res_emotion) -> str | None:
    if not res_emotion:
        return None
    first = res_emotion[0] if isinstance(res_emotion, list) else res_emotion
    if not isinstance(first, dict):
        return None
    scores = first.get("scores") or []
    labels = first.get("labels") or []
    if not scores or not labels:
        return None
    max_idx = scores.index(max(scores))
    if max_idx >= len(labels):
        return None
    return labels[max_idx]


def _format_spk_role(spk) -> str | None:
    if spk is None or spk == "":
        return None
    spk_text = str(spk)
    return spk_text if spk_text.startswith("spk_") else f"spk_{spk_text}"


def _convert_segments_to_spk_roles(segments: List[dict]) -> List[dict]:
    role_counts = {}
    for segment in segments:
        role = _format_spk_role(segment.get("role"))
        if role:
            role_counts[role] = role_counts.get(role, 0) + 1

    fallback_role = max(role_counts, key=role_counts.get) if role_counts else "spk_0"
    for segment in segments:
        segment["role"] = _format_spk_role(segment.get("role")) or fallback_role
    return segments


@router.post("/v1.1.8/seacraft_asr")
async def api_asr_mul(request: AsrRequestParams = Depends(get_asr_params)):
    """
    语音转写 + 小语种(whisper) + 身份判定 + (可选)情绪识别 + (可选)音轨分离
    保持原接口契约不变
    """
    if request.audioFile is None:
        logger.error("音频为空")
        return {"msg": "音频文件不能为空", "code": 4001}

    tmp_paths: List[str] = []
    try:
        logger.info(f"request: \n{request}")
        # funasr 识别参数
        param_dict = {
            "batch_size_s": 300,
            "language": "auto",
            "spk_model": "open"
        }
        local_param_dict = deepcopy(param_dict)

        # 处理 hotWords
        if not request.hotWords:
            hotword_str = ""
        else:
            if len(request.hotWords) == 1 and isinstance(request.hotWords[0], str) and "," in request.hotWords[0]:
                request.hotWords = request.hotWords[0].split(",")
            hotword_str = " ".join(request.hotWords)
        local_param_dict["hotword"] = "" if settings.ban_hotword else hotword_str

        # 读取音频
        try:
            start_time = time.perf_counter()
            content = await request.audioFile.read()
            suffix = request.audioFile.filename.split(".")[-1].lower()
            filename = request.audioFile.filename
            if suffix == "m4a":
                suffix = "mp4"
            audio_bytes = await preprocess_audio(content, suffix, force_resample=False)
            if len(audio_bytes) < 1024:
                logger.error(f"音频文件过小，疑似空白音频：{request.audioFile.filename}, 大小：{len(audio_bytes)}")
                update_fail_task()
                raise HTTPException(status_code=400, detail="音频文件过小，疑似空白音频")
            load_audio_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"加载音频文件耗时：{load_audio_time_ms:.2f} ms")
        except Exception as e:
            logger.error(f"读取音频文件发生错误，文件名:{request.audioFile.filename}, 错误信息：\n{e}")
            raise HTTPException(status_code=400, detail=f"读取音频文件发生错误，文件名:{request.audioFile.filename}")


        common_langs = ["fr"]  # 上交大 fr
        audioFile_bytes = content

        # 保存为临时文件（供 whisper / 说话人分离使用）
        tmp_path = write_audio_bytes_to_temp_file(audioFile_bytes, file_name=filename, suffix=f"{suffix}")
        tmp_paths.append(tmp_path)

        # 读取音频 tensor
        audio_tensor, sample_rate = await asyncio.to_thread(torchaudio.load, tmp_path, backend="ffmpeg")

        task_id = f"{_generate_task_id()}_{filename}"

        # 多语种识别不启用写死
        if request.language is None or request.language != "auto":
            request.language = "auto"

        # ---------- Paraformer 路线 ----------
        start_time = time.perf_counter()
        model_asr = get_asr_model()

        if request.showSpk:
            if not settings.open_spk or model_asr is None:
                logger.error(f"open_spk={settings.open_spk} 未开启音轨分离模型 文件名:{filename} 转写失败")
                update_fail_task()
                raise HTTPException(status_code=400, detail=f"open_spk={settings.open_spk} 未开启音轨分离模型")
                # return {"msg": f"open_spk={settings.open_spk} 未开启音轨分离模型", "code": 4003}

            try:
                async with acquire_gpu_slot(task_id=task_id):
                    rec_results = await generate_with_gpu_lock(
                        model_asr, input=audio_bytes, is_final=True, **local_param_dict
                    )
            except (asyncio.TimeoutError, IndexError, HTTPException):
                update_fail_task()
                raise HTTPException(status_code=400, detail="请求过多或超时，请稍后再试")
                # return {"msg": "请求过多或超时，请稍后再试", "code": 4004}

            gpu_time_ms = (time.perf_counter() - start_time) * 1000

            if len(rec_results) == 0:
                update_fail_task()
                return {"text": "", "sentences": [], "code": 0}
            elif len(rec_results) == 1:
                rec_result = rec_results[0]
                text = rec_result["text"]
                if "sentence_info" not in rec_result or not rec_result["sentence_info"]:
                    logger.error(f"音频文件为空或未检测到任何人声，可能是静音，文件名:{filename} 转写失败")
                    raise HTTPException(status_code=400, detail="音频文件为空或未检测到任何人声")
                    # return {"msg": "音频文件为空或未检测到任何人声", "code": 4008}

                segments = []
                for segment in rec_result["sentence_info"]:
                    segment_words = []
                    emotion = None
                    if request.wordTimestamps:
                        words = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z']+|\b[a-zA-Z']+\b", segment["text"])
                        timestamps = segment["timestamp"]
                        for i, word_text in enumerate(words):
                            if i < len(timestamps):
                                segment_words.append({
                                    "bg": f"{timestamps[i][0] / 1000:.2f}",
                                    "ed": f"{timestamps[i][1] / 1000:.2f}",
                                    "word_text": word_text
                                })

                    if request.showEmotion:
                        if not settings.open_emotion or get_emotion_model() is None:
                            logger.error(f"open_emotion={settings.open_emotion} 未开启情感分析模型 文件名:{filename} 转写失败")
                            update_fail_task()
                            raise HTTPException(status_code=400, detail=f"open_emotion={settings.open_emotion} 未开启情感分析模型")
                            # return {"msg": f"open_emotion={settings.open_emotion} 未开启情感分析模型", "code": 4003}

                        seg_len_ms = segment["end"] - segment["start"]
                        if seg_len_ms > 10000:
                            emotion = "平淡"
                        else:
                            # 音频片段识别
                            cropped = crop_audio(audio_tensor, segment["start"] + 0.1, segment["end"] + 0.1, sample_rate)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as segf:
                                seg_path = segf.name
                                tmp_paths.append(seg_path)
                            torchaudio.save(seg_path, cropped, sample_rate)
                            res_emotion = get_emotion_model().generate(seg_path, granularity="utterance", extract_embedding=False)
                            emotion_label = _select_top_emotion_label(res_emotion)
                            emotion = _map_emotion_label(emotion_label)

                    speed = calculate_speech_rate(
                        segment["text"],
                        segment["start"] / 1000,
                        segment["end"] / 1000,
                        settings.speech_rate_factor,
                    )
                    item = {
                        "segment_text": segment["text"],
                        "bg": f"{segment['start'] / 1000:.2f}",
                        "ed": f"{segment['end'] / 1000:.2f}",
                        "speed": speed,
                        "segment_words": segment_words,
                        "role": segment.get("spk")
                    }
                    if request.showEmotion:
                        item["emotion"] = emotion if emotion is not None else "平淡"
                    segments.append(item)

                if request.showRoleIdentify:
                    # 身份识别（老师/学生）
                    spk_features = extract_features(segments)
                    teacher_role, scores, student_roles = identify_teacher(spk_features)
                    if teacher_role is None:
                        teacher_role = max(spk_features, key=lambda x: spk_features[x]["keyword_count"])
                    segments = convert_role_ids(segments, teacher_role, student_roles)
                else:
                    segments = _convert_segments_to_spk_roles(segments)

                ret = {
                    "language": request.language,
                    "segments": segments,
                    "speed_info": calculate_speed_info(segments, audio_tensor.shape[-1] / sample_rate),
                    "text": text,
                    "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
                    "gpu_time_ms": f"{gpu_time_ms:.2f}",
                }
                update_stat("offline")

                # 清理
                for p in tmp_paths:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass
                return ret

        # ---------- 不开音轨分离 ----------
        try:
            local_param_dict["spk_model"] = None
            async with acquire_gpu_slot(task_id=task_id):
                rec_results = await generate_with_gpu_lock(
                    model_asr, input=audio_bytes, is_final=True, **local_param_dict
                )
        except (asyncio.TimeoutError, IndexError, HTTPException):
            update_fail_task()
            raise HTTPException(status_code=400, detail="请求过多或超时，请稍后再试")
            # return {"msg": "请求过多或超时，请稍后再试", "code": 4004}

        gpu_time_ms = (time.perf_counter() - start_time) * 1000
        if len(rec_results) == 0:
            update_fail_task()
            return {"text": "", "sentences": [], "code": 0}
        elif len(rec_results) == 1:
            rec_result = rec_results[0]
            text = rec_result["text"]
            if "sentence_info" not in rec_result or not rec_result["sentence_info"]:
                logger.error(f"音频文件为空或未检测到任何人声, 可能是静音文件, 文件名:{filename} 转写失败")
                raise HTTPException(status_code=400, detail="音频文件为空或未检测到任何人声,可能是静音文件")
                # return {"msg": "音频文件为空或未检测到任何人声,可能是静音文件", "code": 4008}

            segments = []
            for segment in rec_result["sentence_info"]:
                segment_words = []
                if request.wordTimestamps:
                    words = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z']+|\b[a-zA-Z']+\b", segment["text"])
                    timestamps = segment["timestamp"]
                    for i, word_text in enumerate(words):
                        if i < len(timestamps):
                            segment_words.append({
                                "bg": f"{timestamps[i][0] / 1000:.2f}",
                                "ed": f"{timestamps[i][1] / 1000:.2f}",
                                "word_text": word_text
                            })
                speed = calculate_speech_rate(
                    segment["text"],
                    segment["start"] / 1000,
                    segment["end"] / 1000,
                    settings.speech_rate_factor,
                )
                segments.append({
                    "segment_text": segment["text"],
                    "bg": f"{segment['start'] / 1000:.2f}",
                    "ed": f"{segment['end'] / 1000:.2f}",
                    "speed": speed,
                    "segment_words": segment_words
                })
            ret = {
                "language": request.language,
                "segments": segments,
                "speed_info": calculate_speed_info(segments, audio_tensor.shape[-1] / sample_rate),
                "text": text,
                "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
                "gpu_time_ms": f"{gpu_time_ms:.2f}"
            }
            update_stat("offline")
            for p in tmp_paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            return ret
        else:
            update_fail_task()
            for p in tmp_paths:
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
            return {"msg": "未知错误", "code": 4005}
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
