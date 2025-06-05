import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io

class NoKeypointError(Exception):
    """无关键点异常"""
    pass

# 全局变量存储模型实例
_interpreter = None
# 默认MoveNet模型路径，可通过环境变量覆盖
MODEL_PATH = os.environ.get(
    "MOVENET_MODEL_PATH",
    "/home/ubuntu/model/movenet_lightning_float16.tflite",
)

def _get_interpreter():
    """获取或初始化TFLite解释器（单例模式）"""
    global _interpreter
    if _interpreter is None:
        _interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        _interpreter.allocate_tensors()
    return _interpreter

def infer_keypoints(image_bytes: bytes) -> dict:
    """
    使用MoveNet模型推理关键点
    
    Args:
        image_bytes: 图片的二进制数据
        
    Returns:
        dict: 17个关键点的坐标，格式如 {"left_shoulder": [x, y], ...}
        
    Raises:
        NoKeypointError: 当无法检测到关键点时
    """
    try:
        # 1. 加载并预处理图片
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_resized = img.resize((256, 256))

        # 2. 转换为模型输入格式 (MoveNet使用float32输入，范围0-1)
        input_data = np.array(img_resized, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        # 3. 获取模型并推理
        interpreter = _get_interpreter()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 4. 获取输出 (MoveNet输出可能包含额外维度)
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        print("Raw model output:", keypoints_with_scores)
        print("Raw output shape:", np.shape(keypoints_with_scores))

        if keypoints_with_scores.ndim == 4:
            keypoints = keypoints_with_scores[0, 0, :17, :3]
        elif keypoints_with_scores.ndim == 3:
            keypoints = keypoints_with_scores[0, :17, :3]
        else:
            keypoints = keypoints_with_scores.reshape(-1, keypoints_with_scores.shape[-1])[:17, :3]

        print("Processed keypoints:", keypoints)
        print("Processed keypoints length:", len(keypoints))

        if keypoints.shape[0] < 17:
            return {"code": "NO_KEYPOINT", "msg": "No valid keypoints detected", "raw_shape": list(keypoints_with_scores.shape)}
        
        # 5. 检查置信度（第3个值是置信度分数）
        confidence_threshold = 0.1
        valid_keypoints = 0
        for i in range(17):
            if keypoints[i, 2] > confidence_threshold:
                valid_keypoints += 1
        if valid_keypoints < 5:  # 至少需要部分有效关键点
            raise NoKeypointError("检测到的关键点数量不足")
        
        # 6. 转换为标准格式
        # MoveNet关键点顺序：
        keypoint_names = [
            "nose",
            "left_eye", "right_eye",
            "left_ear", "right_ear",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]
        
        result = {}
        for i, name in enumerate(keypoint_names):
            # y, x 是归一化坐标 (0-1之间)
            y, x, score = keypoints[i]
            # 转换为像素坐标(保持归一化坐标，便于后续处理)
            result[name] = [float(x), float(y)]
        
        return result
        
    except NoKeypointError:
        raise
    except Exception as e:
        raise NoKeypointError(f"推理失败: {str(e)}")