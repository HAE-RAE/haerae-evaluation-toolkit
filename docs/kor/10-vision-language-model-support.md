# 비전-언어 모델 지원 (VQA)

Haerae Evaluation Toolkit은 이제 모든 모델 백엔드에서 비전-언어 벤치마크(VQA)를 지원하여, 텍스트와 이미지를 모두 사용하는 멀티모달 평가 작업을 수행할 수 있습니다.

## 개요

비전-언어 모델은 텍스트와 이미지를 동시에 처리할 수 있어 다음과 같은 작업에 적합합니다:
- 시각적 질문 답변 (VQA)
- 이미지 캡셔닝
- 시각적 추론
- 멀티모달 이해

모든 백엔드(OpenAI, HuggingFace, LiteLLM, vLLM)에서 `is_vision_model=True` 매개변수를 통해 비전 입력을 지원합니다.

## 지원되는 입력 형식

### 형식 1: 직접 이미지 필드
```python
{
    "input": "이 이미지에서 무엇을 보시나요?",
    "images": ["https://example.com/image.jpg"],
    "reference": "고양이"
}
```

### 형식 2: 다중 이미지
```python
{
    "input": "이 두 이미지를 비교해주세요",
    "images": [
        "https://example.com/image1.jpg",
        "/local/path/image2.png"
    ],
    "reference": "첫 번째는 고양이, 두 번째는 개를 보여줍니다"
}
```

### 형식 3: 구조화된 입력
```python
{
    "input": {
        "text": "이 의료 스캔을 분석해주세요",
        "images": ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD"]
    },
    "reference": "정상 흉부 X-ray"
}
```

### 형식 4: 이미지가 포함된 객관식
```python
{
    "input": "이미지에 나타난 동물은 무엇인가요?",
    "images": ["https://example.com/animal.jpg"],
    "options": ["고양이", "개", "새", "물고기"],
    "reference": "고양이"
}
```

## 지원되는 이미지 형식

- **URL**: `https://example.com/image.jpg`
- **파일 경로**: `/path/to/local/image.png`
- **Base64 데이터 URL**: `data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD`
- **PIL Image 객체** (HuggingFace 및 vLLM 백엔드용)

## 백엔드 사용 예시

### 1. OpenAI 비전 모델 (GPT-4V)

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()

results = evaluator.run(
    model="openai",
    model_params={
        "model_name": "gpt-4-vision-preview",
        "api_key": "your-openai-api-key",
        "is_vision_model": True,
        "max_tokens": 512,
        "temperature": 0.0
    },
    dataset="your_vqa_dataset",
    split="test",
    evaluation_method="string_match"
)
```

### 2. HuggingFace 비전-언어 모델

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()

results = evaluator.run(
    model="huggingface",
    model_params={
        "model_name_or_path": "microsoft/kosmos-2-patch14-224",
        "is_vision_model": True,
        "max_new_tokens": 256,
        "temperature": 0.0
    },
    dataset="your_vqa_dataset",
    split="test",
    evaluation_method="string_match"
)
```

### 3. LiteLLM 비전 모델

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()

results = evaluator.run(
    model="litellm",
    model_params={
        "provider": "openai",
        "model_name": "gpt-4-vision-preview",
        "api_key": "your-api-key",
        "is_vision_model": True,
        "max_tokens": 512,
        "temperature": 0.0
    },
    dataset="your_vqa_dataset",
    split="test",
    evaluation_method="string_match"
)
```

### 4. vLLM 비전-언어 모델

```python
from llm_eval.evaluator import Evaluator

evaluator = Evaluator()

results = evaluator.run(
    model="vllm",
    model_params={
        "model_name_or_path": "llava-hf/llava-1.5-7b-hf",
        "is_vision_model": True,
        "max_tokens": 256,
        "temperature": 0.0
    },
    dataset="your_vqa_dataset",
    split="test",
    evaluation_method="string_match"
)
```

## 직접 모델 사용

Evaluator 없이 비전 모델을 직접 사용할 수도 있습니다:

### OpenAI 백엔드
```python
from llm_eval.models import OpenAIModel

model = OpenAIModel(
    api_key="your-api-key",
    model_name="gpt-4-vision-preview",
    is_vision_model=True
)

inputs = [{
    "input": "이 이미지에 무엇이 있나요?",
    "images": ["https://example.com/image.jpg"]
}]

results = model.generate_batch(inputs)
print(results[0]["prediction"])
```

### HuggingFace 백엔드
```python
from llm_eval.models import HuggingFaceModel

model = HuggingFaceModel(
    model_name_or_path="microsoft/kosmos-2-patch14-224",
    is_vision_model=True
)

inputs = [{
    "input": {
        "text": "이 이미지를 설명해주세요",
        "images": ["/path/to/image.jpg"]
    }
}]

results = model.generate_batch(inputs)
print(results[0]["prediction"])
```

## 인기 있는 비전-언어 모델

### OpenAI 모델
- `gpt-4-vision-preview`
- `gpt-4-turbo-vision`
- `gpt-4o` (비전 기능 포함)

### HuggingFace 모델
- `microsoft/kosmos-2-patch14-224`
- `llava-hf/llava-1.5-7b-hf`
- `llava-hf/llava-1.5-13b-hf`
- `Salesforce/blip2-opt-2.7b`
- `Salesforce/instructblip-vicuna-7b`

### vLLM 지원 모델
- `llava-hf/llava-1.5-7b-hf`
- `llava-hf/llava-1.5-13b-hf`
- 기타 LLaVA 변형 모델

## 의존성

비전 모델을 사용할 때 추가 의존성이 필요할 수 있습니다:

```bash
pip install pillow requests  # 이미지 처리용
```

이러한 의존성은 비전 모델 사용 시 자동으로 설치됩니다.

## 하위 호환성

- 기존의 모든 텍스트 전용 기능은 변경되지 않습니다
- 비전 지원은 `is_vision_model=True` 매개변수를 통해 선택적으로 활성화됩니다
- 기존 데이터셋과 벤치마크는 수정 없이 계속 작동합니다
- 기존 API에 대한 호환성을 깨뜨리는 변경사항은 없습니다

## 모범 사례

1. **이미지 품질**: 더 나은 모델 성능을 위해 고품질 이미지를 사용하세요
2. **이미지 크기**: 일부 모델에는 크기 제한이 있으니 모델 문서를 확인하세요
3. **배치 처리**: 비전 모델은 더 느릴 수 있으니 더 작은 배치 크기를 고려하세요
4. **오류 처리**: 이미지가 제대로 로드되지 않을 수 있는 경우를 항상 처리하세요
5. **메모리 사용량**: 비전 모델은 더 많은 메모리가 필요하니 리소스 사용량을 모니터링하세요

## 문제 해결

### 일반적인 문제

1. **이미지 로딩 오류**
   - 이미지 URL이 접근 가능한지 확인하세요
   - 파일 경로가 올바른지 확인하세요
   - 이미지 형식이 지원되는지 확인하세요

2. **메모리 문제**
   - 비전 모델의 배치 크기를 줄이세요
   - 가능하면 더 작은 이미지를 사용하세요
   - GPU 메모리 사용량을 모니터링하세요

3. **모델 로딩 문제**
   - 모델이 비전 입력을 지원하는지 확인하세요
   - 추가 모델 파일이 필요한지 확인하세요
   - 모델 이름이 올바른지 확인하세요

### 오류 메시지

- `"No images found in input"`: 입력 형식에 이미지가 포함되어 있는지 확인하세요
- `"Vision model not enabled"`: `is_vision_model=True`로 설정하세요
- `"Image processing failed"`: 이미지 형식과 접근성을 확인하세요

## 예시 및 튜토리얼

더 많은 예시와 자세한 튜토리얼은 다음을 확인하세요:
- 저장소의 `examples/` 디렉토리
- 테스트 파일: `test_vqa_*.py`
- 각 백엔드의 모델별 문서

---

비전-언어 지원을 통해 Haerae Evaluation Toolkit은 멀티모달 벤치마크와 평가 작업을 처리할 수 있게 되어, 텍스트 전용 모델을 넘어서는 기능을 제공합니다.