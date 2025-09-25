# Vision-Language Model Support (VQA)

The Haerae Evaluation Toolkit now supports Vision-Language Benchmarks (VQA) across all model backends, enabling multimodal evaluation tasks with both text and images.

## Overview

Vision-Language models can process both text and images simultaneously, making them suitable for tasks like:
- Visual Question Answering (VQA)
- Image Captioning
- Visual Reasoning
- Multimodal Understanding

All backends (OpenAI, HuggingFace, LiteLLM, vLLM) now support vision inputs with the `is_vision_model=True` parameter.

## Supported Input Formats

### Format 1: Direct Images Field
```python
{
    "input": "What do you see in this image?",
    "images": ["https://example.com/image.jpg"],
    "reference": "A cat"
}
```

### Format 2: Multiple Images
```python
{
    "input": "Compare these two images",
    "images": [
        "https://example.com/image1.jpg",
        "/local/path/image2.png"
    ],
    "reference": "The first shows a cat, the second shows a dog"
}
```

### Format 3: Structured Input
```python
{
    "input": {
        "text": "Analyze this medical scan",
        "images": ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD"]
    },
    "reference": "Normal chest X-ray"
}
```

### Format 4: Multiple Choice with Images
```python
{
    "input": "What animal is shown in the image?",
    "images": ["https://example.com/animal.jpg"],
    "options": ["Cat", "Dog", "Bird", "Fish"],
    "reference": "Cat"
}
```

## Supported Image Formats

- **URLs**: `https://example.com/image.jpg`
- **File paths**: `/path/to/local/image.png`
- **Base64 data URLs**: `data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD`
- **PIL Image objects** (for HuggingFace and vLLM backends)

## Backend Usage Examples

### 1. OpenAI Vision Models (GPT-4V)

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

### 2. HuggingFace Vision-Language Models

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

### 3. LiteLLM Vision Models

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

### 4. vLLM Vision-Language Models

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

## Direct Model Usage

You can also use vision models directly without the Evaluator:

### OpenAI Backend
```python
from llm_eval.models import OpenAIModel

model = OpenAIModel(
    api_key="your-api-key",
    model_name="gpt-4-vision-preview",
    is_vision_model=True
)

inputs = [{
    "input": "What's in this image?",
    "images": ["https://example.com/image.jpg"]
}]

results = model.generate_batch(inputs)
print(results[0]["prediction"])
```

### HuggingFace Backend
```python
from llm_eval.models import HuggingFaceModel

model = HuggingFaceModel(
    model_name_or_path="microsoft/kosmos-2-patch14-224",
    is_vision_model=True
)

inputs = [{
    "input": {
        "text": "Describe this image",
        "images": ["/path/to/image.jpg"]
    }
}]

results = model.generate_batch(inputs)
print(results[0]["prediction"])
```

## Popular Vision-Language Models

### OpenAI Models
- `gpt-4-vision-preview`
- `gpt-4-turbo-vision`
- `gpt-4o` (with vision capabilities)

### HuggingFace Models
- `microsoft/kosmos-2-patch14-224`
- `llava-hf/llava-1.5-7b-hf`
- `llava-hf/llava-1.5-13b-hf`
- `Salesforce/blip2-opt-2.7b`
- `Salesforce/instructblip-vicuna-7b`

### vLLM Supported Models
- `llava-hf/llava-1.5-7b-hf`
- `llava-hf/llava-1.5-13b-hf`
- Other LLaVA variants

## Dependencies

When using vision models, you may need additional dependencies:

```bash
pip install pillow requests  # For image processing
```

These are automatically installed when using vision models.

## Backward Compatibility

- All existing text-only functionality remains unchanged
- Vision support is opt-in via the `is_vision_model=True` parameter
- Existing datasets and benchmarks continue to work without modification
- No breaking changes to existing APIs

## Best Practices

1. **Image Quality**: Use high-quality images for better model performance
2. **Image Size**: Some models have size limitations; check model documentation
3. **Batch Processing**: Vision models may be slower; consider smaller batch sizes
4. **Error Handling**: Always handle cases where images might not load properly
5. **Memory Usage**: Vision models require more memory; monitor resource usage

## Troubleshooting

### Common Issues

1. **Image Loading Errors**
   - Ensure image URLs are accessible
   - Check file paths are correct
   - Verify image formats are supported

2. **Memory Issues**
   - Reduce batch size for vision models
   - Use smaller images if possible
   - Monitor GPU memory usage

3. **Model Loading Issues**
   - Ensure the model supports vision inputs
   - Check if additional model files are needed
   - Verify model name is correct

### Error Messages

- `"No images found in input"`: Check your input format includes images
- `"Vision model not enabled"`: Set `is_vision_model=True`
- `"Image processing failed"`: Check image format and accessibility

## Examples and Tutorials

For more examples and detailed tutorials, check:
- `examples/` directory in the repository
- Test files: `test_vqa_*.py`
- Model-specific documentation in each backend

---

Vision-Language support enables the Haerae Evaluation Toolkit to handle multimodal benchmarks and evaluation tasks, expanding its capabilities beyond text-only models.