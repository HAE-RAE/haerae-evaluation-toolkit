import logging
from typing import List, Dict, Any, Optional
import time
import litellm

from .base import BaseModel

logger = logging.getLogger(__name__)

class LiteLLMBackend(BaseModel):
    """
    LiteLLM을 활용한 다양한 LLM Provider들의 백엔드 구현
    
    Attributes:
        provider (str): LLM 제공자 (예: openai, anthropic, bedrock, azure 등)
        model_name (str): 사용할 모델명
        max_new_tokens (int): 생성할 최대 토큰 수
        temperature (float): 생성 텍스트의 무작위성 정도 (0.0 ~ 1.0)
    """
    
    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        LiteLLM 백엔드 초기화
        
        Args:
            provider: LLM 제공자명
            model_name: 사용할 모델명
            api_key: API 키 (OpenAI, Azure)
            api_base: API 기본 URL (OpenAI, Azure)
            aws_access_key_id: AWS 액세스 키 ID (Bedrock)
            aws_secret_access_key: AWS 시크릿 키 (Bedrock)
            anthropic_api_key: Anthropic API 키
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 텍스트 무작위성 정도
            **kwargs: 추가 설정값
        """
        super().__init__(
            backend_type=provider,
            model_name=model_name,
        )
        
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.extra_kwargs = kwargs

        # LiteLLM 인스턴스별 설정을 위한 completion_kwargs 구성
        self.completion_kwargs = {
            "api_key": api_key,
            "api_base": api_base,
        }
        
        if provider == "bedrock":
            self.completion_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key,
            })
        elif provider == "anthropic":
            self.completion_kwargs["api_key"] = anthropic_api_key

    def _prepare_completion_kwargs(self, prompt: str) -> Dict[str, Any]:
        """
        LiteLLM completion 함수에 전달할 인자 준비
        
        Args:
            prompt: 입력 프롬프트
            
        Returns:
            Dict[str, Any]: LiteLLM completion 함수에 전달할 인자 딕셔너리
        """
        # 기본 completion 설정
        completion_kwargs = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            **self.completion_kwargs,  # 인스턴스별 API 설정 추가
            **self.extra_kwargs
        }
        
        # 프로바이더별 모델명 포맷 설정
        if self.provider == "huggingface":
            completion_kwargs["model"] = f"{self.model_name}"
        elif self.provider == "azure":
            completion_kwargs.update({
                "model": self.model_name,
                "engine": self.model_name,
            })
        elif self.provider == "bedrock":
            completion_kwargs["model"] = f"bedrock/{self.model_name}"
        else:
            completion_kwargs["model"] = self.model_name
            
        logger.debug(f"Prepared completion kwargs: {completion_kwargs}")
        return completion_kwargs

    def _generate_with_retry(
        self, 
        completion_kwargs: Dict[str, Any], 
        max_attempts: int = 3,
        initial_wait: int = 4
    ) -> str:
        """
        재시도 로직이 포함된 텍스트 생성 함수
        
        Args:
            completion_kwargs: LiteLLM completion 함수에 전달할 인자
            max_attempts: 최대 시도 횟수
            initial_wait: 첫 재시도 전 대기 시간(초)
            
        Returns:
            str: 생성된 텍스트
            
        Raises:
            Exception: 모든 재시도 실패 시 발생
        """
        attempt = 0
        last_exception = None
        
        while attempt < max_attempts:
            try:
                response = litellm.completion(**completion_kwargs)
                return response.choices[0].message.content
            except Exception as e:
                attempt += 1
                last_exception = e
                
                if attempt < max_attempts:
                    # 지수 백오프: 4초, 8초, 16초...
                    wait_time = initial_wait * (2 ** (attempt - 1))
                    logger.warning(
                        f"Attempt {attempt} failed with error: {str(e)}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
        
        error_msg = f"All {max_attempts} attempts failed. Last error: {str(last_exception)}"
        logger.error(error_msg)
        raise last_exception or Exception(error_msg)

    def generate_batch(self, input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        배치 형태의 텍스트 생성 수행
        
        Args:
            input_list: 입력 데이터 리스트. 각 항목은 'input'과 선택적 'reference' 키를 포함
            
        Returns:
            List[Dict[str, Any]]: 생성 결과 리스트. 각 항목은 입력, 참조, 예측값을 포함
        """
        results = []
        
        for idx, item in enumerate(input_list):
            prompt = item["input"]
            reference = item.get("reference", "")

            logger.info(
                f"[LiteLLMBackend] Generating text for item {idx+1}/{len(input_list)}: {prompt[:50]}..."
            )

            try:
                completion_kwargs = self._prepare_completion_kwargs(prompt)
                prediction = self._generate_with_retry(completion_kwargs)
            except Exception as e:
                logger.error(f"Error generating completion after retries: {str(e)}")
                prediction = f"Error: {str(e)}"

            results.append({
                "input": prompt,
                "reference": reference,
                "prediction": prediction
            })

        return results
    def generate_until(
        self,
        prompt: str, 
        stop_sequences: List[str],
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """LiteLLM을 사용하여 특정 시퀀스가 나올 때까지 텍스트 생성
        
        Args:
            prompt (str): 입력 프롬프트 텍스트
            stop_sequences (List[str]): 생성을 중단할 시퀀스 목록
            max_tokens (Optional[int]): 생성할 최대 토큰 수. 
                None인 경우 기본값 사용
            **kwargs: 추가 생성 파라미터

        Returns:
            str: 생성된 텍스트
        """
        completion_kwargs = self._prepare_completion_kwargs(prompt)
        
        # Override max_tokens if specified
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens
            
        # Add stop sequences
        completion_kwargs["stop"] = stop_sequences
        
        try:
            response = self._generate_with_retry(completion_kwargs)
            return response
        except Exception as e:
            logger.error(f"Error in generate_until: {str(e)}")
            raise