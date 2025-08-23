import os
from typing import Optional, Dict, Any
from openai import OpenAI
import logging
from config import settings

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        """Khởi tạo LLM service với cấu hình từ biến môi trường"""
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL if hasattr(settings, 'OPENAI_BASE_URL') else None
        )
        self.model_name = getattr(settings, 'LLM_MODEL', 'gpt-3.5-turbo')
        self.default_params = {
            'temperature': 0.7,
            'max_tokens': 1000,
            'top_p': 1.0,
        }

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Tạo văn bản từ prompt

        Args:
            prompt: Nội dung prompt đầu vào
            **kwargs: Các tham số bổ sung cho LLM

        Returns:
            Nội dung văn bản được tạo
        """
        try:
            params = {**self.default_params, **kwargs}

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **params
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    async def generate_from_template(self, template: str, context: Dict[str, Any]) -> str:
        """
        Tạo văn bản từ template và context

        Args:
            template: Template với chỗ trống {variable}
            context: Dict chứa giá trị các biến

        Returns:
            Nội dung đã được thay thế
        """
        try:
            prompt = template.format(**context)
            return await self.generate_text(prompt)
        except KeyError as e:
            logger.error(f"Missing context variable: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Template generation failed: {str(e)}")
            raise

    async def analyze_video_content(self, video_description: str) -> Dict[str, Any]:
        """
        Phân tích nội dung video và trả về structured data

        Args:
            video_description: Mô tả nội dung video

        Returns:
            Dict chứa các thông tin phân tích
        """
        prompt = f"""
        Phân tích nội dung video sau và trả về kết quả dạng JSON:
        {video_description}

        Kết quả bao gồm:
        - main_objects (list)
        - main_actions (list)
        - overall_sentiment (string)
        - key_scenes (list)
        """

        response = await self.generate_text(prompt)
        try:
            return eval(response)  # Chuyển string response thành dict
        except:
            return {'error': 'Invalid response format'}