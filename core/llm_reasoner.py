from typing import List, Optional
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from core.models import VideoTree
from config import settings
import logging

logger = logging.getLogger(__name__)


class LLMReasoner:
    def __init__(
            self,
            model_name: str = settings.LLM_MODEL_NAME,
            temperature: float = settings.LLM_TEMPERATURE,
            api_key: Optional[str] = None
    ):
        api_key = api_key or settings.OPENAI_API_KEY
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )

    def _generate_frame_descriptions(self, frames: List[VideoFrame]) -> List[str]:
        """Generate descriptions for frames (simplified - in practice use vision model)"""
        return [f"Frame at {frame.timestamp:.2f} seconds" for frame in frames]

    def _generate_tree_description(self, tree: VideoTree) -> str:
        """Generate text description of tree structure"""
        description = []
        root_nodes = [node for node in tree.nodes.values() if node.parent is None]

        for node in root_nodes:
            description.append(self._describe_node(tree, node))

        return "\n\n".join(description)

    def _describe_node(self, tree: VideoTree, node: VideoNode, indent: int = 0) -> str:
        """Recursively describe a node and its children"""
        prefix = "  " * indent
        desc = f"{prefix}Node {node.id} (Level {node.level}, {len(node.frame_indices)} frames)"

        if node.representative_frame_idx is not None:
            frame = tree.frames[node.representative_frame_idx]
            desc += f"\n{prefix}Representative frame at {frame.timestamp:.2f}s"

        for child_id in node.children:
            child = tree.nodes[child_id]
            desc += "\n" + self._describe_node(tree, child, indent + 1)

        return desc

    def summarize_video(self, tree: VideoTree) -> str:
        """Generate a summary of the video content"""
        prompt_template = """
        You are a video understanding assistant. Analyze the following hierarchical structure of a video and generate a comprehensive summary.

        Video Structure:
        {tree_structure}

        Key Frames:
        {frame_descriptions}

        Please provide a detailed summary that:
        1. Identifies the main scenes and events
        2. Describes the temporal progression
        3. Highlights important transitions
        4. Provides overall context and interpretation
        """

        frame_descriptions = self._generate_frame_descriptions(tree.frames)
        tree_structure = self._generate_tree_description(tree)

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["tree_structure", "frame_descriptions"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(
            tree_structure=tree_structure,
            frame_descriptions="\n".join(frame_descriptions)
        )

    def answer_question(self, tree: VideoTree, question: str) -> str:
        """Answer questions about the video content"""
        prompt_template = """
        You are a video question answering assistant. Use the following video structure and frame information to answer the question.

        Video Structure:
        {tree_structure}

        Key Frames:
        {frame_descriptions}

        Question: {question}

        Guidelines:
        1. Be precise and factual
        2. Reference specific parts of the video when possible
        3. If unsure, say "I cannot determine from the video"
        4. Keep answers concise but informative
        """

        frame_descriptions = self._generate_frame_descriptions(tree.frames)
        tree_structure = self._generate_tree_description(tree)

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["tree_structure", "frame_descriptions", "question"]
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(
            tree_structure=tree_structure,
            frame_descriptions="\n".join(frame_descriptions),
            question=question
        )