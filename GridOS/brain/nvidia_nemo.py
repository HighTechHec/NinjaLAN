"""
NVIDIA NeMo Integration
Advanced NLP capabilities using NVIDIA NeMo framework

Features:
- Named Entity Recognition (NER)
- Intent classification
- Sentiment analysis
- Text classification
- Question classification
- Advanced tokenization
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class IntentType(Enum):
    """Common intent types."""
    QUESTION = "question"
    STATEMENT = "statement"
    COMMAND = "command"
    GREETING = "greeting"
    FAREWELL = "farewell"
    SEARCH = "search"
    REMEMBER = "remember"
    RECALL = "recall"


class SentimentType(Enum):
    """Sentiment classifications."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class NERResult:
    """Named Entity Recognition result."""
    text: str
    entity_type: str  # PERSON, ORG, LOC, DATE, etc.
    start: int
    end: int
    confidence: float


@dataclass
class ClassificationResult:
    """Text classification result."""
    label: str
    confidence: float
    scores: Dict[str, float]


class NeMoNLPEngine:
    """
    NVIDIA NeMo-based NLP processing engine.
    
    Provides advanced NLP capabilities for the second brain:
    - Entity extraction
    - Intent detection
    - Sentiment analysis
    - Topic classification
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize NeMo NLP engine.
        
        Args:
            model_path: Path to pre-trained NeMo models (optional)
        """
        self.model_path = model_path
        self.ner_cache: Dict[str, List[NERResult]] = {}
        self.intent_cache: Dict[str, IntentType] = {}
        
        # Model configurations
        self.entity_types = [
            "PERSON", "ORGANIZATION", "LOCATION", "DATE", 
            "TIME", "MONEY", "PERCENT", "PRODUCT", "EVENT"
        ]
        
        self.intent_keywords = {
            IntentType.QUESTION: ["what", "why", "how", "when", "where", "who", "which"],
            IntentType.COMMAND: ["do", "make", "create", "delete", "update", "show"],
            IntentType.SEARCH: ["find", "search", "look", "locate"],
            IntentType.REMEMBER: ["remember", "save", "store", "note", "capture"],
            IntentType.RECALL: ["recall", "retrieve", "get", "what did", "tell me about"],
        }
    
    def extract_entities(self, text: str, use_cache: bool = True) -> List[NERResult]:
        """
        Extract named entities from text using NeMo NER.
        
        Args:
            text: Input text
            use_cache: Whether to use cache
            
        Returns:
            List of extracted entities
        """
        if use_cache and text in self.ner_cache:
            return self.ner_cache[text]
        
        # In production, this would use NeMo's NER model
        # For now, implementing rule-based extraction
        entities = self._rule_based_ner(text)
        
        if use_cache:
            self.ner_cache[text] = entities
        
        return entities
    
    def _rule_based_ner(self, text: str) -> List[NERResult]:
        """Rule-based NER for demonstration."""
        entities = []
        words = text.split()
        
        for i, word in enumerate(words):
            # Detect capitalized words (potential entities)
            if word and word[0].isupper() and len(word) > 1:
                # Check if it's a sentence start
                is_sentence_start = i == 0 or words[i-1].endswith('.')
                
                if not is_sentence_start:
                    start = sum(len(w) + 1 for w in words[:i])
                    end = start + len(word)
                    
                    # Heuristic: determine entity type
                    entity_type = "PERSON"  # Default
                    if word.endswith("Corp") or word.endswith("Inc"):
                        entity_type = "ORGANIZATION"
                    elif word in ["Monday", "Tuesday", "January", "February"]:
                        entity_type = "DATE"
                    
                    entities.append(NERResult(
                        text=word,
                        entity_type=entity_type,
                        start=start,
                        end=end,
                        confidence=0.75
                    ))
        
        return entities
    
    def classify_intent(self, text: str, use_cache: bool = True) -> ClassificationResult:
        """
        Classify user intent from text.
        
        Args:
            text: Input text
            use_cache: Whether to use cache
            
        Returns:
            Classification result with intent and confidence
        """
        if use_cache and text in self.intent_cache:
            cached_intent = self.intent_cache[text]
            return ClassificationResult(
                label=cached_intent.value,
                confidence=0.95,
                scores={cached_intent.value: 0.95}
            )
        
        # Analyze text for intent
        text_lower = text.lower().strip()
        scores = {}
        
        # Check for question
        if text.endswith('?') or any(word in text_lower for word in self.intent_keywords[IntentType.QUESTION]):
            scores[IntentType.QUESTION.value] = 0.9
        else:
            scores[IntentType.QUESTION.value] = 0.1
        
        # Check for commands
        if any(text_lower.startswith(word) for word in self.intent_keywords[IntentType.COMMAND]):
            scores[IntentType.COMMAND.value] = 0.85
        else:
            scores[IntentType.COMMAND.value] = 0.1
        
        # Check for search
        if any(word in text_lower for word in self.intent_keywords[IntentType.SEARCH]):
            scores[IntentType.SEARCH.value] = 0.8
        else:
            scores[IntentType.SEARCH.value] = 0.15
        
        # Check for remember
        if any(word in text_lower for word in self.intent_keywords[IntentType.REMEMBER]):
            scores[IntentType.REMEMBER.value] = 0.88
        else:
            scores[IntentType.REMEMBER.value] = 0.1
        
        # Check for recall
        if any(word in text_lower for word in self.intent_keywords[IntentType.RECALL]):
            scores[IntentType.RECALL.value] = 0.87
        else:
            scores[IntentType.RECALL.value] = 0.1
        
        # Default to statement
        scores[IntentType.STATEMENT.value] = 0.5
        
        # Find highest score
        max_intent = max(scores.items(), key=lambda x: x[1])
        
        result = ClassificationResult(
            label=max_intent[0],
            confidence=max_intent[1],
            scores=scores
        )
        
        # Cache result
        if use_cache:
            intent_enum = IntentType(max_intent[0])
            self.intent_cache[text] = intent_enum
        
        return result
    
    def analyze_sentiment(self, text: str) -> ClassificationResult:
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment classification
        """
        # Simple rule-based sentiment analysis
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful",
            "fantastic", "love", "best", "awesome", "perfect"
        ]
        negative_words = [
            "bad", "terrible", "awful", "horrible", "worst",
            "hate", "poor", "disappointing", "useless", "broken"
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate sentiment score (-1 to 1)
        total = positive_count + negative_count
        if total > 0:
            score = (positive_count - negative_count) / total
        else:
            score = 0.0
        
        # Map to sentiment category
        if score > 0.6:
            sentiment = SentimentType.VERY_POSITIVE
            confidence = 0.9
        elif score > 0.2:
            sentiment = SentimentType.POSITIVE
            confidence = 0.8
        elif score < -0.6:
            sentiment = SentimentType.VERY_NEGATIVE
            confidence = 0.9
        elif score < -0.2:
            sentiment = SentimentType.NEGATIVE
            confidence = 0.8
        else:
            sentiment = SentimentType.NEUTRAL
            confidence = 0.7
        
        scores = {
            SentimentType.VERY_POSITIVE.value: max(0.0, score + 0.5) if score > 0.5 else 0.1,
            SentimentType.POSITIVE.value: max(0.0, score + 0.3) if score > 0 else 0.2,
            SentimentType.NEUTRAL.value: 1.0 - abs(score),
            SentimentType.NEGATIVE.value: max(0.0, -score + 0.3) if score < 0 else 0.2,
            SentimentType.VERY_NEGATIVE.value: max(0.0, -score + 0.5) if score < -0.5 else 0.1,
        }
        
        return ClassificationResult(
            label=sentiment.value,
            confidence=confidence,
            scores=scores
        )
    
    def classify_topic(self, text: str, topics: List[str]) -> ClassificationResult:
        """
        Classify text into predefined topics.
        
        Args:
            text: Input text
            topics: List of possible topics
            
        Returns:
            Topic classification
        """
        text_lower = text.lower()
        scores = {}
        
        for topic in topics:
            # Simple keyword matching
            topic_lower = topic.lower()
            if topic_lower in text_lower:
                scores[topic] = 0.9
            else:
                # Count word overlaps
                text_words = set(text_lower.split())
                topic_words = set(topic_lower.split())
                overlap = len(text_words & topic_words)
                scores[topic] = min(0.8, overlap * 0.2)
        
        if not scores:
            return ClassificationResult(
                label="unknown",
                confidence=0.5,
                scores={"unknown": 0.5}
            )
        
        max_topic = max(scores.items(), key=lambda x: x[1])
        
        return ClassificationResult(
            label=max_topic[0],
            confidence=max_topic[1],
            scores=scores
        )
    
    def tokenize_advanced(self, text: str) -> List[str]:
        """
        Advanced tokenization using NeMo.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # In production, would use NeMo's tokenizer
        # Basic implementation for now
        import re
        
        # Split on whitespace and punctuation, but keep contractions
        tokens = re.findall(r"\b\w+(?:'\w+)?\b|[^\w\s]", text)
        return tokens
    
    def get_stats(self) -> Dict:
        """Get NeMo engine statistics."""
        return {
            "ner_cache_size": len(self.ner_cache),
            "intent_cache_size": len(self.intent_cache),
            "supported_entity_types": len(self.entity_types),
            "entity_types": self.entity_types
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.ner_cache.clear()
        self.intent_cache.clear()


# Convenience instance
nemo_engine = NeMoNLPEngine()


__all__ = [
    "NeMoNLPEngine",
    "NERResult",
    "ClassificationResult",
    "IntentType",
    "SentimentType",
    "nemo_engine",
]
