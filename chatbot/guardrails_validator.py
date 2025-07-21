"""
Guardrails Validator Module for Patent Chatbot

This module implements content validation using custom validators to ensure:
1. Profanity-free responses (score: 0.0 = clean, 1.0 = profanity detected)
2. Topic restriction to patent-related content (score: 0.0-1.0 based on relevance)
3. Politeness checks (score: 0.0 = impolite, 1.0 = polite)
4. Scoring metrics for evaluation
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class GuardrailScores:
    """Scores for each guardrail metric (1.0 = guardrail triggered, 0.0 = not triggered)"""
    profanity_score: float = 0.0  # 1.0 = profanity detected, 0.0 = clean
    topic_relevance_score: float = 0.0  # 1.0 = off-topic, 0.0 = on-topic
    politeness_score: float = 0.0  # 1.0 = impolite, 0.0 = polite
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'profanity_score': self.profanity_score,
            'topic_relevance_score': self.topic_relevance_score,
            'politeness_score': self.politeness_score
        }
    
    def get_overall_score(self) -> float:
        """Calculate overall guardrail score (average of all scores)"""
        scores = [self.profanity_score, self.topic_relevance_score, self.politeness_score]
        return sum(scores) / len(scores) if scores else 0.0
    
    def is_acceptable(self) -> bool:
        """Check if the response passes all guardrails"""
        return (self.profanity_score < 0.5 and 
                self.topic_relevance_score < 0.5 and 
                self.politeness_score < 0.5)
    
    def get_rejection_reason(self) -> str:
        """Get the reason for rejection if response is not acceptable"""
        reasons = []
        
        if self.profanity_score >= 0.5:
            reasons.append("Content contains inappropriate language")
        
        if self.topic_relevance_score >= 0.5:
            reasons.append("Content is not relevant to patent topics")
        
        if self.politeness_score >= 0.5:
            reasons.append("Content is not professional or polite")
        
        return "; ".join(reasons) if reasons else "Response passed all guardrails"

class CustomGuardrailsValidator:
    """
    Custom validators for content validation without external dependencies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_validators()
        
    def setup_validators(self):
        """Initialize the custom validators"""
        # Profanity patterns (common profane words)
        self.profanity_patterns = [
            r'\b(fuck|shit|damn|hell|bitch|ass|dick|pussy|cunt|cock|whore|slut)\b',
            r'\b(fucking|shitting|damned|hellish|bitchy|asshole|dickhead|pussycat|cuntface|cocky|whorehouse|slutty)\b',
            r'\b(f\*ck|s\*it|d\*mn|h\*ll|b\*tch|a\*s|d\*ck|p\*ssy|c\*nt|c\*ck|wh\*re|sl\*t)\b'
        ]
        
        # Patent-related keywords for topic validation
        self.patent_keywords = [
            'patent', 'invention', 'claim', 'prior art', 'uspto', 'intellectual property',
            'technology', 'innovation', 'device', 'method', 'system', 'apparatus',
            'composition', 'process', 'manufacture', 'machine', 'design',
            'utility', 'provisional', 'non-provisional', 'patent application',
            'patent office', 'examination', 'prosecution', 'infringement',
            'validity', 'novelty', 'obviousness', 'enablement', 'written description',
            'patentability', 'patentee', 'inventor', 'assignee', 'patent attorney',
            'patent agent', 'patent law', 'patent litigation', 'patent portfolio',
            'patent licensing', 'patent valuation', 'patent search', 'patent classification',
            'patent filing', 'patent prosecution', 'patent litigation', 'patent portfolio',
            'patent licensing', 'patent valuation', 'patent strategy', 'patent analysis',
            'technical', 'scientific', 'research', 'development', 'engineering',
            'computer', 'software', 'hardware', 'algorithm', 'data', 'information',
            'electronic', 'digital', 'mechanical', 'chemical', 'biological',
            'medical', 'pharmaceutical', 'biotechnology', 'nanotechnology'
        ]
        
        # Impolite patterns
        self.impolite_patterns = [
            r'\b(terrible|awful|horrible|stupid|idiot|dumb|fool|moron|imbecile)\b',
            r'\b(useless|worthless|garbage|trash|rubbish|nonsense|ridiculous|absurd)\b',
            r'\b(hate|loathe|despise|abhor|detest|disgusting|revolting|appalling)\b',
            r'\b(you are wrong|you are stupid|you are an idiot|you don\'t know anything)\b',
            r'\b(this is terrible|this is awful|this is horrible|this is stupid)\b',
            r'\b(I don\'t care|I don\'t give a damn|who cares|whatever)\b'
        ]
        
        self.logger.info("Custom guardrails validators initialized successfully")
    
    def check_profanity(self, text: str) -> Tuple[bool, float]:
        """
        Check for profanity in text
        
        Returns:
            Tuple of (is_clean, score) where score is 0.0 if clean, 1.0 if profane
        """
        text_lower = text.lower()
        
        for pattern in self.profanity_patterns:
            if re.search(pattern, text_lower):
                return False, 1.0
        
        return True, 0.0
    
    def check_topic_relevance(self, text: str) -> Tuple[bool, float]:
        """
        Check if text is relevant to patent topics
        
        Returns:
            Tuple of (is_relevant, score) where score is 0.0 if relevant, 1.0 if off-topic
        """
        text_lower = text.lower()
        word_count = len(text.split())
        
        if word_count == 0:
            return False, 1.0  # Empty text is considered off-topic
        
        keyword_count = 0
        for keyword in self.patent_keywords:
            if keyword.lower() in text_lower:
                keyword_count += 1
        
        # Calculate relevance score based on keyword density
        relevance_score = min(1.0, keyword_count / max(1, word_count / 10))
        
        # Consider relevant if score > 0.1 (at least some patent-related content)
        is_relevant = relevance_score > 0.1
        
        # Return off-topic score: 1.0 if off-topic, 0.0 if relevant
        off_topic_score = 1.0 if not is_relevant else 0.0
        
        return is_relevant, off_topic_score
    
    def check_politeness(self, text: str) -> Tuple[bool, float]:
        """
        Check if text is polite and professional
        
        Returns:
            Tuple of (is_polite, score) where score is 0.0 if polite, 1.0 if impolite
        """
        text_lower = text.lower()
        
        # Check for impolite patterns
        for pattern in self.impolite_patterns:
            if re.search(pattern, text_lower):
                return False, 1.0
        
        # Check for professional indicators
        professional_indicators = [
            'please', 'thank you', 'appreciate', 'respectfully', 'professionally',
            'carefully', 'thoroughly', 'accurately', 'precisely', 'clearly',
            'helpful', 'useful', 'beneficial', 'valuable', 'important'
        ]
        
        professional_count = sum(1 for indicator in professional_indicators 
                              if indicator in text_lower)
        
        # If professional language is present, consider polite
        if professional_count > 0:
            return True, 0.0
        
        # Default to polite if no impolite or professional indicators
        return True, 0.0
    
    def validate_response(self, response: str) -> Tuple[str, GuardrailScores]:
        """
        Validate a chatbot response using all guardrails
        
        Args:
            response: The chatbot response to validate
            
        Returns:
            Tuple of (validated_response, scores)
        """
        try:
            scores = GuardrailScores()
            
            # Check profanity
            is_clean, profanity_score = self.check_profanity(response)
            scores.profanity_score = profanity_score
            
            # Check topic relevance
            is_relevant, topic_score = self.check_topic_relevance(response)
            scores.topic_relevance_score = topic_score
            
            # Check politeness
            is_polite, politeness_score = self.check_politeness(response)
            scores.politeness_score = politeness_score
            
            # If any validation fails, try to improve the response
            validated_response = response
            if not is_clean or not is_relevant or not is_polite:
                validated_response = self._improve_response(response, is_clean, is_relevant, is_polite)
            
            return validated_response, scores
            
        except Exception as e:
            self.logger.error(f"Error during response validation: {e}")
            # Return original response with zero scores on error
            return response, GuardrailScores()
    
    def _improve_response(self, response: str, is_clean: bool, is_relevant: bool, is_polite: bool) -> str:
        """
        Attempt to improve the response based on validation results
        """
        improved = response
        
        # Add professional prefix if not polite
        if not is_polite:
            improved = f"Professionally speaking, {improved}"
        
        # Add patent context if not relevant
        if not is_relevant:
            improved = f"In the context of patent analysis, {improved}"
        
        return improved
    
    def validate_batch(self, responses: List[str]) -> List[Tuple[str, GuardrailScores]]:
        """
        Validate a batch of responses
        
        Args:
            responses: List of responses to validate
            
        Returns:
            List of tuples (validated_response, scores)
        """
        results = []
        for response in responses:
            validated_response, scores = self.validate_response(response)
            results.append((validated_response, scores))
        return results
    
    def get_validation_summary(self, responses: List[str]) -> Dict:
        """
        Get a summary of validation results for multiple responses
        
        Args:
            responses: List of responses to analyze
            
        Returns:
            Dictionary with validation statistics
        """
        if not responses:
            return {
                "total_responses": 0,
                "average_scores": GuardrailScores().to_dict(),
                "overall_score": 0.0
            }
        
        validated_results = self.validate_batch(responses)
        
        # Calculate average scores
        total_profanity = sum(result[1].profanity_score for result in validated_results)
        total_topic = sum(result[1].topic_relevance_score for result in validated_results)
        total_politeness = sum(result[1].politeness_score for result in validated_results)
        
        num_responses = len(validated_results)
        
        avg_scores = GuardrailScores(
            profanity_score=total_profanity / num_responses,
            topic_relevance_score=total_topic / num_responses,
            politeness_score=total_politeness / num_responses
        )
        
        return {
            "total_responses": num_responses,
            "average_scores": avg_scores.to_dict(),
            "overall_score": avg_scores.get_overall_score(),
            "individual_results": [
                {
                    "original": responses[i],
                    "validated": result[0],
                    "scores": result[1].to_dict()
                }
                for i, result in enumerate(validated_results)
            ]
        }

# Alias for backward compatibility
GuardrailsValidator = CustomGuardrailsValidator

# Example usage and testing
def test_guardrails():
    """Test the guardrails validator with sample responses"""
    validator = CustomGuardrailsValidator()
    
    test_responses = [
        "This patent describes a novel method for data encryption using quantum computing principles.",
        "This is a terrible patent that should be rejected immediately!",
        "The invention relates to a new type of semiconductor device.",
        "I don't care about your stupid patent questions."
    ]
    
    print("Testing Custom Guardrails Validator...")
    print("=" * 50)
    
    for i, response in enumerate(test_responses, 1):
        print(f"\nTest {i}:")
        print(f"Original: {response}")
        
        validated_response, scores = validator.validate_response(response)
        print(f"Validated: {validated_response}")
        print(f"Scores: {scores.to_dict()}")
        print(f"Overall Score: {scores.get_overall_score():.2f}")
    
    # Get summary
    summary = validator.get_validation_summary(test_responses)
    print(f"\nSummary:")
    print(f"Total responses: {summary['total_responses']}")
    print(f"Average scores: {summary['average_scores']}")
    print(f"Overall score: {summary['overall_score']:.2f}")

if __name__ == "__main__":
    test_guardrails() 