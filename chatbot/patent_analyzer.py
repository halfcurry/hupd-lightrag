#!/usr/bin/env python3
"""
Patent Analysis Module

This module provides three types of patent analysis:
1. Existing patent analysis (from database)
2. New invention evaluation (with probability)
3. Patent search by technology/topic
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PatentInfo:
    """Container for patent information"""
    title: str
    abstract: Optional[str] = None
    description: Optional[str] = None
    tech_area: Optional[str] = None
    patent_id: Optional[str] = None

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    probability: Optional[float] = None
    key_factors: List[str] = None
    analysis: str = ""
    recommendations: List[str] = None
    search_results: List[Dict] = None

class PatentProbabilityCalculator:
    """Calculate acceptance probability for new inventions"""
    
    def __init__(self):
        # Technology area weights (more realistic acceptance chances)
        self.tech_area_weights = {
            "ai/ml": 0.75,      # High competition, but strong market
            "software": 0.65,    # Saturated market, moderate acceptance
            "hardware": 0.70,    # Good acceptance for novel hardware
            "iot": 0.60,         # Growing but competitive
            "biotech": 0.80,     # High value, good acceptance
            "blockchain": 0.55,  # Declining interest, lower acceptance
            "cybersecurity": 0.70, # Growing field, good acceptance
            "robotics": 0.75,    # High growth, good acceptance
            "nanotechnology": 0.85, # High value, excellent acceptance
            "renewable energy": 0.80, # High priority, excellent acceptance
            "quantum computing": 0.90, # Cutting edge, excellent acceptance
            "autonomous vehicles": 0.70, # Competitive but valuable
            "augmented reality": 0.65, # Growing but competitive
            "5g/6g": 0.75,      # High priority, good acceptance
            "edge computing": 0.70, # Growing field, good acceptance
        }
        
        # Default weight for unknown technology areas
        self.default_tech_weight = 0.60  # More conservative default
    
    def calculate_probability(self, patent_info: PatentInfo) -> Dict:
        """Calculate acceptance probability based on patent information"""
        
        # Analyze title quality
        title_score = self._analyze_title(patent_info.title)
        
        # Analyze abstract quality
        abstract_score = self._analyze_abstract(patent_info.abstract)
        
        # Analyze description (if available)
        description_score = self._analyze_description(patent_info.description)
        
        # Analyze technology area
        tech_area_score = self._analyze_tech_area(patent_info.tech_area)
        
        # Calculate overall probability
        scores = [title_score, abstract_score, tech_area_score]
        if description_score is not None:
            scores.append(description_score)
        
        overall_probability = self._calculate_overall_score(scores)
        
        # Generate factors and recommendations
        factors = self._generate_factors(patent_info, title_score, abstract_score, description_score, tech_area_score)
        recommendations = self._generate_recommendations(patent_info, overall_probability)
        
        return {
            "overall_probability": overall_probability,
            "factors": factors,
            "recommendations": recommendations,
            "analysis": self._generate_analysis(patent_info, overall_probability)
        }
    
    def _analyze_title(self, title: str) -> float:
        """Analyze title quality (0.0-1.0) with improved scoring"""
        if not title:
            return 0.0
        
        title_lower = title.lower()
        score = 0.3  # Lower base score
        
        # Length check (not too short, not too long)
        title_length = len(title)
        if 15 <= title_length <= 80:
            score += 0.2
        elif 10 <= title_length <= 100:
            score += 0.1
        else:
            score -= 0.1  # Penalty for poor length
        
        # Check for technical terms
        tech_terms = ["system", "method", "apparatus", "device", "process", "algorithm", "optimization", "prediction", "mechanism", "technique"]
        tech_term_count = sum(1 for term in tech_terms if term in title_lower)
        if tech_term_count >= 2:
            score += 0.25
        elif tech_term_count == 1:
            score += 0.15
        else:
            score -= 0.1  # Penalty for lack of technical terms
        
        # Check for clarity (no vague terms)
        vague_terms = ["improved", "better", "new", "novel", "enhanced", "advanced"]
        vague_count = sum(1 for term in vague_terms if term in title_lower)
        if vague_count == 0:
            score += 0.15
        elif vague_count == 1:
            score += 0.05
        else:
            score -= 0.1  # Penalty for multiple vague terms
        
        # Check for specificity
        specific_indicators = ["for", "using", "with", "via", "through", "based on"]
        if any(indicator in title_lower for indicator in specific_indicators):
            score += 0.1
        
        # Check for proper capitalization and formatting
        if title[0].isupper() and not title.isupper():
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _analyze_abstract(self, abstract: str) -> float:
        """Analyze abstract quality (0.0-1.0) with improved scoring"""
        if not abstract:
            return 0.0
        
        abstract_lower = abstract.lower()
        score = 0.3  # Lower base score
        
        # Length check (good abstracts are 50-200 words)
        word_count = len(abstract.split())
        if 80 <= word_count <= 150:
            score += 0.25
        elif 50 <= word_count <= 200:
            score += 0.15
        elif word_count < 30:
            score -= 0.2  # Penalty for too short
        elif word_count > 300:
            score -= 0.1  # Penalty for too long
        
        # Check for problem-solution structure
        problem_indicators = ["problem", "issue", "challenge", "difficulty", "limitation", "drawback", "deficiency"]
        solution_indicators = ["solves", "addresses", "overcomes", "improves", "optimizes", "provides", "enables", "achieves"]
        
        problem_count = sum(1 for indicator in problem_indicators if indicator in abstract_lower)
        solution_count = sum(1 for indicator in solution_indicators if indicator in abstract_lower)
        
        if problem_count >= 1 and solution_count >= 1:
            score += 0.25
        elif problem_count >= 1 or solution_count >= 1:
            score += 0.1
        else:
            score -= 0.15  # Penalty for missing problem-solution structure
        
        # Check for technical depth
        tech_terms = ["algorithm", "system", "method", "process", "technology", "implementation", "mechanism", "architecture", "framework"]
        tech_term_count = sum(1 for term in tech_terms if term in abstract_lower)
        if tech_term_count >= 3:
            score += 0.2
        elif tech_term_count >= 1:
            score += 0.1
        else:
            score -= 0.1  # Penalty for lack of technical terms
        
        # Check for novelty indicators
        novelty_indicators = ["novel", "innovative", "unique", "distinctive", "original", "first", "pioneering"]
        novelty_count = sum(1 for indicator in novelty_indicators if indicator in abstract_lower)
        if novelty_count >= 1:
            score += 0.1
        
        # Check for benefits/advantages
        benefit_indicators = ["benefit", "advantage", "efficiency", "performance", "accuracy", "reliability", "cost-effective"]
        benefit_count = sum(1 for indicator in benefit_indicators if indicator in abstract_lower)
        if benefit_count >= 1:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _analyze_description(self, description: Optional[str]) -> Optional[float]:
        """Analyze description quality (0.0-1.0) with improved scoring"""
        if not description:
            return None
        
        description_lower = description.lower()
        score = 0.4  # Lower base score for having a description
        
        # Length check (good descriptions are 200+ words)
        word_count = len(description.split())
        if word_count >= 500:
            score += 0.25
        elif word_count >= 200:
            score += 0.15
        elif word_count >= 100:
            score += 0.05
        else:
            score -= 0.2  # Penalty for too short
        
        # Check for technical details
        tech_indicators = ["comprises", "includes", "connected to", "configured to", "processor", "algorithm", "module", "component", "interface", "protocol"]
        tech_count = sum(1 for indicator in tech_indicators if indicator in description_lower)
        if tech_count >= 5:
            score += 0.25
        elif tech_count >= 2:
            score += 0.15
        elif tech_count >= 1:
            score += 0.05
        else:
            score -= 0.15  # Penalty for lack of technical details
        
        # Check for implementation details
        implementation_indicators = ["step", "procedure", "workflow", "sequence", "iteration", "loop", "condition", "parameter"]
        impl_count = sum(1 for indicator in implementation_indicators if indicator in description_lower)
        if impl_count >= 3:
            score += 0.2
        elif impl_count >= 1:
            score += 0.1
        
        # Check for diagrams/figures references
        figure_indicators = ["figure", "diagram", "illustration", "drawing", "chart", "graph"]
        if any(indicator in description_lower for indicator in figure_indicators):
            score += 0.1
        
        # Check for claims-like language
        claim_indicators = ["wherein", "thereby", "such that", "characterized by", "comprising", "consisting of"]
        if any(indicator in description_lower for indicator in claim_indicators):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _analyze_tech_area(self, tech_area: Optional[str]) -> float:
        """Analyze technology area potential (0.0-1.0) with improved scoring"""
        if not tech_area:
            return self.default_tech_weight
        
        tech_area_lower = tech_area.lower()
        
        # Get weight for this technology area
        for area, weight in self.tech_area_weights.items():
            if area in tech_area_lower:
                return weight
        
        # Check for multiple tech areas (bonus for interdisciplinary)
        tech_areas = ["ai", "ml", "software", "hardware", "iot", "biotech", "blockchain", "cybersecurity", "robotics", "nanotechnology", "renewable energy"]
        found_areas = [area for area in tech_areas if area in tech_area_lower]
        
        if len(found_areas) >= 2:
            return min(1.0, self.default_tech_weight + 0.1)  # Bonus for interdisciplinary
        elif len(found_areas) == 1:
            return self.default_tech_weight
        else:
            return max(0.5, self.default_tech_weight - 0.1)  # Penalty for unclear tech area
    
    def _calculate_overall_score(self, scores: List[float]) -> float:
        """Calculate overall probability from individual scores with improved weighting"""
        if not scores:
            return 0.0
        
        # Use fixed weights but with quality-based adjustments
        if len(scores) == 4:  # Has description
            # Fixed weights: title, abstract, tech_area, description
            base_weights = [0.25, 0.30, 0.25, 0.20]
        else:  # No description
            # Fixed weights: title, abstract, tech_area
            base_weights = [0.30, 0.40, 0.30]
        
        # Apply quality adjustments to weights (not to scores)
        adjusted_weights = []
        for i, score in enumerate(scores):
            weight = base_weights[i]
            
            # Reduce weight for very low scores
            if score < 0.3:
                weight *= 0.7
            elif score < 0.5:
                weight *= 0.9
            # Increase weight for high scores
            elif score > 0.8:
                weight *= 1.1
            
            adjusted_weights.append(weight)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(adjusted_weights)
        normalized_weights = [w / total_weight for w in adjusted_weights]
        
        # Calculate weighted sum
        weighted_sum = sum(score * weight for score, weight in zip(scores, normalized_weights))
        
        # Apply overall quality penalty for very low scores
        if any(score < 0.3 for score in scores):
            weighted_sum *= 0.85  # 15% penalty for very low scores
        
        # Convert to percentage (0-100)
        return max(0.0, min(100.0, weighted_sum * 100))
    
    def _generate_factors(self, patent_info: PatentInfo, title_score: float, abstract_score: float, 
                         description_score: Optional[float], tech_area_score: float) -> List[str]:
        """Generate key factors for the analysis with improved specificity"""
        factors = []
        
        # Title factors with more specific feedback
        if title_score >= 0.8:
            factors.append("✅ Clear and descriptive title with technical specificity")
        elif title_score >= 0.6:
            factors.append("⚠️  Title is adequate but could be more specific")
        elif title_score >= 0.4:
            factors.append("⚠️  Title needs improvement - add technical terms and specificity")
        else:
            factors.append("❌ Title needs significant improvement - too vague or lacks technical content")
        
        # Abstract factors with detailed feedback
        if abstract_score >= 0.8:
            factors.append("✅ Well-defined problem and solution with technical depth")
        elif abstract_score >= 0.6:
            factors.append("⚠️  Abstract is good but could include more technical details")
        elif abstract_score >= 0.4:
            factors.append("⚠️  Abstract needs improvement - strengthen problem-solution structure")
        else:
            factors.append("❌ Abstract needs significant improvement - lacks technical depth and structure")
        
        # Description factors with specific guidance
        if description_score is not None:
            if description_score >= 0.8:
                factors.append("✅ Comprehensive technical description with implementation details")
            elif description_score >= 0.6:
                factors.append("⚠️  Description is good but could include more implementation details")
            elif description_score >= 0.4:
                factors.append("⚠️  Description needs improvement - add more technical specifications")
            else:
                factors.append("❌ Description needs significant improvement - too brief or lacks technical depth")
        else:
            factors.append("ℹ️  No detailed description provided - consider adding technical specifications")
        
        # Technology area factors with market context
        if tech_area_score >= 0.8:
            factors.append("✅ High-growth technology area with strong market potential")
        elif tech_area_score >= 0.6:
            factors.append("⚠️  Moderate technology potential - consider market positioning")
        elif tech_area_score >= 0.4:
            factors.append("⚠️  Technology area may have limited potential - research market demand")
        else:
            factors.append("❌ Technology area may have limited potential - consider alternative applications")
        
        return factors
    
    def _generate_recommendations(self, patent_info: PatentInfo, probability: float) -> List[str]:
        """Generate recommendations based on probability and patent info with improved specificity"""
        recommendations = []
        
        if probability >= 85:
            recommendations.append("Excellent potential - consider filing within 2-3 months")
            recommendations.append("Strengthen claims around core innovation and unique features")
            recommendations.append("Consider international patent filing for broader protection")
        elif probability >= 70:
            recommendations.append("Good potential - consider filing within 3-6 months")
            recommendations.append("Address prior art concerns and strengthen novelty arguments")
            recommendations.append("Conduct thorough competitive analysis before filing")
        elif probability >= 50:
            recommendations.append("Moderate potential - consider filing within 6-12 months")
            recommendations.append("Improve technical specifications and novelty aspects")
            recommendations.append("Conduct comprehensive prior art search")
        elif probability >= 30:
            recommendations.append("Limited potential - consider improving invention before filing")
            recommendations.append("Strengthen technical implementation and market differentiation")
            recommendations.append("Consider alternative patent strategies or trade secrets")
        else:
            recommendations.append("Low potential - significant improvements needed before filing")
            recommendations.append("Consider redesigning core innovation or finding new applications")
            recommendations.append("Evaluate whether patent protection is the right strategy")
        
        # Technology-specific recommendations
        if patent_info.tech_area:
            tech_area_lower = patent_info.tech_area.lower()
            
            if "ai/ml" in tech_area_lower:
                recommendations.append("Focus on unique ML algorithms and data processing methods")
                recommendations.append("Emphasize technical implementation details and performance metrics")
            elif "software" in tech_area_lower:
                recommendations.append("Emphasize technical implementation and user benefits")
                recommendations.append("Include detailed system architecture and workflow diagrams")
            elif "hardware" in tech_area_lower:
                recommendations.append("Include detailed component specifications and manufacturing processes")
                recommendations.append("Emphasize physical implementation and material properties")
            elif "iot" in tech_area_lower:
                recommendations.append("Focus on connectivity protocols and sensor integration")
                recommendations.append("Emphasize real-time data processing and system reliability")
            elif "blockchain" in tech_area_lower:
                recommendations.append("Focus on consensus mechanisms and cryptographic methods")
                recommendations.append("Emphasize security features and distributed architecture")
            elif "cybersecurity" in tech_area_lower:
                recommendations.append("Focus on threat detection and prevention mechanisms")
                recommendations.append("Emphasize security protocols and vulnerability mitigation")
        
        # Quality-specific recommendations
        if patent_info.title and len(patent_info.title.split()) < 3:
            recommendations.append("Expand title to include technical specificity and application context")
        
        if patent_info.abstract and len(patent_info.abstract.split()) < 50:
            recommendations.append("Expand abstract to include problem statement, solution approach, and benefits")
        
        if patent_info.description and len(patent_info.description.split()) < 200:
            recommendations.append("Add detailed technical description including implementation steps and system architecture")
        
        return recommendations
    
    def _generate_analysis(self, patent_info: PatentInfo, probability: float) -> str:
        """Generate analysis text with improved specificity"""
        analysis = f"Your invention shows "
        
        if probability >= 85:
            analysis += "excellent potential with strong innovation, clear technical implementation, and high market relevance."
        elif probability >= 70:
            analysis += "strong potential with good innovation and market relevance, with room for minor improvements."
        elif probability >= 50:
            analysis += "moderate potential with some innovative aspects, but needs refinement in key areas."
        elif probability >= 30:
            analysis += "limited potential that requires significant improvements in technical implementation and market positioning."
        else:
            analysis += "low potential that needs substantial redesign or reconsideration of the core concept."
        
        if patent_info.tech_area:
            analysis += f" The {patent_info.tech_area} focus "
            if probability >= 70:
                analysis += "aligns well with current market trends and has strong growth potential."
            elif probability >= 50:
                analysis += "has moderate market potential but may face competitive challenges."
            else:
                analysis += "may face market challenges and should be carefully evaluated against alternatives."
        
        # Add specific improvement suggestions
        if probability < 70:
            analysis += " Consider focusing on unique technical aspects and strengthening the novelty claims."
        
        return analysis

class PatentSearcher:
    """Search for patents in the database"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def search_patents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for patents by technology/topic"""
        try:
            # This would integrate with your existing patent database
            # For now, return mock results
            return self._get_mock_search_results(query, limit)
        except Exception as e:
            self.logger.error(f"Error searching patents: {e}")
            return []
    
    def _get_mock_search_results(self, query: str, limit: int) -> List[Dict]:
        """Get mock search results for demonstration"""
        # Mock data based on common patent types
        mock_patents = [
            {
                "id": "US12345678",
                "title": "Neural Network Optimization System",
                "tech_area": "AI/ML",
                "status": "PENDING",
                "brief_description": "Advanced neural network architecture for optimization problems with improved training algorithms."
            },
            {
                "id": "US12345679", 
                "title": "ML-Based Predictive Analytics",
                "tech_area": "AI/ML",
                "status": "ACCEPTED",
                "brief_description": "Machine learning system for predictive data analysis with real-time processing capabilities."
            },
            {
                "id": "US12345680",
                "title": "Deep Learning Image Recognition",
                "tech_area": "AI/ML", 
                "status": "PENDING",
                "brief_description": "Computer vision system using deep learning algorithms for image classification and object detection."
            },
            {
                "id": "US12345681",
                "title": "IoT Smart Home Controller",
                "tech_area": "IoT",
                "status": "ACCEPTED",
                "brief_description": "Internet of Things system for smart home automation with wireless sensor integration."
            },
            {
                "id": "US12345682",
                "title": "Blockchain Data Verification",
                "tech_area": "Blockchain",
                "status": "PENDING", 
                "brief_description": "Distributed ledger technology for secure data verification and tamper-proof record keeping."
            }
        ]
        
        # Filter based on query
        query_lower = query.lower()
        filtered_patents = []
        
        for patent in mock_patents:
            if (query_lower in patent["title"].lower() or 
                query_lower in patent["tech_area"].lower() or
                query_lower in patent["brief_description"].lower()):
                filtered_patents.append(patent)
        
        return filtered_patents[:limit]

class PatentAnalyzer:
    """Main patent analysis orchestrator"""
    
    def __init__(self):
        self.probability_calculator = PatentProbabilityCalculator()
        self.patent_searcher = PatentSearcher()
        self.logger = logging.getLogger(__name__)
    
    def analyze_existing_patent(self, patent_id: str) -> AnalysisResult:
        """Analyze existing patent without probability"""
        try:
            # This would integrate with your existing patent database
            patent_data = self._get_mock_patent_data(patent_id)
            
            key_factors = [
                "✅ Strong technical novelty in neural network optimization",
                "✅ Experienced inventor team with 5+ previous patents", 
                "✅ Well-documented implementation with detailed claims",
                "⚠️  Moderate prior art overlap in similar ML systems"
            ]
            
            analysis = f"""This patent demonstrates strong innovation in neural network architecture, with comprehensive technical documentation and experienced inventors. The moderate prior art overlap suggests the need for careful claim drafting to distinguish from existing solutions."""
            
            return AnalysisResult(
                key_factors=key_factors,
                analysis=analysis
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing existing patent: {e}")
            return AnalysisResult(
                key_factors=["❌ Error retrieving patent data"],
                analysis="Unable to analyze patent due to data retrieval error."
            )
    
    def analyze_new_invention(self, patent_info: PatentInfo) -> AnalysisResult:
        """Analyze new invention with probability"""
        try:
            probability_result = self.probability_calculator.calculate_probability(patent_info)
            
            return AnalysisResult(
                probability=probability_result["overall_probability"],
                key_factors=probability_result["factors"],
                analysis=probability_result["analysis"],
                recommendations=probability_result["recommendations"]
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing new invention: {e}")
            return AnalysisResult(
                key_factors=["❌ Error analyzing invention"],
                analysis="Unable to analyze invention due to processing error."
            )
    
    def search_patents(self, query: str) -> AnalysisResult:
        """Search patents with brief details"""
        try:
            search_results = self.patent_searcher.search_patents(query)
            
            return AnalysisResult(
                search_results=search_results
            )
            
        except Exception as e:
            self.logger.error(f"Error searching patents: {e}")
            return AnalysisResult(
                search_results=[]
            )
    
    def _get_mock_patent_data(self, patent_id: str) -> Dict:
        """Get mock patent data for demonstration"""
        return {
            "id": patent_id,
            "title": "Neural Network Optimization System",
            "abstract": "Advanced neural network architecture for optimization problems.",
            "status": "PENDING",
            "tech_area": "AI/ML"
        } 