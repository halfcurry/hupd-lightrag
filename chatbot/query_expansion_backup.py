#!/usr/bin/env python3
"""
Query Expansion and Synonym Management for Patent Search

This module handles intelligent query expansion, synonym mappings, and 
abbreviation resolution to improve patent search results.
"""

import logging
from typing import List, Dict, Set, Optional
import re

logger = logging.getLogger(__name__)

class QueryExpansion:
    """
    Handles query expansion and synonym management for patent searches
    """
    
    def __init__(self):
        # Comprehensive technology synonym mappings
        self.TECHNOLOGY_SYNONYMS = {
            # Internet of Things
            "iot": ["internet of things", "connected devices", "smart devices", "wireless sensors", 
                   "smart home", "industrial iot", "iiot", "edge computing", "sensor networks"],
            
            # Machine Learning & AI
            "machine learning": ["ml", "ai", "artificial intelligence", "deep learning", "neural networks",
                               "predictive analytics", "data science", "statistical learning"],
            "ai": ["artificial intelligence", "machine learning", "ml", "deep learning", "neural networks"],
            "ml": ["machine learning", "artificial intelligence", "ai", "deep learning"],
            
            # Blockchain & Cryptocurrency
            "blockchain": ["distributed ledger", "dlt", "cryptocurrency", "bitcoin technology", 
                          "smart contracts", "decentralized", "web3", "defi"],
            "cryptocurrency": ["bitcoin", "ethereum", "blockchain", "digital currency", "crypto"],
            
            # Wireless & Communications
            "5g": ["fifth generation", "5th generation", "next generation wireless", "5g network",
                   "mobile broadband", "wireless communication"],
            "wifi": ["wireless fidelity", "wireless networking", "802.11", "wireless lan"],
            
            # Cloud Computing
            "cloud computing": ["cloud", "saas", "software as a service", "virtualization", 
                              "distributed computing", "edge computing", "fog computing"],
            "saas": ["software as a service", "cloud computing", "web applications"],
            
            # Data & Analytics
            "big data": ["data analytics", "data mining", "business intelligence", "analytics",
                        "data science", "predictive analytics", "data warehouse"],
            "data analytics": ["analytics", "business intelligence", "data mining", "big data"],
            
            # Security
            "cybersecurity": ["security", "information security", "network security", "cyber security",
                            "data protection", "privacy", "encryption", "authentication"],
            
            # Quantum Computing
            "quantum computing": ["quantum", "quantum algorithms", "quantum cryptography", 
                                "quantum mechanics", "qubits", "quantum gates"],
            
            # Augmented/Virtual Reality
            "augmented reality": ["ar", "mixed reality", "virtual reality", "vr", "extended reality", "xr"],
            "virtual reality": ["vr", "augmented reality", "ar", "mixed reality", "extended reality"],
            "ar": ["augmented reality", "mixed reality", "virtual reality"],
            "vr": ["virtual reality", "augmented reality", "mixed reality"],
            
            # Autonomous Vehicles
            "autonomous vehicles": ["self-driving", "driverless", "autonomous cars", "tesla technology",
                                  "autonomous driving", "adas", "advanced driver assistance"],
            "self-driving": ["autonomous vehicles", "driverless", "autonomous cars", "autonomous driving"],
            
            # Robotics
            "robotics": ["automation", "industrial robots", "service robots", "automated systems",
                        "mechatronics", "control systems"],
            
            # Biotechnology
            "biotechnology": ["bio", "genetic engineering", "dna", "genomics", "proteomics",
                            "bioinformatics", "synthetic biology"],
            
            # Renewable Energy
            "solar energy": ["photovoltaic", "solar panels", "renewable energy", "clean energy"],
            "wind energy": ["wind power", "wind turbines", "renewable energy", "clean energy"],
            
            # Electric Vehicles
            "electric vehicles": ["ev", "electric cars", "battery electric", "hybrid vehicles"],
            "ev": ["electric vehicles", "electric cars", "battery electric"],
            
            # Internet & Web
            "web3": ["blockchain", "decentralized web", "cryptocurrency", "defi", "nft"],
            "api": ["application programming interface", "web services", "integration"],
        }
        
        # Common abbreviations and their full forms
        self.ABBREVIATIONS = {
            "iot": "internet of things",
            "ai": "artificial intelligence", 
            "ml": "machine learning",
            "ar": "augmented reality",
            "vr": "virtual reality",
            "ev": "electric vehicles",
            "api": "application programming interface",
            "saas": "software as a service",
            "bi": "business intelligence",
            "crm": "customer relationship management",
            "erp": "enterprise resource planning",
            "iot": "internet of things",
            "iiot": "industrial internet of things",
            "5g": "fifth generation wireless",
            "4g": "fourth generation wireless",
            "3g": "third generation wireless",
            "wifi": "wireless fidelity",
            "bluetooth": "wireless personal area network",
            "nfc": "near field communication",
            "rfid": "radio frequency identification",
            "gps": "global positioning system",
            "led": "light emitting diode",
            "oled": "organic light emitting diode",
            "lcd": "liquid crystal display",
            "cpu": "central processing unit",
            "gpu": "graphics processing unit",
            "ram": "random access memory",
            "rom": "read only memory",
            "ssd": "solid state drive",
            "hdd": "hard disk drive",
            "usb": "universal serial bus",
            "hdmi": "high definition multimedia interface",
            "vga": "video graphics array",
            "dvi": "digital visual interface",
            "pci": "peripheral component interconnect",
            "sata": "serial advanced technology attachment",
            "nvme": "non-volatile memory express",
            "dna": "deoxyribonucleic acid",
            "rna": "ribonucleic acid",
            "pcr": "polymerase chain reaction",
            "crispr": "clustered regularly interspaced short palindromic repeats",
        }
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand a search query with synonyms and related terms
        
        Args:
            query: Original search query
            
        Returns:
            List of expanded search terms
        """
        query_lower = query.lower().strip()
        expanded_terms = [query]  # Always include original query
        
        # Check for direct matches in synonyms
        if query_lower in self.TECHNOLOGY_SYNONYMS:
            expanded_terms.extend(self.TECHNOLOGY_SYNONYMS[query_lower])
        
        # Check for partial matches (query contains synonym key)
        for key, synonyms in self.TECHNOLOGY_SYNONYMS.items():
            if key in query_lower and key != query_lower:
                expanded_terms.extend(synonyms)
        
        # Check for abbreviation expansion
        if query_lower in self.ABBREVIATIONS:
            expanded_terms.append(self.ABBREVIATIONS[query_lower])
        
        # Check if query contains abbreviations
        for abbrev, full_form in self.ABBREVIATIONS.items():
            if abbrev in query_lower and abbrev != query_lower:
                expanded_terms.append(full_form)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_terms.append(term)
        
        logger.info(f"Query '{query}' expanded to {len(unique_terms)} terms: {unique_terms}")
        return unique_terms
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess query to handle common abbreviations and variations
        
        Args:
            query: Original query
            
        Returns:
            Preprocessed query
        """
        query_lower = query.lower()
        
        # Handle common abbreviations
        if query_lower == "iot":
            return "internet of things"
        elif query_lower == "ai":
            return "artificial intelligence"
        elif query_lower == "ml":
            return "machine learning"
        elif query_lower == "ar":
            return "augmented reality"
        elif query_lower == "vr":
            return "virtual reality"
        elif query_lower == "ev":
            return "electric vehicles"
        elif query_lower == "5g":
            return "fifth generation wireless"
        elif query_lower == "4g":
            return "fourth generation wireless"
        
        # Handle common phrases
        if "internet of things" in query_lower:
            return query.replace("internet of things", "IoT").replace("Internet of Things", "IoT")
        elif "artificial intelligence" in query_lower:
            return query.replace("artificial intelligence", "AI").replace("Artificial Intelligence", "AI")
        elif "machine learning" in query_lower:
            return query.replace("machine learning", "ML").replace("Machine Learning", "ML")
        
        return query
    
    def get_related_terms(self, query: str) -> List[str]:
        """
        Get related terms for a query (broader, narrower, and related concepts)
        
        Args:
            query: Search query
            
        Returns:
            List of related terms
        """
        query_lower = query.lower()
        related_terms = []
        
        # Add broader terms
        if "iot" in query_lower or "internet of things" in query_lower:
            related_terms.extend(["connected devices", "smart home", "industrial automation"])
        elif "ai" in query_lower or "machine learning" in query_lower:
            related_terms.extend(["data science", "predictive analytics", "neural networks"])
        elif "blockchain" in query_lower:
            related_terms.extend(["cryptocurrency", "smart contracts", "decentralized systems"])
        elif "5g" in query_lower:
            related_terms.extend(["wireless communication", "mobile broadband", "network infrastructure"])
        elif "cloud" in query_lower:
            related_terms.extend(["distributed computing", "virtualization", "web services"])
        
        return related_terms
    
    def smart_search_terms(self, query: str) -> List[str]:
        """
        Generate comprehensive search terms for a query
        
        Args:
            query: Original search query
            
        Returns:
            List of search terms to try
        """
        # Extract key terms from the query
        key_terms = self._extract_key_terms(query)
        
        # Start with the original query
        search_terms = [query]
        
        # Add key terms
        search_terms.extend(key_terms)
        
        # Expand key terms with synonyms
        for term in key_terms:
            expanded = self.expand_query(term)
            search_terms.extend(expanded)
        
        # Add related terms for key concepts
        related_terms = []
        for term in key_terms:
            related = self.get_related_terms(term)
            related_terms.extend(related)
        
        # Combine all terms
        all_terms = search_terms + related_terms
        
        # Remove duplicates and limit to reasonable number
        seen = set()
        unique_terms = []
        for term in all_terms:
            if term.lower() not in seen and len(unique_terms) < 10:  # Limit to 10 terms
                seen.add(term.lower())
                unique_terms.append(term)
        
        logger.info(f"Smart search terms for '{query}': {unique_terms}")
        return unique_terms
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key technology terms from a query
        
        Args:
            query: Original search query
            
        Returns:
            List of key technology terms
        """
        query_lower = query.lower()
        key_terms = []
        
        # Define technology keywords to look for
        tech_keywords = [
            # Robotics and Automation
            "robot", "robotic", "automation", "automated", "autonomous",
            
            # AI and ML
            "ai", "artificial intelligence", "machine learning", "ml", "deep learning", "neural network",
            
            # IoT and Connected Systems
            "iot", "internet of things", "connected", "smart", "wireless", "sensor",
            
            # Blockchain
            "blockchain", "cryptocurrency", "bitcoin", "ethereum", "smart contract",
            
            # Wireless and Communications
            "5g", "4g", "wifi", "bluetooth", "wireless", "communication",
            
            # Cloud and Computing
            "cloud", "computing", "saas", "api", "software", "application",
            
            # Data and Analytics
            "data", "analytics", "big data", "predictive", "analysis",
            
            # Security
            "security", "cybersecurity", "encryption", "authentication",
            
            # Quantum
            "quantum", "quantum computing", "qubit",
            
            # AR/VR
            "ar", "augmented reality", "vr", "virtual reality", "mixed reality",
            
            # Autonomous Vehicles
            "autonomous", "self-driving", "driverless", "tesla",
            
            # Biotechnology
            "bio", "biotechnology", "dna", "genetic", "crispr",
            
            # Renewable Energy
            "solar", "wind", "renewable", "clean energy", "green",
            
            # Electric Vehicles
            "electric", "ev", "battery", "hybrid",
            
            # Web3
            "web3", "defi", "nft", "decentralized"
        ]
        
        # Find matching keywords in the query
        for keyword in tech_keywords:
            if keyword in query_lower:
                key_terms.append(keyword)
        
        # If no specific tech keywords found, try to extract meaningful terms
        if not key_terms:
            # Split query into words and look for meaningful terms
            words = query_lower.split()
            for word in words:
                if len(word) > 3 and word not in ['tell', 'me', 'about', 'patents', 'on', 'using', 'with', 'the', 'and', 'or', 'for']:
                    key_terms.append(word)
        
        # Remove duplicates
        key_terms = list(set(key_terms))
        
        logger.info(f"Extracted key terms from '{query}': {key_terms}")
        return key_terms

# Global instance for easy access
query_expander = QueryExpansion() 