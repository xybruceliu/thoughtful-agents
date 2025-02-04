"""Text splitting utilities."""
from typing import List
import re

class SentenceTextSplitter:
    """Split text into sentences and paragraphs using punctuation and newlines as delimiters."""
    
    def __init__(self):
        """Initialize the text splitter."""
        # Define the regex pattern for splitting
        # Matches:
        # 1. Sentence endings (., !, ?, ;) followed by whitespace
        # 2. Double newlines (paragraph breaks)
        self.split_pattern = r'(?<=[.!?;])\s+|\n\s*\n'
        
    def split_text(self, text: str) -> List[str]:
        """Split text into sentences and paragraphs.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks split by sentence endings and paragraph breaks
        """
        if not text:
            return []
            
        # Normalize spaces first (replace multiple spaces with single space)
        text = re.sub(r'\s+', ' ', text.strip())
            
        # Split text using the pattern
        raw_chunks = re.split(self.split_pattern, text)
        
        # Clean up chunks:
        # 1. Strip whitespace
        # 2. Normalize spaces within each chunk
        # 3. Remove empty chunks
        chunks = []
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if chunk:
                # Normalize spaces within the chunk
                chunk = re.sub(r'\s+', ' ', chunk)
                chunks.append(chunk)
        
        return chunks 