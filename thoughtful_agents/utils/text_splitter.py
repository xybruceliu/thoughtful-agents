"""Text splitting utilities."""
from typing import List, Optional, Dict
import re

class RecursiveCharacterTextSplitter:
    """Split text recursively by different characters."""
    
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        """Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            separators: List of separators to split on, from high to low granularity.
                       Default: ["\n\n", "\n", " ", ""]
            keep_separator: Whether to keep the separator in the chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        self.keep_separator = keep_separator
        
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Return empty list for empty text
        if not text:
            return []
            
        # Return single chunk if text is short enough
        if len(text) <= self.chunk_size:
            return [text]
            
        # Try each separator in order
        for separator in self.separators:
            if separator == "":
                # Character-level splitting as fallback
                return self._split_by_character(text)
                
            if separator in text:
                return self._split_by_separator(text, separator)
                
        # Fallback to character-level splitting
        return self._split_by_character(text)
        
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks of appropriate size.
        
        Args:
            splits: List of text splits
            separator: Separator used to create the splits
            
        Returns:
            List of merged text chunks
        """
        # Initialize chunks and current chunk
        chunks = []
        current_chunk = []
        current_length = 0
        separator_len = len(separator) if self.keep_separator else 0
        
        for split in splits:
            split_len = len(split) + (separator_len if current_chunk else 0)
            
            # Add split to current chunk if it fits
            if current_length + split_len <= self.chunk_size:
                if current_chunk:
                    current_chunk.append(separator if self.keep_separator else "")
                current_chunk.append(split)
                current_length += split_len
            else:
                # Add current chunk to chunks if non-empty
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    
                # Handle splits longer than chunk_size
                if split_len > self.chunk_size:
                    split_chunks = self._split_by_character(split)
                    chunks.extend(split_chunks[:-1])
                    current_chunk = [split_chunks[-1]]
                    current_length = len(split_chunks[-1])
                else:
                    current_chunk = [split]
                    current_length = split_len
        
        # Add final chunk if non-empty
        if current_chunk:
            chunks.append("".join(current_chunk))
            
        # Add overlaps
        if self.chunk_overlap > 0:
            return self._add_overlaps(chunks)
            
        return chunks
        
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator and merge into chunks.
        
        Args:
            text: Text to split
            separator: Separator to split on
            
        Returns:
            List of text chunks
        """
        splits = text.split(separator)
        return self._merge_splits(splits, separator)
        
    def _split_by_character(self, text: str) -> List[str]:
        """Split text into chunks by character.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks
        
    def _add_overlaps(self, chunks: List[str]) -> List[str]:
        """Add overlaps between chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of text chunks with overlaps
        """
        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
                continue
                
            # Add overlap from previous chunk
            overlap_start = max(0, len(chunks[i-1]) - self.chunk_overlap)
            overlap = chunks[i-1][overlap_start:]
            
            # Only add non-empty overlaps
            if overlap.strip():
                result.append(overlap + chunk)
            else:
                result.append(chunk)
                
        return result 