import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any

print = c.print

class Tool:

    def forward(self,  
              path: str = './',
              start_anchor: str = 'blah blah',
              end_anchor: str = 'is up',
              mode: str = 'between',  # 'between', 'inclusive', 'exclusive'
              create_if_missing: bool = False,
              **kwargs) -> str:
        """
        Remove content from a file at various positions
        
        Args:
            path: Path to the file
            start_anchor: Starting anchor text
            end_anchor: Ending anchor text
            mode: How to remove content:
                - 'between': Remove content between anchors (keeping anchors)
                - 'inclusive': Remove content including the anchors
                - 'exclusive': Remove only content between anchors
            create_if_missing: Create file if it doesn't exist (usually False for removal)
        """
        path = os.path.abspath(path)
        
        # Check if file exists
        if not os.path.exists(path):
            if create_if_missing:
                c.print(f"Creating new file: {path}", color='yellow')
                c.write(path, '')
                return ''
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        # Read the file content
        text = c.text(path)
        
        # Check if anchors exist
        if start_anchor not in text or end_anchor not in text:
            c.print(f"Warning: One or both anchors not found. No changes made.", color='yellow')
            return text
        
        # Find positions of anchors
        start_pos = text.find(start_anchor)
        end_pos = text.find(end_anchor, start_pos)
        
        if end_pos <= start_pos:
            # End anchor comes before or at start anchor position
            c.print(f"Warning: End anchor not found after start anchor. No changes made.", color='yellow')
            return text
        
        # Handle different removal modes
        if mode == 'between':
            # Keep the anchors, remove content between them
            before_content = text[:start_pos + len(start_anchor)]
            after_content = text[end_pos:]
            result = before_content + after_content
        elif mode == 'inclusive':
            # Remove anchors and content between them
            before_content = text[:start_pos]
            after_content = text[end_pos + len(end_anchor):]
            result = before_content + after_content
        elif mode == 'exclusive':
            # Remove only content between anchors (same as 'between')
            before_content = text[:start_pos + len(start_anchor)]
            after_content = text[end_pos:]
            result = before_content + after_content
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'between', 'inclusive', or 'exclusive'")
        
        # Write the result back to the file
        c.write(path, result)
        c.print(f"Successfully removed content from file: {path}", color='green')
        
        return result