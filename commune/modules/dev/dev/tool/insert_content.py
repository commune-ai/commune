import commune as c
import json
import os
from typing import List, Dict, Union, Optional, Any

print = c.print

class Tool:

    def forward(self,  
              path: str = './',
              content: str = 'hey',
              start_anchor: str = 'blah blah',
              end_anchor: str = 'is up',
              mode: str = 'between',  # 'between', 'after', 'before', 'append', 'prepend'
              create_if_missing: bool = True,
              **kwargs) -> str:
        """
        Insert content into a file at various positions
        
        Args:
            path: Path to the file
            content: Content to insert
            start_anchor: Starting anchor text
            end_anchor: Ending anchor text (used in 'between' mode)
            mode: Where to insert content:
                - 'between': Insert between start and end anchors
                - 'after': Insert after start_anchor
                - 'before': Insert before start_anchor
                - 'append': Append to end of file
                - 'prepend': Prepend to beginning of file
            create_if_missing: Create file if it doesn't exist
        """
        path = os.path.abspath(path)
        
        # Handle file creation if needed
        if not os.path.exists(path):
            if create_if_missing:
                c.print(f"Creating new file: {path}", color='yellow')
                c.write(path, '')
                text = ''
            else:
                raise FileNotFoundError(f"File not found: {path}")
        else:
            # Read the file content
            text = c.text(path)
        
        # Handle different insertion modes
        if mode == 'append':
            result = text + content
        elif mode == 'prepend':
            result = content + text
        elif mode == 'after':
            if start_anchor not in text:
                c.print(f"Warning: Start anchor '{start_anchor}' not found. Appending to end.", color='yellow')
                result = text + content
            else:
                pos = text.find(start_anchor) + len(start_anchor)
                result = text[:pos] + content + text[pos:]
        elif mode == 'before':
            if start_anchor not in text:
                c.print(f"Warning: Start anchor '{start_anchor}' not found. Prepending to beginning.", color='yellow')
                result = content + text
            else:
                pos = text.find(start_anchor)
                result = text[:pos] + content + text[pos:]
        elif mode == 'between':
            # Check if anchors exist
            if start_anchor not in text or end_anchor not in text:
                c.print(f"Warning: Anchors not found. Using fallback mode.", color='yellow')
                if start_anchor in text:
                    # Only start anchor found, insert after it
                    pos = text.find(start_anchor) + len(start_anchor)
                    result = text[:pos] + content + text[pos:]
                else:
                    # No anchors found, append to end
                    result = text + content
            else:
                # Find positions of anchors
                start_pos = text.find(start_anchor)
                end_pos = text.find(end_anchor, start_pos + len(start_anchor))
                
                if end_pos <= start_pos:
                    # End anchor comes before start anchor, try to find next occurrence
                    end_pos = text.find(end_anchor, start_pos + len(start_anchor))
                    if end_pos == -1:
                        c.print(f"Warning: End anchor after start anchor not found. Inserting after start anchor.", color='yellow')
                        pos = start_pos + len(start_anchor)
                        result = text[:pos] + content + text[pos:]
                    else:
                        # Insert between anchors
                        before_start = text[:start_pos + len(start_anchor)]
                        after_end = text[end_pos:]
                        result = before_start + content + after_end
                else:
                    # Normal case: insert between anchors
                    before_start = text[:start_pos + len(start_anchor)]
                    after_end = text[end_pos:]
                    result = before_start + content + after_end
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'between', 'after', 'before', 'append', or 'prepend'")
        
        # Write the result back to the file
        c.write(path, result)
        c.print(f"Successfully updated file: {path}", color='green')
        
        return result