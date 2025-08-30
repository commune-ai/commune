import os
import commune as c
def backup_file(path:str) -> str:
    """Create a backup of a file before modifying it"""
    backup_path = path + '.bak'
    if os.path.exists(path):
        with open(path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
        return backup_path
    return None
def add_lines(path:str, start_line:int, content:str):
    """Add lines to a file at specified position"""
    backup_file(path)
    text = c.get_text(path)
    lines = text.split('\n')
    lines = lines[:start_line] + [content] + lines[start_line:]
    text = '\n'.join(lines)
    c.put_text(path, text)
    assert c.get_text(path) == text, f'Failed to write to file: {path}'
    return path
def add_lines_after(path:str, after_text:str, content:str):
    """Add lines after a piece of text in a file"""
    backup_file(path)
    text = c.get_text(path)
    after_idx = text.find(after_text)
    if after_idx == -1:
        raise ValueError(f"Could not find after_text: {after_text}")
    
    # Insert the content after the piece of text
    new_text = text[:after_idx + len(after_text)] + "\n" + content + "\n" + text[after_idx + len(after_text):]
    c.put_text(path, new_text)
    assert c.get_text(path) == new_text, f'Failed to write to file: {path}'
    return path
def add_lines_before(path:str, before_text:str, content:str):
    """Add lines before a piece of text in a file"""
    backup_file(path)
    text = c.get_text(path)
    before_idx = text.find(before_text)
    if before_idx == -1:
        raise ValueError(f"Could not find before_text: {before_text}")
    
    # Insert the content before the piece of text
    new_text = text[:before_idx] + content + "\n" + text[before_idx:]
    c.put_text(path, new_text)
    assert c.get_text(path) == new_text, f'Failed to write to file: {path}'
    return path
def add_file(path:str, content:str):
    """Add a file with the specified content"""
    with open(path, 'w') as f:
        f.write(content)
    return path 
def add_between(path:str, before_text:str, after_text:str, content:str):
    """Add lines between two pieces of text in a file"""
    backup_file(path)
    text = c.get_text(path)
    before_idx = text.find(before_text)
    if before_idx == -1:
        raise ValueError(f"Could not find before_text: {before_text}")
    
    after_idx = text.find(after_text, before_idx + len(before_text))
    if after_idx == -1:
        raise ValueError(f"Could not find after_text: {after_text}")
    
    # Insert the content between the two pieces of text
    new_text = text[:after_idx] + "\n" + content + "\n" + text[after_idx:]
    c.put_text(path, new_text)
    assert c.get_text(path) == new_text, f'Failed to write to file: {path}'
    return path
        
def delete_lines(path:str, start_line:int, end_line:int):
    """Delete lines from a file between start_line and end_line"""
    backup_file(path)
    text = c.get_text(path)
    lines = text.split('\n')
    lines = lines[:start_line] + lines[end_line:]
    text = '\n'.join(lines)
    c.put_text(path, text)
    assert c.get_text(path) == text, f'Failed to write to file: {path}'
    return path
def delete_file(path:str):
    """Delete a file or directory"""
    backup_file(path)
    if os.path.exists(path):
        if os.path.isdir(path):
            os.rmdir(path)
        else:
            os.remove(path)
        return True
    return False
def delete_between(path:str, before_text:str, after_text:str):
    """Delete content between two pieces of text in a file"""
    backup_file(path)
    text = c.get_text(path)
    before_idx = text.find(before_text)
    if before_idx == -1:
        raise ValueError(f"Could not find before_text: {before_text}")
    
    after_idx = text.find(after_text, before_idx + len(before_text))
    if after_idx == -1:
        raise ValueError(f"Could not find after_text: {after_text}")
    
    # Delete the content between the two pieces of text
    new_text = text[:before_idx + len(before_text)] + text[after_idx:]
    c.put_text(path, new_text)
    assert c.get_text(path) == new_text, f'Failed to write to file: {path}'
    return path